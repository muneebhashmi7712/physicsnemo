# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Tuple, Union

import torch
import torch.nn as nn
from jaxtyping import Float
from torch_geometric.utils import add_self_loops, to_undirected

from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.graph_types import GraphType
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP


class GraphLowPassFilter(nn.Module):
    """Symmetric-normalized adjacency low-pass filter on node features.

    Applies h' = (1 - alpha) * h + alpha * (A_norm)^K @ h
    where A_norm = D_hat^{-1/2} (A_sym + I) D_hat^{-1/2}, A_sym is the
    symmetrized edge_index (KNN graphs are directed), and D_hat counts
    self-loops. K iterations are applied as repeated sparse mat-vecs.

    alpha is parameterized via a sigmoid for stable [0,1] confinement.
    When learnable_alpha=False, raw_alpha is stored as a non-persistent
    buffer so state_dict keys remain identical to baseline runs.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_iterations: int = 1,
        alpha_init: float = 0.2,
        learnable_alpha: bool = False,
        alpha_per_channel: bool = False,
    ):
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.learnable_alpha = bool(learnable_alpha)
        self.alpha_per_channel = bool(alpha_per_channel)
        shape = (hidden_dim,) if alpha_per_channel else ()

        eps = 1e-6
        a = max(min(float(alpha_init), 1.0 - eps), eps)
        raw_init = float(torch.log(torch.tensor(a / (1.0 - a))).item())
        init_tensor = torch.full(shape, raw_init)

        if learnable_alpha:
            self.raw_alpha = nn.Parameter(init_tensor)
        else:
            self.register_buffer("raw_alpha", init_tensor, persistent=False)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_alpha)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.num_iterations <= 0:
            return h
        num_nodes = h.size(0)

        ei = to_undirected(edge_index, num_nodes=num_nodes)
        ei, _ = add_self_loops(ei, num_nodes=num_nodes)
        src, dst = ei[0], ei[1]

        deg = torch.zeros(num_nodes, device=h.device, dtype=h.dtype)
        ones = torch.ones(src.size(0), device=h.device, dtype=h.dtype)
        deg.scatter_add_(0, src, ones)
        deg_inv_sqrt = deg.clamp(min=1.0).rsqrt()
        edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        smoothed = h
        for _ in range(self.num_iterations):
            msg = smoothed[src] * edge_weight.unsqueeze(-1)
            agg = torch.zeros_like(smoothed)
            agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
            smoothed = agg

        a = self.alpha.to(h.dtype)
        if a.dim() == 0:
            return (1.0 - a) * h + a * smoothed
        return (1.0 - a).unsqueeze(0) * h + a.unsqueeze(0) * smoothed


class HydroGraphKANWithEdgeDecoder(MeshGraphKAN):
    """MeshGraphKAN with an additional edge decoder for per-edge flow predictions.

    Adds a small MLP edge decoder that takes the final edge embeddings from the
    processor and outputs a scalar per edge (predicted change in flow, delta-Q).
    The forward method returns both node predictions and edge predictions.
    """

    def __init__(
        self,
        input_dim_nodes: int,
        input_dim_edges: int,
        output_dim: int,
        processor_size: int = 15,
        mlp_activation_fn: str = "relu",
        num_layers_node_processor: int = 2,
        num_layers_edge_processor: int = 2,
        hidden_dim_processor: int = 128,
        hidden_dim_node_encoder: int = 128,
        num_layers_node_encoder: Union[int, None] = 2,
        hidden_dim_edge_encoder: int = 128,
        num_layers_edge_encoder: Union[int, None] = 2,
        hidden_dim_node_decoder: int = 128,
        num_layers_node_decoder: Union[int, None] = 2,
        aggregation: Literal["sum", "mean"] = "sum",
        do_concat_trick: bool = False,
        num_processor_checkpoint_segments: int = 0,
        checkpoint_offloading: bool = False,
        recompute_activation: bool = False,
        num_harmonics: int = 5,
        lowpass_enabled: bool = False,
        lowpass_iterations: int = 1,
        lowpass_alpha_init: float = 0.2,
        lowpass_learnable_alpha: bool = False,
        lowpass_alpha_per_channel: bool = False,
    ):
        super().__init__(
            input_dim_nodes=input_dim_nodes,
            input_dim_edges=input_dim_edges,
            output_dim=output_dim,
            processor_size=processor_size,
            mlp_activation_fn=mlp_activation_fn,
            num_layers_node_processor=num_layers_node_processor,
            num_layers_edge_processor=num_layers_edge_processor,
            hidden_dim_processor=hidden_dim_processor,
            hidden_dim_node_encoder=hidden_dim_node_encoder,
            num_layers_node_encoder=num_layers_node_encoder,
            hidden_dim_edge_encoder=hidden_dim_edge_encoder,
            num_layers_edge_encoder=num_layers_edge_encoder,
            hidden_dim_node_decoder=hidden_dim_node_decoder,
            num_layers_node_decoder=num_layers_node_decoder,
            aggregation=aggregation,
            do_concat_trick=do_concat_trick,
            num_processor_checkpoint_segments=num_processor_checkpoint_segments,
            checkpoint_offloading=checkpoint_offloading,
            recompute_activation=recompute_activation,
            num_harmonics=num_harmonics,
        )

        activation_fn = get_activation(mlp_activation_fn)

        self.edge_decoder = MeshGraphMLP(
            hidden_dim_processor,
            output_dim=1,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

        if lowpass_enabled:
            self.lowpass = GraphLowPassFilter(
                hidden_dim=hidden_dim_processor,
                num_iterations=lowpass_iterations,
                alpha_init=lowpass_alpha_init,
                learnable_alpha=lowpass_learnable_alpha,
                alpha_per_channel=lowpass_alpha_per_channel,
            )
        else:
            self.lowpass = None

    def forward(
        self,
        node_features: Float[torch.Tensor, "num_nodes input_dim_nodes"],
        edge_features: Float[torch.Tensor, "num_edges input_dim_edges"],
        graph: Union[GraphType, list[GraphType]],
        **kwargs,
    ) -> Tuple[
        Float[torch.Tensor, "num_nodes output_dim"],
        Float[torch.Tensor, "num_edges 1"],
    ]:
        # Encode
        edge_features = self.edge_encoder(edge_features)
        node_features = self.node_encoder(node_features)

        if self.lowpass is not None:
            node_features = self.lowpass(node_features, graph.edge_index)

        # Process — replicate processor.forward() to capture both node and edge embeddings
        with self.processor.checkpoint_offload_ctx:
            for seg_start, seg_end in self.processor.checkpoint_segments:
                edge_features, node_features = self.processor.checkpoint_fn(
                    self.processor._run_function(seg_start, seg_end),
                    node_features,
                    edge_features,
                    graph,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )

        # Decode both node and edge predictions
        node_pred = self.node_decoder(node_features)
        edge_pred = self.edge_decoder(edge_features)
        return node_pred, edge_pred
