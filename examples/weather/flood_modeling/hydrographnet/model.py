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
from jaxtyping import Float

from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.nn import get_activation
from physicsnemo.nn.module.gnn_layers.graph_types import GraphType
from physicsnemo.nn.module.gnn_layers.mesh_graph_mlp import MeshGraphMLP


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
        concat_endpoints: bool = False,
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

        # DUALFloodGNN-style edge head: when concat_endpoints=True the head
        # takes [h_u, h_v, e_uv] (3 * hidden_dim_processor) instead of just
        # e_uv (hidden_dim_processor). Couples edge predictions to the node
        # states whose volumes the LMC residual references.
        self.concat_endpoints = concat_endpoints
        edge_decoder_input_dim = (
            3 * hidden_dim_processor if concat_endpoints else hidden_dim_processor
        )
        self.edge_decoder = MeshGraphMLP(
            edge_decoder_input_dim,
            output_dim=1,
            hidden_dim=hidden_dim_node_decoder,
            hidden_layers=num_layers_node_decoder,
            activation_fn=activation_fn,
            norm_type=None,
            recompute_activation=recompute_activation,
        )

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
        if self.concat_endpoints:
            g_ref = graph[0] if isinstance(graph, list) else graph
            src, dst = g_ref.edge_index[0], g_ref.edge_index[1]
            edge_decoder_in = torch.cat(
                [node_features[src], node_features[dst], edge_features], dim=-1
            )
            edge_pred = self.edge_decoder(edge_decoder_in)
        else:
            edge_pred = self.edge_decoder(edge_features)
        return node_pred, edge_pred
