#!/usr/bin/env python3
"""Measure LMC value and output-gradient shares by UrbanFlood node group.

This reproduces the one-step + pushforward-stability LMC branches used during
training. It reports the actual loss (with GT connection-flow substitution) and
a diagnostic counterfactual that leaves model-predicted connection flow in the
same unchanged mass-balance residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset
from physicsnemo.utils import load_checkpoint
from model import HydroGraphKANWithEdgeDecoder
from utils import compute_local_conservation_loss


GROUPS = ("interior_2d", "inlet_2d", "manhole_1d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-name", choices=("Model_1", "Model_2"), required=True)
    parser.add_argument("--num-events", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", default="0,5,10,20,30,40,49")
    parser.add_argument("--sample-count", type=int, default=256)
    return parser.parse_args()


def build_model(device: torch.device) -> torch.nn.Module:
    model = HydroGraphKANWithEdgeDecoder(
        input_dim_nodes=20,
        input_dim_edges=10,
        output_dim=2,
        processor_size=5,
        hidden_dim_processor=64,
        hidden_dim_node_encoder=64,
        hidden_dim_edge_encoder=64,
        hidden_dim_node_decoder=64,
        num_layers_node_processor=1,
        num_layers_edge_processor=1,
        num_layers_edge_encoder=1,
        num_layers_node_decoder=1,
        num_harmonics=5,
        concat_endpoints=False,
    )
    return model.to(device)


def residual_terms(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    graph,
    *,
    use_gt_connection_flow: bool,
    smooth_l1_beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized residual, per-node loss, and connection-edge mask."""
    device = node_pred.device
    num_nodes = node_pred.shape[0]
    volume_std = graph.V_std_per_node.to(device)
    flow_sigma = graph.Q_sigma_per_edge_phys.to(device)

    delta_v_phys = node_pred[:, 1] * volume_std
    delta_q = edge_pred.squeeze(-1)
    conn_mask = graph.edge_attr[:, 9].to(device).bool()
    if use_gt_connection_flow:
        delta_q = delta_q.clone()
        delta_q[conn_mask] = graph.edge_y.squeeze(-1).to(device)[conn_mask]

    q_phys = (graph.edge_q_prev.to(device) + delta_q) * flow_sigma
    src, dst = graph.edge_index.to(device)
    q_in = torch.zeros(num_nodes, device=device, dtype=q_phys.dtype)
    q_out = torch.zeros(num_nodes, device=device, dtype=q_phys.dtype)
    q_in.scatter_add_(0, dst, q_phys)
    q_out.scatter_add_(0, src, q_phys)
    net_flow = (q_in - q_out) / 2.0
    source_phys = graph.source_term.to(device) * volume_std
    residual_norm = (delta_v_phys - net_flow - source_phys) / volume_std
    per_node = F.smooth_l1_loss(
        residual_norm,
        torch.zeros_like(residual_norm),
        beta=smooth_l1_beta,
        reduction="none",
    )
    return residual_norm, per_node, conn_mask


def group_masks(graph, device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    node_type = graph.node_type.to(device)
    keep = torch.ones_like(node_type, dtype=torch.bool)
    if hasattr(graph, "boundary_mask") and graph.boundary_mask is not None:
        keep &= ~graph.boundary_mask.to(device).bool()

    conn_edges = graph.edge_attr[:, 9].to(device).bool()
    conn_nodes = torch.unique(graph.edge_index[:, conn_edges].to(device))
    is_conn_node = torch.zeros_like(node_type, dtype=torch.bool)
    is_conn_node[conn_nodes] = True
    masks = {
        "interior_2d": keep & (node_type == 0) & ~is_conn_node,
        "inlet_2d": keep & (node_type == 0) & is_conn_node,
        "manhole_1d": keep & (node_type == 1),
    }
    covered = masks["interior_2d"] | masks["inlet_2d"] | masks["manhole_1d"]
    if not torch.equal(covered, keep):
        raise AssertionError("node groups do not partition the kept LMC nodes")
    return masks, keep


def grad_norms(
    group_loss: torch.Tensor,
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    graph,
) -> dict[str, float]:
    grads = torch.autograd.grad(
        group_loss,
        outputs,
        retain_graph=True,
        allow_unused=True,
    )
    node_grads = [grads[0], grads[2]]
    edge_grads = [grads[1], grads[3]]

    def combined_l2(parts: list[torch.Tensor | None]) -> float:
        total = torch.zeros((), device=group_loss.device, dtype=torch.float64)
        for part in parts:
            if part is not None:
                total += part.to(torch.float64).square().sum()
        return float(total.sqrt().item())

    result = {
        "node_output_grad_l2": combined_l2(node_grads),
        "edge_output_grad_l2": combined_l2(edge_grads),
    }
    edge_type_masks = {
        "2d": graph.edge_attr[:, 7].to(group_loss.device).bool(),
        "1d": graph.edge_attr[:, 8].to(group_loss.device).bool(),
        "conn": graph.edge_attr[:, 9].to(group_loss.device).bool(),
    }
    for edge_type, mask in edge_type_masks.items():
        parts = [None if grad is None else grad[mask] for grad in edge_grads]
        result[f"edge_{edge_type}_grad_l2"] = combined_l2(parts)
    return result


def select_indices(dataset: UrbanFloodDataset, count: int) -> list[int]:
    count = min(count, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=np.int64)
    selected = sorted(set(int(i) for i in indices))
    if len(selected) != count:
        raise AssertionError(f"wanted {count} unique samples, got {len(selected)}")
    return selected


def summarize(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        buckets[(int(row["epoch"]), str(row["group"]))].append(row)

    out = []
    numeric = (
        "nodes",
        "actual_group_mean",
        "actual_contribution",
        "counterfactual_group_mean",
        "counterfactual_contribution",
        "actual_node_output_grad_l2",
        "actual_edge_output_grad_l2",
        "actual_edge_2d_grad_l2",
        "actual_edge_1d_grad_l2",
        "actual_edge_conn_grad_l2",
        "counterfactual_node_output_grad_l2",
        "counterfactual_edge_output_grad_l2",
        "counterfactual_edge_2d_grad_l2",
        "counterfactual_edge_1d_grad_l2",
        "counterfactual_edge_conn_grad_l2",
    )
    for (epoch, group), rows in sorted(buckets.items()):
        item: dict[str, object] = {
            "epoch": epoch,
            "group": group,
            "samples": len(rows),
        }
        for key in numeric:
            item[f"mean_{key}"] = float(np.mean([float(row[key]) for row in rows]))
        out.append(item)

    by_epoch: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in out:
        by_epoch[int(row["epoch"])].append(row)
    for epoch_rows in by_epoch.values():
        actual_total = sum(float(row["mean_actual_contribution"]) for row in epoch_rows)
        cf_total = sum(
            float(row["mean_counterfactual_contribution"]) for row in epoch_rows
        )
        actual_grad_total = sum(
            float(row["mean_actual_node_output_grad_l2"])
            + float(row["mean_actual_edge_output_grad_l2"])
            for row in epoch_rows
        )
        cf_grad_total = sum(
            float(row["mean_counterfactual_node_output_grad_l2"])
            + float(row["mean_counterfactual_edge_output_grad_l2"])
            for row in epoch_rows
        )
        for row in epoch_rows:
            row["actual_loss_fraction_pct"] = (
                100.0 * float(row["mean_actual_contribution"]) / actual_total
            )
            row["counterfactual_loss_fraction_pct"] = (
                100.0 * float(row["mean_counterfactual_contribution"]) / cf_total
            )
            cf = float(row["mean_counterfactual_contribution"])
            row["gt_connection_substitution_effect_pct"] = (
                (float(row["mean_actual_contribution"]) / cf - 1.0) * 100.0
                if cf != 0.0
                else float("nan")
            )
            row["actual_output_grad_norm_fraction_pct"] = 100.0 * (
                float(row["mean_actual_node_output_grad_l2"])
                + float(row["mean_actual_edge_output_grad_l2"])
            ) / actual_grad_total
            row["counterfactual_output_grad_norm_fraction_pct"] = 100.0 * (
                float(row["mean_counterfactual_node_output_grad_l2"])
                + float(row["mean_counterfactual_edge_output_grad_l2"])
            ) / cf_grad_total
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = [int(value) for value in args.epochs.split(",")]
    output_dir = Path(args.output_dir) / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = UrbanFloodDataset(
        data_dir=args.data_dir,
        model_name=args.model_name,
        split="train",
        n_time_steps=2,
        noise_type="pushforward",
        noise_std=0.01,
        num_samples=args.num_events,
        return_physics=True,
        delta_t=300.0,
        use_1d=True,
        compute_boundary_mask=True,
        mask_inlets_in_lc=False,
        edge_q_prev_as_input=False,
        lmc_antisymmetric=False,
        train_rollout_length=1,
    )
    print(
        f"Dataset initialized in {time.perf_counter() - started_at:.1f}s; "
        f"selecting {args.sample_count} windows across {len(dataset)} samples.",
        flush=True,
    )
    selected = select_indices(dataset, args.sample_count)
    cached = []
    event_coverage = set()
    for sample_idx in selected:
        graph, _ = dataset[sample_idx]
        hydro_idx, time_idx = dataset.sample_index[sample_idx]
        event_id = str(dataset.dynamic_data[hydro_idx]["event_id"])
        event_coverage.add(event_id)
        cached.append((sample_idx, event_id, int(time_idx), graph))

    print(
        f"Cached {len(cached)} windows in {time.perf_counter() - started_at:.1f}s; "
        f"evaluating epochs {epochs}.",
        flush=True,
    )

    model = build_model(device)
    raw_rows: list[dict[str, object]] = []
    parity_deltas = []
    gt_residual_max = 0.0

    for epoch in epochs:
        epoch_started_at = time.perf_counter()
        loaded = load_checkpoint(
            args.checkpoint_dir,
            models=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            epoch=epoch,
            device=device,
        )
        if int(loaded) != epoch:
            raise AssertionError(f"requested epoch {epoch}, loaded {loaded}")
        model.eval()

        for sample_idx, event_id, time_idx, graph_cpu in cached:
            graph = graph_cpu.to(device)
            with torch.no_grad():
                x = graph.x
                n_time = 3  # n_time_steps=2 plus one pushforward column
                n_static = x.shape[1] - 2 * n_time
                static = x[:, :n_static]
                wd_full = x[:, n_static : n_static + n_time]
                vol_full = x[:, n_static + n_time : n_static + 2 * n_time]

                x_one = torch.cat([static, wd_full[:, 1:], vol_full[:, 1:]], dim=1)
                pred_one, edge_one = model(x_one, graph.edge_attr, graph)

                x_stab = torch.cat(
                    [static, wd_full[:, : n_time - 1], vol_full[:, : n_time - 1]],
                    dim=1,
                )
                pred_stab, _ = model(x_stab, graph.edge_attr, graph)
                wd_updated = torch.cat(
                    [wd_full[:, 1:2], wd_full[:, 1:2] + pred_stab[:, 0:1]], dim=1
                )
                vol_updated = torch.cat(
                    [vol_full[:, 1:2], vol_full[:, 1:2] + pred_stab[:, 1:2]], dim=1
                )
                x_stab2 = torch.cat([static, wd_updated, vol_updated], dim=1)
                pred_stab2, edge_stab2 = model(x_stab2, graph.edge_attr, graph)

            outputs = tuple(
                tensor.detach().requires_grad_(True)
                for tensor in (pred_one, edge_one, pred_stab2, edge_stab2)
            )
            n_one, e_one, n_stab, e_stab = outputs
            _, actual_one, _ = residual_terms(
                n_one, e_one, graph, use_gt_connection_flow=True
            )
            _, actual_stab, _ = residual_terms(
                n_stab, e_stab, graph, use_gt_connection_flow=True
            )
            _, cf_one, _ = residual_terms(
                n_one, e_one, graph, use_gt_connection_flow=False
            )
            _, cf_stab, _ = residual_terms(
                n_stab, e_stab, graph, use_gt_connection_flow=False
            )
            masks, keep = group_masks(graph, device)
            n_kept = int(keep.sum().item())

            # Independent parity gate against the production LMC function.
            production = compute_local_conservation_loss(
                n_one,
                e_one,
                graph,
                None,
                smooth_l1_beta=0.1,
                apply_boundary_mask=True,
                restrict_to_2d=False,
                node_type_weighting="none",
            ) + compute_local_conservation_loss(
                n_stab,
                e_stab,
                graph,
                None,
                smooth_l1_beta=0.1,
                apply_boundary_mask=True,
                restrict_to_2d=False,
                node_type_weighting="none",
            )
            reproduced = (actual_one[keep].sum() + actual_stab[keep].sum()) / n_kept
            parity_deltas.append(float((production - reproduced).abs().item()))

            # GT node/edge targets exactly define source_term, so this is a
            # data-pipeline consistency check rather than a model diagnostic.
            gt_residual, _, _ = residual_terms(
                graph.y.to(device),
                graph.edge_y.to(device),
                graph,
                use_gt_connection_flow=True,
            )
            gt_residual_max = max(
                gt_residual_max, float(gt_residual[keep].abs().max().item())
            )

            for group in GROUPS:
                mask = masks[group]
                nodes = int(mask.sum().item())
                if nodes == 0:
                    raise AssertionError(
                        f"sample {sample_idx} has no kept nodes in group {group}"
                    )
                actual_sum = actual_one[mask].sum() + actual_stab[mask].sum()
                cf_sum = cf_one[mask].sum() + cf_stab[mask].sum()
                group_loss = actual_sum / n_kept
                cf_group_loss = cf_sum / n_kept
                row: dict[str, object] = {
                    "epoch": epoch,
                    "sample_index": sample_idx,
                    "event_id": event_id,
                    "time_index": time_idx,
                    "group": group,
                    "nodes": nodes,
                    "kept_nodes": n_kept,
                    "actual_group_mean": float((actual_sum / nodes).item()),
                    "actual_contribution": float(group_loss.item()),
                    "counterfactual_group_mean": float((cf_sum / nodes).item()),
                    "counterfactual_contribution": float((cf_sum / n_kept).item()),
                }
                row.update(
                    {
                        f"actual_{key}": value
                        for key, value in grad_norms(
                            group_loss, outputs, graph
                        ).items()
                    }
                )
                row.update(
                    {
                        f"counterfactual_{key}": value
                        for key, value in grad_norms(
                            cf_group_loss, outputs, graph
                        ).items()
                    }
                )
                raw_rows.append(row)

        print(
            f"Epoch {epoch} diagnostic complete in "
            f"{time.perf_counter() - epoch_started_at:.1f}s "
            f"({time.perf_counter() - started_at:.1f}s total).",
            flush=True,
        )

    summary_rows = summarize(raw_rows)
    max_parity_delta = max(parity_deltas)
    if max_parity_delta > 1.0e-7:
        raise AssertionError(
            f"production LMC parity failed: max delta {max_parity_delta:.3e}"
        )
    if gt_residual_max > 1.0e-4:
        raise AssertionError(
            f"ground-truth mass-balance residual failed: {gt_residual_max:.3e}"
        )
    write_csv(output_dir / "teacher_forced_raw.csv", raw_rows)
    write_csv(output_dir / "teacher_forced_summary.csv", summary_rows)
    metadata = {
        "label": args.label,
        "model_name": args.model_name,
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": epochs,
        "sample_count": len(selected),
        "total_training_samples": len(dataset),
        "events_covered": len(event_coverage),
        "total_events": len(dataset.event_ids),
        "groups": list(GROUPS),
        "loss": "one-step + pushforward-stability SmoothL1(beta=0.1)",
        "boundary_mask_applied": True,
        "max_production_parity_delta": max_parity_delta,
        "production_parity_gate": "<=1e-7",
        "max_gt_residual_abs": gt_residual_max,
        "ground_truth_residual_gate": "<=1e-4",
        "gradient_definition": (
            "L2 norm with respect to detached node/edge decoder outputs; each "
            "group term retains its actual 1/N_kept weight in total LMC."
        ),
        "counterfactual_definition": (
            "Same residual and model outputs, but model-predicted connection "
            "delta-Q is retained instead of substituting recorded delta-Q."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {output_dir / 'teacher_forced_raw.csv'}")
    print(f"Wrote {output_dir / 'teacher_forced_summary.csv'}")


if __name__ == "__main__":
    main()
