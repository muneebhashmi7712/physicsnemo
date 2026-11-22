#!/usr/bin/env python3
"""Measure LMC group shares on autoregressive train-window rollouts.

This is the recursive companion to ``analyze_lmc_node_groups.py``.  Each
sample starts from the same ground-truth two-step window used by training, then
unrolls the model for a fixed eight-step horizon while updating water depth,
volume, forcing, and predicted edge-flow state.  At each exposed rollout state
it measures the production LMC residual (recorded connection-flow substitution)
and a paired all-model-flow counterfactual.

The output gradients are local gradients with respect to the current detached
decoder outputs, conditional on the recursively generated state.  They do not
backpropagate through earlier rollout states; that matches the teacher-forced
diagnostic's output-gradient interpretation and isolates where the loss applies
at each state.
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

from physicsnemo.datapipes.gnn.hydrographnet_dataset import UrbanFloodDataset
from physicsnemo.utils import load_checkpoint
from utils import compute_local_conservation_loss

from analyze_lmc_node_groups import (
    GROUPS,
    build_model,
    group_masks,
    residual_terms,
    select_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-name", choices=("Model_1", "Model_2"), required=True)
    parser.add_argument("--num-events", type=int, required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", default="0,5,10,20,30,40,49")
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=8)
    parser.add_argument(
        "--edge-q-prev-as-input",
        action="store_true",
        help="Mirror an 11th autoregressive Q-prev edge feature when present.",
    )
    return parser.parse_args()


def rollout_grad_norms(
    group_loss: torch.Tensor,
    node_output: torch.Tensor,
    edge_output: torch.Tensor,
    graph,
) -> dict[str, float]:
    """L2 norms of one rollout state's group loss with respect to outputs."""
    node_grad, edge_grad = torch.autograd.grad(
        group_loss,
        (node_output, edge_output),
        retain_graph=True,
        allow_unused=True,
    )

    def l2(parts: list[torch.Tensor | None]) -> float:
        total = torch.zeros((), device=group_loss.device, dtype=torch.float64)
        for part in parts:
            if part is not None:
                total += part.to(torch.float64).square().sum()
        return float(total.sqrt().item())

    result = {
        "node_output_grad_l2": l2([node_grad]),
        "edge_output_grad_l2": l2([edge_grad]),
    }
    for edge_type, column in (("2d", 7), ("1d", 8), ("conn", 9)):
        mask = graph.edge_attr[:, column].to(group_loss.device).bool()
        result[f"edge_{edge_type}_grad_l2"] = l2(
            [None if edge_grad is None else edge_grad[mask]]
        )
    return result


def summarize(raw_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[int, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        buckets[
            (int(row["epoch"]), int(row["rollout_step"]), str(row["group"]))
        ].append(row)

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
    summary_rows: list[dict[str, object]] = []
    for (epoch, rollout_step, group), rows in sorted(buckets.items()):
        item: dict[str, object] = {
            "epoch": epoch,
            "rollout_step": rollout_step,
            "group": group,
            "samples": len(rows),
        }
        for key in numeric:
            item[f"mean_{key}"] = float(
                np.mean([float(row[key]) for row in rows])
            )
        summary_rows.append(item)

    by_state: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        by_state[(int(row["epoch"]), int(row["rollout_step"]))].append(row)
    for rows in by_state.values():
        actual_total = sum(float(row["mean_actual_contribution"]) for row in rows)
        counterfactual_total = sum(
            float(row["mean_counterfactual_contribution"]) for row in rows
        )
        actual_grad_total = sum(
            float(row["mean_actual_node_output_grad_l2"])
            + float(row["mean_actual_edge_output_grad_l2"])
            for row in rows
        )
        counterfactual_grad_total = sum(
            float(row["mean_counterfactual_node_output_grad_l2"])
            + float(row["mean_counterfactual_edge_output_grad_l2"])
            for row in rows
        )
        for row in rows:
            actual = float(row["mean_actual_contribution"])
            counterfactual = float(row["mean_counterfactual_contribution"])
            row["actual_loss_fraction_pct"] = 100.0 * actual / actual_total
            row["counterfactual_loss_fraction_pct"] = (
                100.0 * counterfactual / counterfactual_total
            )
            row["gt_connection_substitution_effect_pct"] = (
                (actual / counterfactual - 1.0) * 100.0
                if counterfactual != 0.0
                else float("nan")
            )
            row["actual_output_grad_norm_fraction_pct"] = 100.0 * (
                float(row["mean_actual_node_output_grad_l2"])
                + float(row["mean_actual_edge_output_grad_l2"])
            ) / actual_grad_total
            row["counterfactual_output_grad_norm_fraction_pct"] = 100.0 * (
                float(row["mean_counterfactual_node_output_grad_l2"])
                + float(row["mean_counterfactual_edge_output_grad_l2"])
            ) / counterfactual_grad_total
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise AssertionError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_finite(
    tensors: dict[str, torch.Tensor],
    *,
    epoch: int,
    sample_idx: int,
    event_id: str,
    rollout_step: int,
) -> None:
    nonfinite = {
        name: int((~torch.isfinite(value)).sum().item())
        for name, value in tensors.items()
        if not torch.isfinite(value).all()
    }
    if nonfinite:
        raise FloatingPointError(
            "non-finite recursive diagnostic output: "
            f"epoch={epoch}, sample={sample_idx}, event={event_id}, "
            f"rollout_step={rollout_step}, fields={nonfinite}"
        )


def main() -> None:
    args = parse_args()
    if args.rollout_length < 1:
        raise ValueError("--rollout-length must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = time.perf_counter()
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
        edge_q_prev_as_input=args.edge_q_prev_as_input,
        lmc_antisymmetric=False,
        train_rollout_length=args.rollout_length,
    )
    print(
        f"Dataset initialized in {time.perf_counter() - started_at:.1f}s; "
        f"selecting {args.sample_count} windows × {args.rollout_length} steps.",
        flush=True,
    )
    selected = select_indices(dataset, args.sample_count)
    cached = []
    event_coverage = set()
    for sample_idx in selected:
        graph, _ = dataset[sample_idx]
        hydro_idx, time_idx = dataset.sample_index[sample_idx]
        dynamic = dataset.dynamic_data[hydro_idx]
        target_time = time_idx + dataset.n_time_steps
        event_id = str(dynamic["event_id"])
        event_coverage.add(event_id)
        flow = np.asarray(
            dynamic["inflow_hydrograph"][
                target_time : target_time + args.rollout_length
            ],
            dtype=np.float32,
        )
        precip = np.asarray(
            dynamic["precipitation"][
                target_time : target_time + args.rollout_length
            ],
            dtype=np.float32,
        )
        if len(flow) != args.rollout_length or len(precip) != args.rollout_length:
            raise AssertionError(
                f"sample {sample_idx} does not contain the requested rollout horizon"
            )
        if not hasattr(graph, "edge_y_rollout") or not hasattr(
            graph, "source_term_rollout"
        ):
            raise AssertionError("rollout physics fields are missing")
        cached.append((sample_idx, event_id, int(time_idx), graph, flow, precip))

    print(
        f"Cached {len(cached)} windows in {time.perf_counter() - started_at:.1f}s; "
        f"evaluating epochs {epochs}.",
        flush=True,
    )

    model = build_model(device)
    raw_rows: list[dict[str, object]] = []
    parity_deltas: list[float] = []
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

        for sample_idx, event_id, time_idx, graph_cpu, flow, precip in cached:
            # ``Data.to`` can mutate a data object; clone to preserve the exact
            # GT initial state for the next epoch and for the other checkpoints.
            graph = graph_cpu.clone().to(device)
            masks, keep = group_masks(graph, device)
            n_kept = int(keep.sum().item())
            conn_mask = graph.edge_attr[:, 9].to(device).bool()

            x = graph.x
            n_time = 3  # n_time_steps=2 plus one pushforward column
            n_static = x.shape[1] - 2 * n_time
            static = x[:, :n_static].clone()
            wd_full = x[:, n_static : n_static + n_time]
            vol_full = x[:, n_static + n_time : n_static + 2 * n_time]
            wd_window = wd_full[:, 1:].clone()
            vol_window = vol_full[:, 1:].clone()
            edge_features = graph.edge_attr.clone()

            # Three physical-flow state variants are required.  The model sees
            # only its predicted state (when Q_prev is an input); actual LMC
            # treats connection increments as recorded, counterfactual LMC
            # treats every increment as model-predicted, and the GT stream is
            # used only for the independent mass-balance consistency gate.
            q_predicted = graph.edge_q_prev.clone()
            q_actual = graph.edge_q_prev.clone()
            q_ground_truth = graph.edge_q_prev.clone()

            for rollout_step in range(args.rollout_length):
                graph.source_term = graph.source_term_rollout[:, rollout_step]
                graph.edge_y = graph.edge_y_rollout[:, rollout_step, :]

                graph.edge_q_prev = q_predicted
                x_input = torch.cat([static, wd_window, vol_window], dim=1)
                with torch.no_grad():
                    model_out = model(x_input, edge_features, graph)
                if not isinstance(model_out, tuple) or len(model_out) < 2:
                    raise AssertionError("LMC diagnostic requires node and edge outputs")
                node_model, edge_model = model_out[0], model_out[1]
                assert_finite(
                    {"node_output": node_model, "edge_output": edge_model},
                    epoch=epoch,
                    sample_idx=sample_idx,
                    event_id=event_id,
                    rollout_step=rollout_step,
                )
                node_out = node_model.detach().requires_grad_(True)
                edge_out = edge_model.detach().requires_grad_(True)

                graph.edge_q_prev = q_actual
                _, actual_per_node, _ = residual_terms(
                    node_out, edge_out, graph, use_gt_connection_flow=True
                )
                graph.edge_q_prev = q_predicted
                _, counterfactual_per_node, _ = residual_terms(
                    node_out, edge_out, graph, use_gt_connection_flow=False
                )

                graph.edge_q_prev = q_actual
                production = compute_local_conservation_loss(
                    node_out,
                    edge_out,
                    graph,
                    None,
                    smooth_l1_beta=0.1,
                    apply_boundary_mask=True,
                    restrict_to_2d=False,
                    node_type_weighting="none",
                )
                reproduced = actual_per_node[keep].sum() / n_kept
                parity_deltas.append(float((production - reproduced).abs().item()))

                # The ground-truth path must use the true Q state, rather than
                # either of the recursively generated predicted states.
                graph.edge_q_prev = q_ground_truth
                gt_residual, _, _ = residual_terms(
                    graph.y_rollout[:, rollout_step, :],
                    graph.edge_y_rollout[:, rollout_step, :],
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
                    actual_sum = actual_per_node[mask].sum()
                    counterfactual_sum = counterfactual_per_node[mask].sum()
                    actual_group_loss = actual_sum / n_kept
                    counterfactual_group_loss = counterfactual_sum / n_kept
                    row: dict[str, object] = {
                        "epoch": epoch,
                        "rollout_step": rollout_step,
                        "sample_index": sample_idx,
                        "event_id": event_id,
                        "time_index": time_idx,
                        "group": group,
                        "nodes": nodes,
                        "kept_nodes": n_kept,
                        "actual_group_mean": float((actual_sum / nodes).item()),
                        "actual_contribution": float(actual_group_loss.item()),
                        "counterfactual_group_mean": float(
                            (counterfactual_sum / nodes).item()
                        ),
                        "counterfactual_contribution": float(
                            counterfactual_group_loss.item()
                        ),
                    }
                    row.update(
                        {
                            f"actual_{key}": value
                            for key, value in rollout_grad_norms(
                                actual_group_loss, node_out, edge_out, graph
                            ).items()
                        }
                    )
                    row.update(
                        {
                            f"counterfactual_{key}": value
                            for key, value in rollout_grad_norms(
                                counterfactual_group_loss, node_out, edge_out, graph
                            ).items()
                        }
                    )
                    raw_rows.append(row)

                # Advance the exposed model state exactly as inference does:
                # autoregressive depth/volume, forcing from the just-predicted
                # target time, and model-predicted Q for model input.  The two
                # LMC residual streams retain their stated flow semantics.
                new_wd = wd_window[:, -1:] + node_model[:, 0:1]
                new_vol = vol_window[:, -1:] + node_model[:, 1:2]
                wd_window = torch.cat([wd_window[:, 1:], new_wd], dim=1)
                vol_window = torch.cat([vol_window[:, 1:], new_vol], dim=1)
                static = static.clone()
                static[:, 10] = torch.as_tensor(
                    flow[rollout_step], device=device, dtype=static.dtype
                )
                static[:, 11] = torch.as_tensor(
                    precip[rollout_step], device=device, dtype=static.dtype
                )

                edge_delta = edge_model.detach().squeeze(-1)
                q_actual_previous = q_actual
                q_predicted = q_predicted + edge_delta
                q_actual = q_actual + edge_delta
                q_actual = torch.where(
                    conn_mask,
                    q_actual_previous + graph.edge_y.squeeze(-1),
                    q_actual,
                )
                q_ground_truth = q_ground_truth + graph.edge_y.squeeze(-1)
                if args.edge_q_prev_as_input:
                    edge_features = edge_features.clone()
                    edge_features[:, -1] = edge_features[:, -1] + edge_delta

        print(
            f"Epoch {epoch} recursive diagnostic complete in "
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
    write_csv(output_dir / "recursive_rollout_raw.csv", raw_rows)
    write_csv(output_dir / "recursive_rollout_summary.csv", summary_rows)
    metadata = {
        "label": args.label,
        "model_name": args.model_name,
        "checkpoint_dir": args.checkpoint_dir,
        "epochs": epochs,
        "sample_count": len(selected),
        "rollout_length": args.rollout_length,
        "states_per_epoch": len(selected) * args.rollout_length,
        "total_training_samples": len(dataset),
        "events_covered": len(event_coverage),
        "total_events": len(dataset.event_ids),
        "groups": list(GROUPS),
        "mode": "recursive autoregressive train-window rollout",
        "boundary_mask_applied": True,
        "max_production_parity_delta": max_parity_delta,
        "production_parity_gate": "<=1e-7",
        "max_gt_residual_abs": gt_residual_max,
        "ground_truth_residual_gate": "<=1e-4",
        "gradient_definition": (
            "L2 norm with respect to current detached node/edge decoder outputs, "
            "conditional on the recursively generated rollout state; each group "
            "term retains its actual 1/N_kept weight in total LMC."
        ),
        "model_state_definition": (
            "water depth, volume, static forcing, and (if present) model-input "
            "Q-prev are updated autoregressively from model predictions."
        ),
        "production_q_state_definition": (
            "non-connection Q increments use model predictions; connection Q "
            "increments use recorded targets, consistently with production LMC."
        ),
        "counterfactual_definition": (
            "Same recursive state and residual, but model-predicted connection "
            "delta-Q and all-model Q state are retained."
        ),
    }
    (output_dir / "recursive_rollout_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {output_dir / 'recursive_rollout_raw.csv'}")
    print(f"Wrote {output_dir / 'recursive_rollout_summary.csv'}")


if __name__ == "__main__":
    main()
