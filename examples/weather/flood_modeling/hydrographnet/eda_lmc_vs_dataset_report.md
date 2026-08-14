# Why does Local Mass Conservation (LMC) help on HydroGraphNet but hurt on UrbanFlood?

**Author:** Muhammad Hashmi · **Date:** 2026-05-25 (updated 2026-05-29) · **Companion script:** [eda_lmc_vs_dataset.py](eda_lmc_vs_dataset.py) · **Raw outputs:** [eda_results/](eda_results/)

> ## ⚠️ Update 2026-05-29 — the headline question has a sharper answer
>
> A City-2 (Model_2) experiment run after this report **flipped the result**:
> on Model_2's richly-coupled graph (197 connections vs Model_1's 17), **LMC
> *helps* by −6.2%** (2D 0.0140 ft vs no-LMC 0.0149 ft) — a near-mirror-image
> of Model_1's +6.2% penalty. Crucially, Model_2's loss-direction correlation
> is **r = +0.989** — i.e. the diagnostic in this report's headline finding
> *correctly predicts* the flip.
>
> So the corrected takeaway is **not** "LMC fails on UrbanFlood." It is:
> **LMC's sign is governed by 1D-2D coupling *density*, and the loss-direction
> correlation r is a measurable predictor of it** (r > 0.98 ⇒ helps, r < 0 ⇒
> hurts). The structural mechanism below (the §"What in the data causes the
> conflict?" section) is right about *why Model_1 fails* but its directional
> prediction ("more coupling → worse") is **wrong** — dense coupling makes the
> LMC residual *better*-conditioned, not worse. See §"What this implies" for
> the corrected synthesis.

## The puzzle

The same `compute_local_conservation_loss` function ([utils.py:156-314](utils.py)) is used on two flood-modelling datasets:

| Dataset | Best LMC vs no-LMC | Verdict |
|---|---|---|
| HydroGraphNet (White River, KNN k=4, pure 2D) | **−15% to −22% RMSE** | LMC clearly helps |
| UrbanFlood Model_1 (physical mesh + 1D drainage + 2D-1D coupling) | **+6% RMSE** (best LMC 0.0207 ft vs no-LMC 0.0195 ft) | LMC actively hurts |

Same formula. Same hyper-parameter family (smooth-L1 β=0.1, warmup 5 ep, λ ≈ 0.03–0.05). After exhausting every loss-side knob and training-regime variation (v3-v7), the +0.0012 ft gap on UrbanFlood persisted. This EDA isolates the structural cause.

## Headline finding — the LMC and prediction-MSE losses are pulling in opposite directions on UrbanFlood

We mined the per-epoch loss components from both runs' `train.log` files and computed the Pearson correlation between **prediction MSE** (`loss_one + loss_stability`) and **LMC loss** (`local_physics_loss`), over epochs ≥ 5 (post-warmup):

| Dataset | Run | Epochs (post-warmup) | Pearson r(MSE, LMC) | Interpretation |
|---|---|---|---|---|
| HydroGraphNet | v7 R1 rep 1 (`outputs_lc_v7_run1_rep1`) | 15 | **+0.999** | Perfectly co-decreasing — LMC reinforces the prediction objective |
| UrbanFlood Model_1 | LC bundle v3 Exp 1 (`outputs_lc_gt_conn_1d`) | 45 | **−0.199** | Mildly anti-correlated — LMC *fights* the prediction objective |

![Loss trajectory — HydroGraphNet](eda_results/loss_trajectory_hydrographnet.png)

![Loss trajectory — UrbanFlood Model_1, v3 Exp 1](eda_results/loss_trajectory_urbanflood.png)

On HydroGraphNet, the two curves drop together throughout training. On UrbanFlood, the LMC loss bottoms out around epoch 9 and then **starts creeping back up while the prediction MSE keeps falling** — concrete evidence that improving the prediction is *increasing* the local-conservation residual. The model can only optimise one or the other; under the v3 Exp 1 loss weight (λ=0.03) it correctly prioritises prediction, but the LMC term still bleeds gradient that destabilises 2D rollouts.

This is the structural conflict the loss-side knobs (per-type weighting, restrict_to_2d, smooth-L1 β) could not fix — they reshape the *magnitude* of the LMC residual, not its *direction* relative to the prediction gradient.

## What in the data causes the conflict?

Side-by-side structural metrics (from [eda_results/structural_summary.csv](eda_results/structural_summary.csv)):

| Metric | HydroGraphNet (KNN k=4) | UrbanFlood Model_1 | UrbanFlood Model_2 |
|---|---|---|---|
| 2D nodes | 4,787 | 3,716 | 4,299 |
| 1D nodes | 0 | 17 | **198** |
| 2D-1D connections | 0 | 16 | **197** |
| Undirected 2D edges | 14,665 | 7,935 | 9,876 |
| Mean / max degree | 6.1 / **58** | 4.3 / 8 | 4.6 / 8 |
| Edge-length CoV | **11.86** (extreme outliers) | 0.16 | 0.37 |
| Convex-hull boundary fraction | 0.23% | 0.54% | 0.40% |
| BC-union fraction (boundary + 1D inlets) | 0.23% | 0.97% | **4.42%** |

![Node degree distribution](eda_results/degree_distribution.png)

![Edge length distribution](eda_results/edge_length_distribution.png)

### The structural differences that matter

1. **1D-2D coupling exists only on UrbanFlood.** HydroGraphNet is a pure-2D KNN graph; UrbanFlood embeds a 1D drainage network with **inlet nodes** where the 2D mass-balance equation borrows GT connection flow (the v3 Exp 1 fix). On Model_1 there are 16 such inlets; on Model_2 there are **197**. These inlets are *boundary conditions inside the domain*, not on its perimeter. Every inlet seeds a region where the model's LMC residual depends on a GT-substituted flow it cannot reach by gradient descent.

2. **The BC-union fraction is 4× larger on Model_1 and 19× larger on Model_2** than HydroGraphNet. The denser the coupling, the larger the fraction of nodes where the LMC gradient is "trying to drive an unreachable residual to zero." This is the per-node analogue of the loss-direction conflict above.

3. **Degree heterogeneity is the opposite of what one might guess.** HydroGraphNet's KNN graph is *more* irregular (mean degree 6.1, max 58 — KNN-from-A asymmetry makes some nodes very popular targets) than UrbanFlood's physical mesh (max 8, mean ≈ 4.5). So regularity *isn't* the protective property. Pure-2D-ness is.

4. **Edge-length CoV is 30-70× higher on HydroGraphNet** (CoV 11.9 vs 0.16-0.37 on UF). HydroGraphNet survives this because the LMC residual is denormalised by per-edge flow std, which absorbs the geometric variation. UrbanFlood doesn't benefit from this because its edges are uniform anyway — geometry is not what's preventing LMC from working there.

### Synthesis (corrected 2026-05-29)

The headline correlation result remains the right diagnostic: **when LMC and prediction MSE point the same way (r → +1), LMC helps; when they conflict (r < 0), LMC hurts.** What the original draft got wrong was the *direction* in which 1D-2D coupling moves that correlation. The Model_2 experiment shows:

| Dataset | 2D-1D connections (% of nodes) | r(MSE, LMC) | LMC effect on 2D |
|---|---|---|---|
| HydroGraphNet | 0 (pure 2D) | +0.999 | helps −15% to −22% |
| **UrbanFlood Model_2** | **197 (4.6%)** | **+0.989** | **helps −6.2%** |
| UrbanFlood Model_1 | 16-17 (0.4%) | −0.199 | hurts +6.2% |

So **dense coupling makes the LMC residual *better*-conditioned, not worse.** The "internal BC drives an unreachable residual" mechanism (§above) is real, but it dominates only in the *sparse*-coupling regime (Model_1): with just 17 inlets, the GT-substituted connection flow is a sparse, high-leverage, noisy constraint that conflicts with prediction. With 197 inlets (Model_2) the same constraint is dense and well-averaged, so it behaves like an honest conservation law — exactly as in the pure-2D HydroGraphNet case. The earlier monotone reading of the "BC-union fraction" row (point 2 above) had the sign backwards.

## What this implies for next steps

The v7 "no-architecture-change ceiling" conclusion was drawn entirely on **Model_1, the worst-case sparse-coupling regime.** It does not generalise: LMC is already a net win on Model_2 with no architectural change at all.

Paths forward, in corrected order of evidence strength:

1. **✅ Done — Model_2 confirms LMC helps (−6.2%).** SLURM 1645863 (LMC) + 1645864 (nolc); infer 1664051/1664052. This re-opens LMC as a beneficial technique for richly-coupled urban catchments and supplies the predictive r-metric. The original prediction here ("penalty should be larger on Model_2") was **falsified** — the opposite occurred.

2. **Use r(MSE, LMC) as a cheap go/no-go gate.** Before committing to a long LMC run on a new catchment, train a short pilot and measure the post-warmup loss correlation. r > ~0.9 predicts LMC will help; r < 0 predicts it will hurt. This is far cheaper than a full train+infer LMC-vs-nolc bracket per city.

3. **Architectural change at the connection edges (only needed for sparse-coupling cities like Model_1).** The decoupled edge head means the model can't learn to predict consistent inlet flows; the LMC then has nothing to align against. Candidates from the v7 results "future work" list:
   - **FiLM-conditioned edge encoder** (couples edge predictions to node states at the inlet),
   - **Concat-trick edge decoder** (DUALFloodGNN-style endpoint concat — tested with antisym in v4 Exp 4 and regressed, but not in isolation),
   - **Antisym alone** (LMC residual built from forward edges only — never tested without concat).

3. **Drop the 1D drainage from the LMC residual at training time but keep it as input features.** This was Exp 8 (restrict_to_2d) and regressed by +0.0018 ft — but the regression was attributed to "losing an implicit 1D-side stabiliser." Re-reading that result in light of this EDA: the stabiliser may have been a side-effect of the loss-mean dilution, not a feature. Combined with FiLM at the connection edges, this becomes worth a second pass.

## How to reproduce

```
cd /home/hpc/iwi5/iwi5416h/hydrographnet/physicsnemo/examples/weather/flood_modeling/hydrographnet
conda activate hydrographnet
python eda_lmc_vs_dataset.py
```

Results land in `eda_results/`. The script reads raw mesh files (no GPU needed) and the existing training logs at:

* HydroGraphNet: `/home/woody/iwi5/iwi5416h/hydrographnet/outputs_lc_v7_run1_rep1/train.log`
* UrbanFlood: `/home/woody/iwi5/iwi5416h/urbanflood/outputs_lc_gt_conn_1d/train.log`
