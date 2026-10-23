# Metric definitions

Each fitted repeat predicts the same151 archived events once out of fold. Random allocation has10 policy seeds per repeat. Metrics are computed per repeat and policy seed, then averaged; predictions are not averaged before computing nonlinear metrics. MAE uncertainty uses paired per-event absolute errors averaged over repeats and policy seeds, resampling52 animal groups10000 times with fitted models held fixed. Folds/repeats/frames are not independent animals.

Rounded predictions use NumPy nearest-even rounding and clipping to observed labels2-6. Tolerances0.5 and1.0 apply to continuous score differences; they are descriptive, not validated management thresholds. Severe error is absolute continuous error>=2. Class MAE gives equal weight to5 represented classes; animal MAE gives equal weight to52 inherited identifier groups. R2 may be negative.

Depth cost counts actual encoded frames. Probe models pay one frame for every probed event, including RGB fallback. RGB always costs1245 frames. Total reduction uses2490 full RGBD frame calls as denominator. Coverage means outputs/151 retained archived events, not all on-farm acquisition opportunities. Fixed-level baselines are flagged infeasible when they exceed the declared cap.

All nominal confidence intervals are exploratory; repeated reuse and method search are disclosed. No post-hoc noninferiority margin is used.
