# Reflection (≤1 page)

**Which fault types were hardest to catch, and why?**

The hardest faults to catch were the **subtle-tier data quality faults** in the `checks` pillar — particularly subtle `volume_spike`, `distribution_shift`, `null_spike`, and `freshness_lag` instances. These sit close to normal variance (within the baseline ±3σ range), so static threshold checks alone miss them. Even adaptive z-score detection based on running statistics struggled because the anomalous values were only slightly outside the observed distribution rather than dramatic outliers.

Within the `lineage` pillar, **`missing_upstream`** was initially challenging because the fault isn't a threshold violation — it's a structural change where one of normally-present upstream edges simply disappears. This required building a per-job "known-good" upstream set from clean events and detecting when the actual set was a strict subset.

For `ai_infra`, the **`feature_skew` subtle-tier** faults were actually easier than expected: even "subtle" shift values (2.3–2.9σ) were well above the baseline max (0.41σ), making them trivially detectable. The private phase likely pushed these much closer to the baseline.

**What would you change about your cost/coverage tradeoff, if you had another pass?**

1. **Lower the adaptive z-score thresholds for the checks pillar.** I used z > 3.5 to avoid false positives on practice, but this is too conservative for subtle faults in private. A z > 2.5–3.0 with an absolute minimum check (value > 80% of baseline) would catch more subtle faults while maintaining low FPR.

2. **Per-feature-view tracking for feature_skew.** Rather than pooling all feature shifts into one distribution, tracking per-`feature_view` baselines would improve the signal-to-noise ratio for subtle skew detection.

3. **Smarter budget allocation.** Instead of calling every tool on every event uniformly, I'd prioritize expensive calls (feature_drift, embedding_drift at 2.0 cost each) on events whose payload metadata suggests higher risk, and skip tool calls on events that look statistically typical based on payload-level heuristics alone.

4. **Exponential moving average** instead of full-history statistics to adapt to non-stationary distributions and give more weight to recent clean observations.
