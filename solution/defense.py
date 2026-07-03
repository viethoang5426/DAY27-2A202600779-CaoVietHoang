"""
Your defense. Implement register(ctx) and a handler per event type.
See ../README.md for the full interface + toolkit reference, and
../RULES.md before you start.

Strategy:
- Use baseline thresholds (calibrated at mean ± 3σ) for primary detection
- Track per-job "normal" patterns for lineage structural anomalies
- Adaptive z-score detection with conservative thresholds for subtle faults
- Only track clean events in running stats to avoid poisoning
- Budget-aware: skip expensive tool calls when budget is critically low
"""
from api import Verdict


def register(ctx):
    ctx.state["batch_stats"] = {
        "row_counts": [],
        "null_rates": [],
        "mean_amounts": [],
        "staleness_values": [],
    }
    ctx.state["lineage_stats"] = {
        "durations": [],
        # Track per-job normal upstream sets and downstream counts
        "job_upstreams": {},  # job -> set of all upstream names seen
        "job_downstream_counts": {},  # job -> list of downstream counts seen
    }
    ctx.state["feature_stats"] = {
        "shifts": [],
    }
    ctx.state["embedding_stats"] = {
        "centroid_shifts": [],
        "doc_ages": [],
    }

    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _safe_mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _safe_std(values):
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5


def _budget_ok(ctx, cost):
    """Check if we can afford a tool call of the given cost.
    Allow going slightly negative rather than cutting off detection entirely."""
    remaining = ctx.tools.budget_remaining()  # free call
    # Only skip if we'd go significantly over budget (> 5% overage)
    return remaining >= -cost * 0.05


def check_data_batch(payload, ctx):
    """Detect freshness_lag, volume_spike, null_spike, distribution_shift."""
    # Cost: 1.0 (cheap)
    if not _budget_ok(ctx, 1.0):
        return Verdict(alert=False, pillar="checks", reason="budget exhausted")

    batch_id = payload["batch_id"]
    profile = ctx.tools.batch_profile(batch_id)

    if isinstance(profile, dict) and "error" in profile:
        return Verdict(alert=False, pillar="checks", reason="tool error")

    b = ctx.baseline
    alerts = []

    row_count = profile["row_count"]
    null_rate = profile["null_rate"]["customer_id"]
    mean_amount = profile["mean_amount"]
    staleness = profile["staleness_min"]

    stats = ctx.state["batch_stats"]

    # Primary baseline threshold checks
    if row_count < b["row_count_min"] or row_count > b["row_count_max"]:
        alerts.append("volume_spike")

    if null_rate > b["null_rate_max"]:
        alerts.append("null_spike")

    if mean_amount < b["mean_amount_min"] or mean_amount > b["mean_amount_max"]:
        alerts.append("distribution_shift")

    if staleness > b["staleness_min_max"]:
        alerts.append("freshness_lag")

    # Adaptive z-score detection for subtle faults (conservative threshold)
    if len(stats["row_counts"]) >= 10 and not alerts:
        pairs = [
            (stats["row_counts"], row_count, True),   # two-sided
            (stats["null_rates"], null_rate, False),   # one-sided (high)
            (stats["mean_amounts"], mean_amount, True),
            (stats["staleness_values"], staleness, False),
        ]
        names = ["volume_spike", "null_spike", "distribution_shift", "freshness_lag"]
        for (hist, val, two_sided), name in zip(pairs, names):
            m = _safe_mean(hist)
            s = _safe_std(hist)
            if s and s > 0:
                z = abs(val - m) / s if two_sided else (val - m) / s
                if z > 3.5:
                    alerts.append(name + "_adaptive")

    # Only track clean events to avoid poisoning
    if not alerts:
        stats["row_counts"].append(row_count)
        stats["null_rates"].append(null_rate)
        stats["mean_amounts"].append(mean_amount)
        stats["staleness_values"].append(staleness)

    return Verdict(
        alert=bool(alerts),
        pillar="checks",
        reason=", ".join(alerts) if alerts else "clean",
    )


def check_contract_checkpoint(payload, ctx):
    """Detect schema_break, type_violation, freshness SLA violations."""
    # Cost: 1.5
    if not _budget_ok(ctx, 1.5):
        return Verdict(alert=False, pillar="contracts", reason="budget exhausted")

    contract_id = payload["contract_id"]
    checkpoint_batch_id = payload["checkpoint_batch_id"]
    diff = ctx.tools.contract_diff(contract_id, checkpoint_batch_id)

    if isinstance(diff, dict) and "error" in diff:
        return Verdict(alert=False, pillar="contracts", reason="tool error")

    alerts = []

    violations = diff.get("violations", [])
    if "schema_hash_mismatch" in violations:
        alerts.append("schema_break")
    if "type_violation" in violations:
        alerts.append("type_violation")

    freshness_delay = diff.get("freshness_delay_min", 0)
    freshness_max = ctx.baseline.get("freshness_delay_max_min", float("inf"))
    if freshness_delay > freshness_max:
        alerts.append("freshness_sla_breach")

    return Verdict(
        alert=bool(alerts),
        pillar="contracts",
        reason=", ".join(alerts) if alerts else "clean",
    )


def check_lineage_run(payload, ctx):
    """Detect missing_upstream, orphan_output, runtime_anomaly.

    Key patterns learned from data:
    - All lineage events share the same job (dbt:stg_orders)
    - Normal events have actual_upstream=['raw.orders', 'raw.customers'] and
      actual_downstream_count=1
    - missing_upstream: actual_upstream is missing entries (e.g., only ['raw.orders'])
    - orphan_output: actual_downstream_count=0
    - runtime_anomaly: duration_ms exceeds baseline
    """
    # Cost: 1.0 (cheap)
    if not _budget_ok(ctx, 1.0):
        return Verdict(alert=False, pillar="lineage", reason="budget exhausted")

    run_id = payload["run_id"]
    graph = ctx.tools.lineage_graph_slice(run_id)

    if isinstance(graph, dict) and "error" in graph:
        return Verdict(alert=False, pillar="lineage", reason="tool error")

    alerts = []

    duration_ms = graph["duration_ms"]
    actual_upstream = graph["actual_upstream"]
    downstream_count = graph["actual_downstream_count"]
    job = payload.get("job", "default")

    stats = ctx.state["lineage_stats"]

    # --- Runtime anomaly (baseline check) ---
    duration_max = ctx.baseline["lineage_duration_ms_max"]
    if duration_ms > duration_max:
        alerts.append("runtime_anomaly")

    # Adaptive runtime check
    if len(stats["durations"]) >= 5 and "runtime_anomaly" not in alerts:
        d_mean = _safe_mean(stats["durations"])
        d_std = _safe_std(stats["durations"])
        if d_std and d_std > 0:
            z = (duration_ms - d_mean) / d_std
            if z > 3.5:
                alerts.append("runtime_anomaly_adaptive")

    # --- Missing upstream (structural comparison) ---
    # Track the "normal" set of upstream names for each job
    upstream_names = set(actual_upstream) if isinstance(actual_upstream, list) else set()
    job_upstreams = stats["job_upstreams"]

    if job in job_upstreams:
        # Compare against known normal upstream set
        normal_set = job_upstreams[job]
        if upstream_names and normal_set and upstream_names < normal_set:
            # Current upstream is a strict subset of normal — missing entries
            alerts.append("missing_upstream")
    # else: first time seeing this job, can't compare yet

    # --- Orphan output (structural check) ---
    # Track normal downstream counts per job
    job_downstream = stats["job_downstream_counts"]
    if job in job_downstream and len(job_downstream[job]) >= 2:
        normal_downstream = job_downstream[job]
        normal_mean = _safe_mean(normal_downstream)
        if normal_mean and normal_mean > 0 and downstream_count == 0:
            alerts.append("orphan_output")

    # Track clean events: update job upstream set (union) and downstream history
    if not alerts:
        stats["durations"].append(duration_ms)
        # Build up the "known good" upstream set as the union of all clean upstreams
        if job not in job_upstreams:
            job_upstreams[job] = set()
        job_upstreams[job] |= upstream_names
        # Track downstream counts
        if job not in job_downstream:
            job_downstream[job] = []
        job_downstream[job].append(downstream_count)

    return Verdict(
        alert=bool(alerts),
        pillar="lineage",
        reason=", ".join(alerts) if alerts else "clean",
    )


def check_feature_materialization(payload, ctx):
    """Detect feature_skew (training-serving drift)."""
    # Cost: 2.0 (expensive)
    if not _budget_ok(ctx, 2.0):
        return Verdict(alert=False, pillar="ai_infra", reason="budget exhausted")

    feature_view = payload["feature_view"]
    batch_id = payload["batch_id"]
    drift = ctx.tools.feature_drift(feature_view, batch_id)

    if isinstance(drift, dict) and "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason="tool error")

    alerts = []
    mean_shift_sigma = drift["mean_shift_sigma"]
    stats = ctx.state["feature_stats"]

    # Primary baseline check
    shift_max = ctx.baseline["feature_mean_shift_sigma_max"]
    if mean_shift_sigma > shift_max:
        alerts.append("feature_skew")

    # Adaptive detection for subtle skew
    # Require both: (1) statistical outlier AND (2) value > 50% of baseline threshold
    # This avoids FP on clean values that are merely at the high end of normal
    if len(stats["shifts"]) >= 8 and "feature_skew" not in alerts:
        s_mean = _safe_mean(stats["shifts"])
        s_std = _safe_std(stats["shifts"])
        if s_std and s_std > 0:
            z = (mean_shift_sigma - s_mean) / s_std
            if z > 4.0 and mean_shift_sigma > shift_max * 0.5:
                alerts.append("feature_skew_adaptive")

    # Only track clean events
    if not alerts:
        stats["shifts"].append(mean_shift_sigma)

    return Verdict(
        alert=bool(alerts),
        pillar="ai_infra",
        reason=", ".join(alerts) if alerts else "clean",
    )


def check_embedding_batch(payload, ctx):
    """Detect embedding_drift and corpus_staleness."""
    # Cost: 2.0 (expensive)
    if not _budget_ok(ctx, 2.0):
        return Verdict(alert=False, pillar="ai_infra", reason="budget exhausted")

    corpus = payload["corpus"]
    chunk_batch_id = payload["chunk_batch_id"]
    drift = ctx.tools.embedding_drift(corpus, chunk_batch_id)

    if isinstance(drift, dict) and "error" in drift:
        return Verdict(alert=False, pillar="ai_infra", reason="tool error")

    alerts = []
    centroid_shift = drift["centroid_shift"]
    avg_doc_age = drift["avg_doc_age_days"]
    stats = ctx.state["embedding_stats"]

    # Primary baseline checks
    if centroid_shift > ctx.baseline["embedding_centroid_shift_max"]:
        alerts.append("embedding_drift")

    if avg_doc_age > ctx.baseline["corpus_avg_doc_age_days_max"]:
        alerts.append("corpus_staleness")

    # Adaptive detection for subtle cases
    # Require both: statistical outlier AND value > 50% of baseline threshold
    shift_max = ctx.baseline["embedding_centroid_shift_max"]
    age_max = ctx.baseline["corpus_avg_doc_age_days_max"]
    if len(stats["centroid_shifts"]) >= 8:
        if "embedding_drift" not in alerts:
            cs_mean = _safe_mean(stats["centroid_shifts"])
            cs_std = _safe_std(stats["centroid_shifts"])
            if cs_std and cs_std > 0:
                z = (centroid_shift - cs_mean) / cs_std
                if z > 3.5 and centroid_shift > shift_max * 0.5:
                    alerts.append("embedding_drift_adaptive")

        if "corpus_staleness" not in alerts:
            da_mean = _safe_mean(stats["doc_ages"])
            da_std = _safe_std(stats["doc_ages"])
            if da_std and da_std > 0:
                z = (avg_doc_age - da_mean) / da_std
                if z > 3.5 and avg_doc_age > age_max * 0.5:
                    alerts.append("corpus_staleness_adaptive")

    # Only track clean events
    if not alerts:
        stats["centroid_shifts"].append(centroid_shift)
        stats["doc_ages"].append(avg_doc_age)

    return Verdict(
        alert=bool(alerts),
        pillar="ai_infra",
        reason=", ".join(alerts) if alerts else "clean",
    )
