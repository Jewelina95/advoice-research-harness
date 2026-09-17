"""Read-only, development-partition evidence-overlap research utilities.

These diagnostics neither select clinical features nor validate clinical claims.
The replay checker verifies a declared graph/log, not the production execution.
"""

from __future__ import annotations

from collections import defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def historical_inventory(
    history: pd.DataFrame, metrics: list[dict[str, Any]]
) -> pd.DataFrame:
    """Preserve historical rows; exact names are candidates, not equivalence proof."""
    required = {"metric_name", "state_id", "metric_definition", "how_calculated"}
    if not required.issubset(history.columns):
        raise ValueError(f"Historical dictionary missing {sorted(required - set(history.columns))}")
    keep = [
        c for c in ["state_id", "state_name", "state_cluster", "metric_name",
                    "metric_modality", "metric_analysis_type", "metric_definition",
                    "how_calculated", "matched_current_columns", "final_role",
                    "current_evaluation_status", "literature_basis_from_spec"]
        if c in history
    ]
    out = history[keep].copy()
    out.insert(0, "historical_row", np.arange(1, len(out) + 1))
    runtime = {m["id"] for m in metrics}
    out["runtime_name_candidate"] = out["metric_name"].where(
        out["metric_name"].isin(runtime), ""
    )
    out["mapping_status"] = np.where(
        out["metric_name"].isin(runtime),
        "exact_name_only_definition_review_required", "unresolved_not_absent",
    )
    out["repeated_historical_name"] = out["metric_name"].duplicated(keep=False)
    return out


def historical_declared_reuse(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audit explicit historical column references, without inferring aliases.

    A shared declared column is a dependency-review candidate, not proof that
    either the historical extraction or two metric definitions are equivalent.
    """
    if not {"metric_name", "state_id", "matched_current_columns"}.issubset(history):
        raise ValueError("Historical column mapping fields are required")
    rows = []
    for position, (_, row) in enumerate(history.iterrows(), start=1):
        value = row["matched_current_columns"]
        if pd.isna(value):
            continue
        for column in sorted({part.strip() for part in str(value).split(";") if part.strip()}):
            rows.append({"historical_row": position, "metric_name": row["metric_name"],
                         "state_id": row["state_id"], "declared_historical_column": column})
    edges = pd.DataFrame(rows, columns=["historical_row", "metric_name", "state_id", "declared_historical_column"])
    reused = []
    for column, group in edges.groupby("declared_historical_column", sort=True):
        if len(group) > 1:
            reused.append({
                "declared_historical_column": column, "n_rows": len(group),
                "n_metric_names": group.metric_name.nunique(),
                "metric_names": ";".join(sorted(group.metric_name.unique())),
                "state_ids": ";".join(sorted(group.state_id.unique())),
                "scope": "historical_declared_dependency_not_runtime_equivalence",
            })
    return edges, pd.DataFrame(reused)


def state_overlap_tables(
    states: list[dict[str, Any]], metric_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Distinguish direct metric reuse from one reviewed formula relation."""
    if len({s["id"] for s in states}) != len(states):
        raise ValueError("Duplicate state identifiers")
    rows: list[dict[str, Any]] = []
    sets: dict[str, set[str]] = {}
    for state in states:
        ids, weights = state["metrics"], state["weights"]
        if len(ids) != len(weights) or len(set(ids)) != len(ids):
            raise ValueError(f"Invalid metric mapping in {state['id']}")
        if set(ids) - metric_ids:
            raise ValueError(f"Unknown metrics in {state['id']}")
        sets[state["id"]] = set(ids)
        for metric, weight in zip(ids, weights, strict=True):
            if not np.isfinite(weight) or weight < 0:
                raise ValueError("State weights must be finite and nonnegative")
            rows.append({"state_id": state["id"], "metric_id": metric,
                         "configured_weight": weight})
    edges = pd.DataFrame(rows, columns=["state_id", "metric_id", "configured_weight"])
    incidence = pd.DataFrame(
        [{"metric_id": m, **{s: int(m in ids) for s, ids in sets.items()}}
         for m in sorted(metric_ids)]
    )
    # This is a source-scoped relation, not a universal clinical equivalence.
    canonical = {"silence_fraction": "vad_occupancy", "voiced_fraction": "vad_occupancy"}
    pairs = []
    for left, right in combinations(sets, 2):
        a, b = sets[left], sets[right]
        ga = {canonical.get(m, m) for m in a}
        gb = {canonical.get(m, m) for m in b}
        pairs.append({
            "left_state": left, "right_state": right,
            "shared_metric_ids": ";".join(sorted(a & b)),
            "shared_metric_count": len(a & b),
            "definition_jaccard": len(a & b) / len(a | b) if a | b else None,
            "shared_reviewed_information_groups": ";".join(sorted(ga & gb)),
            "scope": "configuration_only_not_prediction_effect",
        })
    return edges, incidence, pd.DataFrame(pairs)


def training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous subject tables and use only explicitly marked train rows."""
    if not {"subject_id", "split"}.issubset(frame.columns):
        raise ValueError("Explicit subject_id and split columns are required")
    ids = frame["subject_id"].astype("string")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any() or ids.ne(ids.str.strip()).any():
        raise ValueError("Null/blank/duplicate/whitespace subject IDs; use a verified subject-level table")
    if "dataset_id" in frame:
        datasets = frame.dataset_id.astype("string").str.strip()
        if datasets.isna().any() or datasets.eq("").any() or datasets.nunique() != 1:
            raise ValueError("Mixed or unknown dataset identities are not allowed")
    if frame["split"].isna().any():
        raise ValueError("Unknown split labels")
    split = frame["split"].astype(str).str.strip().str.lower()
    allowed = {"train", "validation", "val", "dev", "test", "holdout"}
    if not set(split).issubset(allowed):
        raise ValueError(f"Unrecognized split labels: {sorted(set(split) - allowed)}")
    selected = frame.loc[split.eq("train")].copy()
    if selected.empty:
        raise ValueError("No explicitly marked training subjects")
    return selected


def metric_column(column: str, metric_ids: set[str]) -> tuple[str, str] | None:
    if column in metric_ids:
        return "overall", column
    if column.startswith("task_") and "__" in column:
        task, base = column[5:].split("__", 1)
        if task and base in metric_ids:
            return task, base
    return None


def _affine_relation(
    left: pd.Series, right: pd.Series, *, atol: float = 1e-9, rtol: float = 1e-7,
) -> tuple[bool, float | None, float | None, float | None]:
    """Test an observed numerical identity ``right = slope * left + intercept``.

    This is an implementation-level duplicate candidate. It does not establish
    conceptual or clinical equivalence, and it is deliberately not evaluated
    for constant inputs.
    """
    x = left.to_numpy(dtype=float)
    y = right.to_numpy(dtype=float)
    if len(x) < 3 or np.ptp(x) == 0:
        return False, None, None, None
    design = np.column_stack([x, np.ones(len(x))])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = slope * x + intercept
    residual = float(np.max(np.abs(predicted - y)))
    matched = bool(np.allclose(predicted, y, atol=atol, rtol=rtol))
    return matched, float(slope), float(intercept), residual


def overlap_statistics(
    frame: pd.DataFrame, metrics: list[dict[str, Any]], *,
    min_pairs: int = 20, threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Descriptive pairwise-complete Spearman correlations, never label tests.

    Each task scope and observed language gets a separate block. Unknown
    language blocks are retained but explicitly cannot support within-language
    claims. Absence of a column/value is not absence of a clinical state.
    """
    if min_pairs < 3 or not 0 < threshold <= 1:
        raise ValueError("Invalid analysis thresholds")
    train = training_rows(frame)
    definitions = {m["id"]: m for m in metrics}
    if len(definitions) != len(metrics):
        raise ValueError("Duplicate runtime metric identifiers")
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for column in train.columns:
        match = metric_column(column, set(definitions))
        if match:
            task, base = match
            groups[task].append((column, base))
    if not groups:
        raise ValueError("No explicitly registered metric columns")
    missing_names = {"", "nan", "n/a", "na", "none", "null", "<na>", "unknown"}

    def normalized_labels(column: str) -> pd.Series:
        labels = train[column] if column in train else pd.Series("unknown", index=train.index)
        labels = labels.fillna("unknown").astype(str).str.strip().str.lower()
        return labels.where(~labels.isin(missing_names), "unknown")

    languages = normalized_labels("language")
    row_tasks = normalized_labels("task_type")
    unconditioned = {"unknown", "unspecified", "all", "mixed", "multi_task", "multiple", "overall", "other"}
    coverage, pairs = [], []
    strata = sorted(set(zip(languages, row_tasks)))
    for language, row_task in strata:
        rows = train.loc[languages.eq(language) & row_tasks.eq(row_task)]
        for task, columns in sorted(groups.items()):
            values = rows[[c for c, _ in columns]].apply(pd.to_numeric, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan)
            for column, base in columns:
                observed = values[column].dropna()
                coverage.append({
                    "language": language, "task_scope": task, "row_task": row_task, "column": column,
                    "metric_id": base, "role": definitions[base]["role"],
                    "branch": definitions[base].get("branch", "unspecified"),
                    "report_permission": bool(definitions[base].get("report_permission", False)),
                    "n_train_stratum": len(rows), "n_observed": len(observed),
                    "missing_fraction": float(1 - len(observed) / len(rows)),
                    "n_unique": int(observed.nunique()),
                    "status": "unobserved" if observed.empty else
                    "constant" if observed.nunique() < 2 else "variable",
                })
            for (left, lm), (right, rm) in combinations(columns, 2):
                complete = values[[left, right]].dropna()
                n = len(complete)
                constant = n > 0 and min(complete.nunique()) < 2
                status = "insufficient_pairs" if n < min_pairs else "constant" if constant else "ok"
                rho = None
                same = complementary = affine = False
                slope = intercept = affine_residual = None
                if status == "ok":
                    rho = float(complete.corr(method="spearman").iloc[0, 1])
                    same = bool(np.allclose(complete[left], complete[right], atol=1e-9, rtol=1e-9))
                    complementary = bool(np.allclose(
                        complete[left] + complete[right], 1.0, atol=1e-9, rtol=1e-9
                    ))
                    affine, slope, intercept, affine_residual = _affine_relation(
                        complete[left], complete[right]
                    )
                pairs.append({
                    "language": language, "task_scope": task, "row_task": row_task,
                    "left": left, "right": right, "n_pairs": n, "status": status,
                    "left_role": definitions[lm]["role"],
                    "right_role": definitions[rm]["role"],
                    "left_branch": definitions[lm].get("branch", "unspecified"),
                    "right_branch": definitions[rm].get("branch", "unspecified"),
                    "rho": rho, "review_candidate": rho is not None and abs(rho) >= threshold,
                    "equal_on_observed_rows": same,
                    "sum_one_on_observed_rows": complementary,
                    "affine_on_observed_rows": affine,
                    "affine_slope": slope,
                    "affine_intercept": intercept,
                    "affine_max_abs_residual": affine_residual,
                    "reviewed_complement_formula": {lm, rm} == {"silence_fraction", "voiced_fraction"},
                    "language_conditioned": language not in unconditioned,
                    "task_conditioned": task.lower() not in unconditioned or row_task not in unconditioned,
                })
    metadata = {
        "n_input_records": len(frame), "n_train_records": len(train),
        "excluded_nontrain_records": len(frame) - len(train),
        "unique_subject_keys_do_not_verify_person_level_independence": True,
        "task_scopes": sorted(groups), "languages": sorted(languages.unique()),
        "row_task_labels": sorted(row_tasks.unique()),
        "dataset_id_checked": "dataset_id" in train,
        "n_registered_metric_columns": sum(map(len, groups.values())),
        "missing_overall_metric_ids": sorted(set(definitions) - set(train.columns)),
        "analysis": "descriptive_train_only_pairwise_complete_spearman",
        "p_values_computed": False, "feature_selection_performed": False,
        "min_pairs": min_pairs, "review_threshold": threshold,
    }
    return pd.DataFrame(coverage), pd.DataFrame(pairs), metadata


def overlap_components(
    coverage: pd.DataFrame, pairs: pd.DataFrame, *, threshold: float = 0.90,
) -> pd.DataFrame:
    """Build thresholded correlation components, retaining isolated metrics.

    Components are descriptive review units. They are not mutually exclusive
    clinical constructs and must not be used as an automatic deletion list.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Invalid overlap threshold")
    required_coverage = {
        "language", "task_scope", "row_task", "column", "metric_id",
        "role", "branch", "status",
    }
    required_pairs = {
        "language", "task_scope", "row_task", "left", "right", "status", "rho",
        "equal_on_observed_rows", "sum_one_on_observed_rows", "affine_on_observed_rows",
    }
    if not required_coverage.issubset(coverage):
        raise ValueError("Overlap component inputs are incomplete")
    if pairs.empty:
        pairs = pd.DataFrame(columns=sorted(required_pairs))
    elif not required_pairs.issubset(pairs):
        raise ValueError("Overlap component inputs are incomplete")

    output: list[dict[str, Any]] = []
    keys = ["language", "task_scope", "row_task"]
    for key, group in coverage.loc[coverage.status.eq("variable")].groupby(keys, sort=True):
        columns = sorted(group.column.unique())
        parent = {column: column for column in columns}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        mask = (
            pairs.language.eq(key[0]) & pairs.task_scope.eq(key[1]) &
            pairs.row_task.eq(key[2]) & pairs.status.eq("ok") &
            pairs.rho.abs().ge(threshold)
        )
        block_pairs = pairs.loc[mask].copy()
        for row in block_pairs.itertuples(index=False):
            if row.left in parent and row.right in parent:
                union(row.left, row.right)
        members: dict[str, list[str]] = defaultdict(list)
        for column in columns:
            members[find(column)].append(column)
        lookup = group.drop_duplicates("column").set_index("column")
        ordered = sorted((sorted(value) for value in members.values()), key=lambda value: value[0])
        for index, component in enumerate(ordered, start=1):
            selected = block_pairs.loc[
                block_pairs.left.isin(component) & block_pairs.right.isin(component)
            ]
            output.append({
                "language": key[0], "task_scope": key[1], "row_task": key[2],
                "component_id": f"C{index:03d}", "n_members": len(component),
                "members": ";".join(component),
                "metric_ids": ";".join(sorted(set(lookup.loc[component, "metric_id"]))),
                "branches": ";".join(sorted(set(lookup.loc[component, "branch"]))),
                "roles": ";".join(sorted(set(lookup.loc[component, "role"]))),
                "n_threshold_edges": len(selected),
                "max_abs_spearman": float(selected.rho.abs().max()) if len(selected) else None,
                "contains_exact_equal": bool(selected.equal_on_observed_rows.any()) if len(selected) else False,
                "contains_complement": bool(selected.sum_one_on_observed_rows.any()) if len(selected) else False,
                "contains_affine_relation": bool(selected.affine_on_observed_rows.any()) if len(selected) else False,
                "scope": "descriptive_correlation_component_not_feature_selection",
            })
    return pd.DataFrame(output, columns=[
        "language", "task_scope", "row_task", "component_id", "n_members",
        "members", "metric_ids", "branches", "roles", "n_threshold_edges",
        "max_abs_spearman", "contains_exact_equal", "contains_complement",
        "contains_affine_relation", "scope",
    ])


def effective_dimension_statistics(
    frame: pd.DataFrame, metrics: list[dict[str, Any]], *, min_rows: int = 20,
) -> pd.DataFrame:
    """Estimate complete-case rank dimensions within task/language strata.

    Rank-transformed standardized values reduce scale dependence. Effective
    rank and PCA coverage describe this observed matrix only; neither estimates
    the number of clinical constructs.
    """
    if min_rows < 3:
        raise ValueError("min_rows must be at least three")
    train = training_rows(frame)
    definitions = {metric["id"]: metric for metric in metrics}
    if len(definitions) != len(metrics):
        raise ValueError("Duplicate runtime metric identifiers")
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for column in train.columns:
        match = metric_column(column, set(definitions))
        if match:
            groups[match[0]].append((column, match[1]))
    if not groups:
        raise ValueError("No explicitly registered metric columns")

    missing = {"", "nan", "n/a", "na", "none", "null", "<na>", "unknown"}

    def labels(name: str) -> pd.Series:
        raw = train[name] if name in train else pd.Series("unknown", index=train.index)
        normalized = raw.fillna("unknown").astype(str).str.strip().str.lower()
        return normalized.where(~normalized.isin(missing), "unknown")

    languages, row_tasks = labels("language"), labels("task_type")
    result: list[dict[str, Any]] = []
    for language, row_task in sorted(set(zip(languages, row_tasks))):
        rows = train.loc[languages.eq(language) & row_tasks.eq(row_task)]
        for task_scope, columns in sorted(groups.items()):
            raw = rows[[column for column, _ in columns]].apply(pd.to_numeric, errors="coerce")
            raw = raw.replace([np.inf, -np.inf], np.nan)
            subsets: dict[str, list[str]] = {
                "all_variable": [
                    column for column, _ in columns
                    if raw[column].dropna().nunique() >= 2
                ]
            }
            for branch in sorted({definitions[base].get("branch", "unspecified") for _, base in columns}):
                subsets[f"branch:{branch}"] = [
                    column for column, base in columns
                    if definitions[base].get("branch", "unspecified") == branch
                    and raw[column].dropna().nunique() >= 2
                ]
            subsets["report_permitted"] = [
                column for column, base in columns
                if bool(definitions[base].get("report_permission", False))
                and raw[column].dropna().nunique() >= 2
            ]
            for subset, selected in subsets.items():
                complete = raw[selected].dropna() if selected else pd.DataFrame(index=rows.index[:0])
                record: dict[str, Any] = {
                    "language": language, "task_scope": task_scope, "row_task": row_task,
                    "metric_subset": subset, "n_train_stratum": len(rows),
                    "n_variable_metrics": len(selected), "n_complete_rows": len(complete),
                    "sample_to_feature_ratio": len(complete) / len(selected) if selected else None,
                    "n_less_than_p": bool(selected and len(complete) < len(selected)),
                    "matrix_rank": None, "effective_rank": None,
                    "dimensions_90pct": None, "dimensions_95pct": None,
                    "leading_dimension_fraction": None, "status": "ok",
                    "scope": "descriptive_observed_matrix_not_clinical_factor_count",
                }
                if not selected:
                    record["status"] = "no_variable_metrics"
                elif len(complete) < min_rows:
                    record["status"] = "insufficient_complete_rows"
                else:
                    ranked = complete.rank(method="average")
                    std = ranked.std(axis=0, ddof=0)
                    ranked = ranked.loc[:, std.gt(0)]
                    if ranked.empty:
                        record["status"] = "no_variable_metrics"
                    else:
                        z = (ranked - ranked.mean(axis=0)) / ranked.std(axis=0, ddof=0)
                        singular = np.linalg.svd(z.to_numpy(dtype=float), compute_uv=False)
                        eigenvalues = np.square(singular)
                        positive = eigenvalues[eigenvalues > np.finfo(float).eps * max(eigenvalues.max(), 1.0)]
                        mass = positive / positive.sum()
                        cumulative = np.cumsum(np.sort(mass)[::-1])
                        record.update({
                            "matrix_rank": int(len(positive)),
                            "effective_rank": float(np.exp(-(mass * np.log(mass)).sum())),
                            "dimensions_90pct": int(np.searchsorted(cumulative, .90) + 1),
                            "dimensions_95pct": int(np.searchsorted(cumulative, .95) + 1),
                            "leading_dimension_fraction": float(cumulative[0]),
                        })
                result.append(record)
    return pd.DataFrame(result)


def task_scope_reuse_statistics(
    frame: pd.DataFrame, metrics: list[dict[str, Any]], *, min_pairs: int = 20,
) -> pd.DataFrame:
    """Compare each task-scoped metric with its overall metric counterpart.

    An exact copy can be a legitimate routing alias, but it is not independent
    task evidence and must not receive a second vote downstream.
    """
    if min_pairs < 3:
        raise ValueError("min_pairs must be at least three")
    train = training_rows(frame)
    metric_ids = {metric["id"] for metric in metrics}
    missing = {"", "nan", "n/a", "na", "none", "null", "<na>", "unknown"}

    def labels(name: str) -> pd.Series:
        raw = train[name] if name in train else pd.Series("unknown", index=train.index)
        normalized = raw.fillna("unknown").astype(str).str.strip().str.lower()
        return normalized.where(~normalized.isin(missing), "unknown")

    languages, row_tasks = labels("language"), labels("task_type")
    rows_out: list[dict[str, Any]] = []
    task_columns = [
        (column, match[0], match[1]) for column in train.columns
        if (match := metric_column(column, metric_ids)) is not None and match[0] != "overall"
    ]
    for language, row_task in sorted(set(zip(languages, row_tasks))):
        block = train.loc[languages.eq(language) & row_tasks.eq(row_task)]
        for task_column, task_scope, metric_id in task_columns:
            if metric_id not in block:
                continue
            values = block[[metric_id, task_column]].apply(pd.to_numeric, errors="coerce")
            values = values.replace([np.inf, -np.inf], np.nan).dropna()
            status = "insufficient_pairs" if len(values) < min_pairs else (
                "constant" if min(values.nunique()) < 2 else "ok"
            )
            rho = None
            same = affine = False
            slope = intercept = residual = None
            if status == "ok":
                rho = float(values.corr(method="spearman").iloc[0, 1])
                same = bool(np.allclose(values[metric_id], values[task_column], atol=1e-9, rtol=1e-9))
                affine, slope, intercept, residual = _affine_relation(
                    values[metric_id], values[task_column]
                )
            rows_out.append({
                "language": language, "row_task": row_task, "task_scope": task_scope,
                "metric_id": metric_id, "overall_column": metric_id,
                "task_column": task_column, "n_pairs": len(values), "status": status,
                "rho": rho, "equal_on_observed_rows": same,
                "affine_on_observed_rows": affine, "affine_slope": slope,
                "affine_intercept": intercept, "affine_max_abs_residual": residual,
                "interpretation": "routing_alias_not_independent_evidence" if same else
                "task_specific_measurement_candidate",
            })
    return pd.DataFrame(rows_out, columns=[
        "language", "row_task", "task_scope", "metric_id", "overall_column",
        "task_column", "n_pairs", "status", "rho", "equal_on_observed_rows",
        "affine_on_observed_rows", "affine_slope", "affine_intercept",
        "affine_max_abs_residual", "interpretation",
    ])


def replay_check(
    parents: dict[str, list[str]], changed: set[str],
    recomputed: list[str], invalidated: set[str], active: set[str],
    *, revised_parents: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Check declared dependency closure, ordering and invalid active ancestors.

    A passing certificate does not verify the supplied graph's completeness,
    provenance, or that a model was actually reexecuted.
    """
    known = set(parents)
    after = parents if revised_parents is None else revised_parents
    if set(after) != known:
        raise ValueError("Retain the node inventory across revisions; mark withdrawn nodes invalid")
    referenced = set().union(*(set(v) for graph in (parents, after) for v in graph.values())) if parents else set()
    supplied = changed | set(recomputed) | invalidated | active | referenced
    if supplied - known:
        raise ValueError(f"Unknown dependency nodes: {sorted(supplied - known)}")
    if len(recomputed) != len(set(recomputed)):
        raise ValueError("Duplicate recomputation entries")
    try:
        order = list(TopologicalSorter(parents).static_order())
        after_order = list(TopologicalSorter(after).static_order())
    except CycleError as exc:
        raise ValueError("Dependency graph contains a cycle") from exc
    affected = set(changed) | invalidated | {n for n in known if set(parents[n]) != set(after[n])}
    while True:
        previous_size = len(affected)
        for graph, ordering in ((parents, order), (after, after_order)):
            for node in ordering:
                if set(graph[node]) & affected:
                    affected.add(node)
        if len(affected) == previous_size:
            break
    descendants = affected - changed
    unavailable = set(invalidated)
    for node in after_order:
        if set(after[node]) & unavailable:
            unavailable.add(node)
    available = (known - affected | changed) - unavailable
    ordering_errors = []
    invalid_recomputations = []
    successful = set()
    for node in recomputed:
        if node in unavailable:
            invalid_recomputations.append(node)
            continue
        if set(after[node]) - available:
            ordering_errors.append(node)
            continue
        available.add(node)
        successful.add(node)
    stale = descendants - successful - invalidated
    invalid_active = active & unavailable
    violations = {
        "stale_descendants": sorted(stale),
        "out_of_order_recomputations": ordering_errors,
        "recomputations_with_invalid_ancestry": invalid_recomputations,
        "active_nodes_with_invalid_ancestry": sorted(invalid_active),
        "recomputed_and_invalidated": sorted(set(recomputed) & invalidated),
    }
    return {
        "passed": not any(violations.values()),
        "affected_descendants": sorted(descendants), **violations,
        "graph_transition_checked": revised_parents is not None,
        "scope": "declared_graph_and_log_only_not_clinical_or_model_faithfulness_proof",
    }
