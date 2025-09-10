import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, Tuple, Optional
from sklearn.neighbors import NearestNeighbors



def sliding_gap_outliers(x, win=5, exclude_self=True):
    # Compute per-gap context stats using only windows that include the target gap.
    # mu/sigma are computed over the union of gaps covered by those windows (no duplicate counting).
    x, gaps = _compute_gaps(x)
    n_el = len(x)
    n_gap = len(gaps)
    win = max(2, int(win))
    records = []
    for i in range(n_gap):  # target gap index i between x[i] and x[i+1]
        # windows that include this gap satisfy: s <= i <= s + (win - 2)
        s_min = max(0, i - (win - 2))
        s_max = min(i, n_el - win)
        windows_covered = max(0, s_max - s_min + 1)
        # Union of gaps covered by any covering window is a contiguous block:
        # j in [i-(win-2), i+(win-2)] clipped to [0, n_gap-1]
        j_min = max(0, i - (win - 2))
        j_max = min(n_gap - 1, i + (win - 2))
        ctx_values = gaps[j_min:j_max + 1]
        if exclude_self:
            ctx_values = ctx_values[ctx_values != gaps[i]]
        mu = float(np.mean(ctx_values)) if ctx_values.size else float("nan")
        sigma = float(np.std(ctx_values, ddof=1)) if ctx_values.size > 1 else 0.0
        gap_val = float(gaps[i])
        is_outlier = (sigma > 0.0) and (gap_val > mu + sigma)
        records.append({
            "gap_index": i + 1,
            "left_value": x[i],
            "right_value": x[i + 1],
            "gap_value": gap_val,
            "windows_covered": windows_covered,
            "ctx_count": int(ctx_values.size),
            "mean_in_windows": mu,
            "std_in_windows": sigma,
            "is_outlier": is_outlier,
        })
    return pd.DataFrame(records)


def _trimmed_weighted_stats(values: np.ndarray, weights: Optional[np.ndarray], trim: float) -> Tuple[float, float, int]:
    """Compute trimmed, optionally weighted mean/std.

    - values: 1D array of context values
    - weights: same shape or None
    - trim: fraction to drop from each tail by value (0..0.45)
    Returns: (mean, std, count_kept)
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), 0.0, 0
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape != v.shape:
            raise ValueError("weights shape must match values")
    else:
        w = None

    # Trim by value order
    order = np.argsort(v)
    v_sorted = v[order]
    w_sorted = w[order] if w is not None else None
    m = len(v_sorted)
    t = int(max(0, min(np.floor(trim * m), np.floor(0.45 * m))))
    if t > 0 and m - 2 * t >= 1:
        v_kept = v_sorted[t:m - t]
        w_kept = w_sorted[t:m - t] if w_sorted is not None else None
    else:
        v_kept = v_sorted
        w_kept = w_sorted

    if v_kept.size == 0:
        return float("nan"), 0.0, 0

    if w_kept is None:
        mu = float(np.mean(v_kept))
        # ddof=1 if possible
        std = float(np.std(v_kept, ddof=1)) if v_kept.size > 1 else 0.0
        return mu, std, int(v_kept.size)
    else:
        wpos = np.maximum(0.0, w_kept)
        sw = float(np.sum(wpos))
        if sw <= 0:
            mu = float(np.mean(v_kept))
            std = float(np.std(v_kept, ddof=1)) if v_kept.size > 1 else 0.0
            return mu, std, int(v_kept.size)
        mu = float(np.sum(wpos * v_kept) / sw)
        var = float(np.sum(wpos * (v_kept - mu) ** 2) / sw) if v_kept.size > 1 else 0.0
        std = float(np.sqrt(max(var, 0.0)))
        return mu, std, int(v_kept.size)


def robust_multiscale_gap_voting(
    x: Iterable[float],
    k: float = 1.0,
    trim: float = 0.1,
    exclude_self: bool = True,
    weighted: bool = True,
    multi_scales: list[int] = None,
    min_ctx: int = 3,
    min_votes: int = 1,
) -> pd.DataFrame:
    """Improved sliding-gap outlier detection.

    Key improvements over `sliding_gap_outliers`:
    - Excludes the target gap from its own context to avoid self-influence.
    - Uses trimmed statistics (drop `trim` fraction from each tail of context).
    - Optional triangular distance weights toward the target gap.
    - multi-scale consensus across several `win` sizes; flags if votes >= `min_votes`.

    Parameters
    - x: sorted sequence of values along the axis of interest
    - win: base window size (>= 3; larger widens the context union)
    - k: sigma multiplier; flag when z = (gap - mu)/std > k
    - trim: fraction (0..0.45) of context trimmed from each tail by value
    - exclude_self: if True, remove the target gap from context stats
    - weighted: if True, apply triangular weights by distance from target gap
    - multi_scales: list of `win` sizes to aggregate; includes `win` if None
    - min_ctx: minimum context points required to compute stats; otherwise fallback to untrimmed, unweighted local neighborhood
    - min_votes: for multi-scale, minimum number of scales exceeding threshold to flag
    """
    values, gaps = _compute_gaps(x)
    n_el = len(values)
    n_gap = len(gaps)


    scales = multi_scales

    def ctx_bounds(i: int, w: int) -> tuple[int, int]:
        # Union of gaps covered by any window including gap i under window size w
        r = max(1, w - 2)
        j_min = max(0, i - r)
        j_max = min(n_gap - 1, i + r)
        return j_min, j_max

    records = []
    for i in range(n_gap):
        gap_val = float(gaps[i])

        z_by_scale: list[tuple[int, float, float, float, int, int]] = []
        # each entry: (w, z, mu, std, ctx_count, windows_covered)
        for w in scales:
            j_min, j_max = ctx_bounds(i, w)
            idx = np.arange(j_min, j_max + 1)
            if exclude_self:
                idx = idx[idx != i]
            ctx = gaps[idx]
            # weights: triangular kernel centered at i
            if weighted and ctx.size > 0:
                dist = np.abs(idx - i).astype(float)
                radius = max(1.0, float(w - 2))
                ww = np.maximum(0.0, 1.0 - dist / radius)
            else:
                ww = None

            # Ensure enough context; if too small, use immediate neighbors without trimming/weights
            if ctx.size < min_ctx:
                nb = []
                if i - 1 >= 0: nb.append(gaps[i - 1])
                if i + 1 < n_gap: nb.append(gaps[i + 1])
                ctx2 = np.asarray(nb, dtype=float)
                mu, std, cc = _trimmed_weighted_stats(ctx2, None, 0.0)
            else:
                mu, std, cc = _trimmed_weighted_stats(ctx, ww, trim)

            # windows that include this gap for window size w
            s_min = max(0, i - (w - 2))
            s_max = min(i, n_el - w)
            windows_covered = max(0, s_max - s_min + 1)

            if std <= 0.0 or not np.isfinite(std):
                z = 0.0
            else:
                z = float((gap_val - mu) / std)
            z_by_scale.append((w, z, mu, std, cc, windows_covered))

        # aggregate across scales
        if len(z_by_scale) == 1:
            w, z, mu, std, cc, windows_covered = z_by_scale[0]
            votes = 1 if z > k else 0
            is_out = z > k
            best = (w, z, mu, std, cc, windows_covered)
        else:
            votes = sum(1 for _, z, *_ in z_by_scale if z > k)
            best = max(z_by_scale, key=lambda t: t[1])
            w, z, mu, std, cc, windows_covered = best
            is_out = votes >= max(1, int(min_votes))

        records.append({
            "gap_index": i + 1,
            "left_value": values[i],
            "right_value": values[i + 1],
            "gap_value": gap_val,
            "score": float(z),
            "votes": int(votes),
            "best_win": int(w),
            "mean_in_ctx": float(mu),
            "std_in_ctx": float(std),
            "ctx_count": int(cc),
            "windows_covered": int(windows_covered),
            "is_outlier": bool(is_out),
        })
    return pd.DataFrame(records)

def _compute_gaps(x: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Helper: returns (values, gaps) as float arrays.

    - values: shape (n,)
    - gaps: shape (n-1,) where gaps[i] = values[i+1] - values[i]
    """
    values = np.asarray(x, dtype=float)
    gaps = np.diff(values)
    return values, gaps




def rolling_median_mad_gap_outliers(x: Iterable[float], win: int = 5, k: float = 4) -> pd.DataFrame:
    """Rolling robust detection per gap using local median and MAD on gaps.

    - For each gap i, consider gaps in [i - w, i + w] where w = win//2.
    - Compute local median and MAD, flag if robust z > k.
    """
    values, gaps = _compute_gaps(x)
    n_gap = len(gaps)
    half = max(0, win // 2)
    records = []
    for i in range(n_gap):
        s = max(0, i - half)
        e = min(n_gap - 1, i + half)
        ctx = gaps[s:e + 1]
        if ctx.size:
            med = float(np.median(ctx))
            mad = float(np.median(np.abs(ctx - med)))
            scale = 1.4826 * mad if mad > 0 else 0.0
        else:
            med = float("nan"); scale = 0.0
        if scale == 0.0:
            rz = 0.0
            is_out = False
        else:
            rz = float((gaps[i] - med) / scale)
            is_out = rz > k
        records.append({
            "gap_index": i + 1,
            "left_value": values[i],
            "right_value": values[i + 1],
            "gap_value": float(gaps[i]),
            "local_median": med,
            "local_mad_scaled": scale,
            "robust_z": rz,
            "is_outlier": is_out,
        })
    return pd.DataFrame(records)


def knn_gap_outliers(
    x: Iterable[float],
    n_neighbors: int = 5,
    contamination: float = 0.05,
    mode: str = "kth",  # "kth" or "mean"
) -> pd.DataFrame:
    """KNN-based gap outlier detection on 1D gaps.

    - Uses scikit-learn NearestNeighbors to compute neighbor distances in gap space.
    - mode="kth": use k-th nearest neighbor distance (excluding self) as score.
    - mode="mean": mean of the first k neighbor distances (excluding self) as score.
    - Thresholding: if `contamination` is provided, flags top fraction by score; otherwise IQR on scores.
    """
    values, gaps = _compute_gaps(x)
    records = []
    if NearestNeighbors is None or gaps.size == 0:
        for i, g in enumerate(gaps):
            records.append({
                "gap_index": i + 1,
                "left_value": values[i],
                "right_value": values[i + 1],
                "gap_value": float(g),
                "knn_score": float("nan"),
                "threshold": float("nan"),
                "is_outlier": False,
            })
        return pd.DataFrame(records)

    X = gaps.reshape(-1, 1)
    # Need at least 2 points to have a neighbor other than self
    if len(X) < 2:
        for i, g in enumerate(gaps):
            records.append({
                "gap_index": i + 1,
                "left_value": values[i],
                "right_value": values[i + 1],
                "gap_value": float(g),
                "knn_score": 0.0,
                "threshold": float("nan"),
                "is_outlier": False,
            })
        return pd.DataFrame(records)

    k_eff = min(max(1, n_neighbors), len(X) - 1)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")  # +1 to include self
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X, return_distance=True)
    # distances[:, 0] = 0 (self). Exclude for scoring
    neigh_dists = distances[:, 1:]
    if mode == "mean":
        scores = np.mean(neigh_dists[:, :k_eff], axis=1)
    else:
        scores = neigh_dists[:, k_eff - 1]

    # Determine threshold
    if contamination is not None:
        thr = float(np.quantile(scores, 1.0 - contamination))
        flags = scores >= thr
    else:
        q1 = float(np.percentile(scores, 25))
        q3 = float(np.percentile(scores, 75))
        iqr = q3 - q1
        thr = q3 + 1.5 * iqr
        flags = scores > thr

    for i, g in enumerate(gaps):
        records.append({
            "gap_index": i + 1,
            "left_value": values[i],
            "right_value": values[i + 1],
            "gap_value": float(g),
            "knn_score": float(scores[i]),
            "threshold": float(thr),
            "is_outlier": bool(flags[i]),
        })
    return pd.DataFrame(records)


def compute_gap_detection_metrics(outlier_df: pd.DataFrame, test_attr: pd.Series, inclusive: bool = False) -> dict:
    """Metrics per user's definition using Attribute2 as the probe axis.

    - True Positive (TP gaps): an outlier gap whose (left_value, right_value) interval contains ≥1 test value.
    - False Positive (FP gaps): an outlier gap whose interval contains 0 test values.
    - False Negative (FN points): a test value that is not inside any outlier gap.
    - Recall uses point-level definition; Precision uses gap-level definition; F1 is their harmonic mean.
    """
    odf = outlier_df[outlier_df["is_outlier"] == True]
    gaps = list(zip(odf["left_value"].astype(float).to_list(), odf["right_value"].astype(float).to_list()))
    tv = pd.to_numeric(test_attr, errors="coerce").dropna().to_numpy(dtype=float)
    n_gaps = len(gaps)
    n_points = int(tv.size)

    # TP/FP for gaps
    tp_gaps = 0
    for (a, b) in gaps:
        if a > b:
            a, b = b, a
        mask = (tv >= a) & (tv <= b) if inclusive else (tv > a) & (tv < b)
        if np.any(mask):
            tp_gaps += 1
    fp_gaps = n_gaps - tp_gaps

    # TP/FN for test points (covered by any predicted gap?)
    covered = np.zeros(n_points, dtype=bool)
    for (a, b) in gaps:
        if a > b:
            a, b = b, a
        covered |= ((tv >= a) & (tv <= b)) if inclusive else ((tv > a) & (tv < b))
    tp_points = int(np.sum(covered))
    fn_points = n_points - tp_points

    precision = tp_gaps / (tp_gaps + fp_gaps) if (tp_gaps + fp_gaps) > 0 else 0.0
    recall = tp_points / (tp_points + fn_points) if (tp_points + fn_points) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp_gaps": int(tp_gaps),
        "fp_gaps": int(fp_gaps),
        "n_pred_gaps": int(n_gaps),
        "tp_points": int(tp_points),
        "fn_points": int(fn_points),
        "n_test_points": int(n_points),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

def main():
    # Example usage on the provided training data
    data_path = Path(__file__).parent.parent
    from periodic_table import generate_table, mask_table
    metrics_list = []
    for i in range(1000):
        df = generate_table()
        train_df, test_df = mask_table(df)
        series = train_df['Attribute2'].sort_values(ascending=True)


        # result_df = robust_multiscale_gap_voting(
        #         series,
        #         k=1.0,
        #         trim=0.1,
        #         exclude_self=True,
        #         weighted=True,
        #         multi_scales=[3,5,7],
        #         min_ctx=3,
        #         min_votes=2,
        # )
        result_df = sliding_gap_outliers(series, win=5, exclude_self=True)
        # Keep only outlier gaps and evaluate against held-out Attribute2
        result_df = result_df[result_df['is_outlier']]
        metrics = compute_gap_detection_metrics(result_df, test_df['Attribute2'], inclusive=False)
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    print(metrics_df.mean())


if __name__ == "__main__":
    main()
