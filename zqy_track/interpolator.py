from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import math
import pandas as pd
import numpy as np


@dataclass
class CellInfo:
    row: int
    col: int
    value: Optional[float]
    raw: Any
    is_new: bool
    range_lo: Optional[float]
    range_hi: Optional[float]


def _parse_range(v: Any) -> Tuple[Optional[float], Optional[float]]:
    if isinstance(v, str) and "~" in v:
        try:
            a, b = v.split("~", 1)
            lo, hi = float(a), float(b)
            if lo > hi:
                lo, hi = hi, lo
            return lo, hi
        except Exception:
            return None, None
    return None, None


def _to_float(v: Any) -> Optional[float]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    try:
        return float(v)
    except Exception:
        return None


class Interpolator:
    def __init__(self, df: pd.DataFrame, main_attribute: str):
        self.df = df.copy()
        self.attr = main_attribute
        # Build position index: (row,col) -> CellInfo
        self.pos: Dict[Tuple[int, int], CellInfo] = {}
        # Some CSVs may store as objects; ensure numeric coercion for row/col
        for _, r in self.df.iterrows():
            row = int(r["row"]) if not pd.isna(r["row"]) else None
            col = int(r["col"]) if not pd.isna(r["col"]) else None
            if row is None or col is None:
                continue
            raw = r.get(self.attr, None)
            val = _to_float(raw)
            lo, hi = _parse_range(raw)
            # Detect element name from column if present, else from index
            if "Element" in self.df.columns:
                elem_name = str(r.get("Element", ""))
            else:
                elem_name = str(r.name)
            is_new = elem_name.startswith("NewElem")
            self.pos[(row, col)] = CellInfo(row=row, col=col, value=val, raw=raw, is_new=is_new, range_lo=lo, range_hi=hi)

        # Determine grid bounds from present entries
        rows = [rc[0] for rc in self.pos.keys()]
        cols = [rc[1] for rc in self.pos.keys()]
        self.min_row = min(rows) if rows else 1
        self.max_row = max(rows) if rows else 1
        self.min_col = min(cols) if cols else 1
        self.max_col = max(cols) if cols else 1

    def _get(self, row: int, col: int) -> Optional[CellInfo]:
        return self.pos.get((row, col))

    def _neighbors4(self, row: int, col: int) -> List[CellInfo]:
        out = []
        for rr, cc in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            ci = self._get(rr, cc)
            if ci is not None:
                out.append(ci)
        return out

    def _mean(self, values: List[Optional[float]]) -> Optional[float]:
        nums = [v for v in values if v is not None and not math.isnan(v)]
        if not nums:
            return None
        return float(sum(nums) / len(nums))

    def _vertical_context(self, row: int, col: int) -> List[float]:
        vals: List[float] = []
        for dr in (-2, -1, 1, 2):
            ci = self._get(row + dr, col)
            if ci and ci.value is not None:
                vals.append(ci.value)
        return vals

    def _horizontal_context(self, row: int, col: int) -> List[float]:
        vals: List[float] = []
        for dc in (-2, -1, 1, 2):
            ci = self._get(row, col + dc)
            if ci and ci.value is not None:
                vals.append(ci.value)
        return vals

    def _horizontal_two_step_mean(self, row: int, col: int) -> Optional[float]:
        vals = self._horizontal_context(row, col)
        if not vals:
            return None
        if len(vals) == 4:
            return float(sum(vals) / len(vals))
        else:
            return None

    def _direct_neighbor_values_with_cross_edge(self, row: int, col: int) -> List[float]:
        vals: List[float] = []
        for ci in self._neighbors4(row, col):
            if ci.value is not None:
                vals.append(ci.value)
        # Add cross-edge neighbor per spec on left/right edges
        extra = None
        if col == self.min_col:
            extra = self._get(row - 1, self.max_col)
        elif col == self.max_col:
            extra = self._get(row + 1, self.min_col)
        if extra and extra.value is not None:
            vals.append(extra.value)
        return vals

    def _enforce_range(self, row: int, col: int, pred: Optional[float]) -> Optional[float]:
        ci = self._get(row, col)
        if not ci or pred is None:
            return pred
        # Only enforce for NewElem entries that carry a declared range
        if ci.is_new and ci.range_lo is not None and ci.range_hi is not None:
            lo, hi = ci.range_lo, ci.range_hi
            if not (lo <= pred <= hi):
                return (lo + hi) / 2.0
        return pred

    def predict_at(self, row: int, col: int, attribute: Optional[str] = None) -> float:
        attr = attribute or self.attr
        if attr != self.attr:
            # Rebuild index for a different attribute on demand
            self.attr = attr
            self.__init__(self.df, main_attribute=attr)

        # 1) Corner specialized rule: use two-point linear extrapolation.
        # Prefer along-row if available, otherwise along-column.
        is_top = row == self.min_row
        is_bottom = row == self.max_row
        is_left = col == self.min_col
        is_right = col == self.max_col
        if (is_top and is_left) or (is_bottom and is_right):
            pred = None
            # Horizontal extrapolation
            if is_left:
                v1 = self._get(row, col + 1)
                v2 = self._get(row, col + 2)
                if v1 and v2 and v1.value is not None and v2.value is not None:
                    pred = 2 * v1.value - v2.value
            elif is_right:
                v1 = self._get(row, col - 1)
                v2 = self._get(row, col - 2)
                if v1 and v2 and v1.value is not None and v2.value is not None:
                    pred = 2 * v1.value - v2.value

            # If horizontal not possible, try vertical extrapolation
            if pred is None:
                if is_top:
                    v1 = self._get(row + 1, col)
                    v2 = self._get(row + 2, col)
                    if v1 and v2 and v1.value is not None and v2.value is not None:
                        pred = 2 * v1.value - v2.value
                elif is_bottom:
                    v1 = self._get(row - 1, col)
                    v2 = self._get(row - 2, col)
                    if v1 and v2 and v1.value is not None and v2.value is not None:
                        pred = 2 * v1.value - v2.value

            if pred is not None:
                return float(self._enforce_range(row, col, float(pred)))

        # 2) If four direct neighbors are all filled with numeric values
        neigh = self._neighbors4(row, col)
        if len(neigh) == 4 and all(ci.value is not None for ci in neigh):
            pred = self._mean([ci.value for ci in neigh])
            return float(self._enforce_range(row, col, pred))

        # 3) Top/bottom edge rule: use horizontal two-step context mean
        if row == self.min_row or row == self.max_row:
            pred = self._horizontal_two_step_mean(row, col)
            if pred is not None:
                return float(self._enforce_range(row, col, pred))

        # 4) Left/right edge rule with specified cross-edge neighbors
        # If at (row, min(col)), include (row-1, max(col)) as an extra horizontal neighbor.
        # If at (row, max(col)), include (row+1, min(col)) as an extra horizontal neighbor.
        if col == self.min_col or col == self.max_col:
            vals = self._direct_neighbor_values_with_cross_edge(row, col)
            if vals:
                pred = self._mean(vals)
                return float(self._enforce_range(row, col, pred))

        # 5) If surrounding contains NewElem vertically, use horizontal two-step context mean
        up = self._get(row - 1, col)
        down = self._get(row + 1, col)
        if (up and up.is_new) or (down and down.is_new):
            pred = self._horizontal_two_step_mean(row, col)
            if pred is not None:
                return float(self._enforce_range(row, col, pred))

        # 6) If surrounding contains NewElem horizontally, use up to three directions
        left = self._get(row, col - 1)
        right = self._get(row, col + 1)
        if (left and left.is_new) or (right and right.is_new):
            vals = []
            for ci in self._neighbors4(row, col):
                if ci.value is not None:
                    vals.append(ci.value)
            pred = self._mean(vals)
            return float(self._enforce_range(row, col, pred))

        # 7) Fallback to mean of available direct numeric neighbors
        vals = [ci.value for ci in neigh if ci.value is not None]
        if vals:
            pred = self._mean(vals)
            return float(self._enforce_range(row, col, pred))

        # 8) Last resort: use midpoint of declared range if any
        ci_tgt = self._get(row, col)
        if ci_tgt and ci_tgt.range_lo is not None and ci_tgt.range_hi is not None:
            return float((ci_tgt.range_lo + ci_tgt.range_hi) / 2.0)

        raise ValueError(f"Cannot interpolate at ({row},{col}) for attribute {self.attr}")
