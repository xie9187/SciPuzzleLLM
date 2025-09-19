from data_utils import AttributeSet, MinorityReport, minority_report_from_series
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
class TableState(object):
    def __init__(self, elem_df, test_df):
        super(TableState, self).__init__()
        self.elem_df = elem_df
        self.elem_attrs = list(elem_df.columns)
        self.elem_df['row'] = None
        self.elem_df['col'] = None
        self.test_df = test_df
    
    def infer_aset(self, numeric_tolerances, exclude):
        self.aset = AttributeSet.infer_from_df(
            df=self.elem_df,
            test_flag="Test",
            numeric_tolerances=numeric_tolerances,
            bind_df=True,
            exclude = exclude
            # 可选：为数值列提供 tol
        )
        return self.aset
    
    def fill_elem_posi(self, actions):
        for elem, row, col in actions:
            self.elem_df.loc[elem, ['row', 'col']] = [row, col]

    def append_new_elem(self, elems):
        for elem_attr in elems:
            # elem_attr = [str(val) for val in elem_attr]
            self.elem_df.loc[elem_attr[0]] = elem_attr[1:] 
        
    def sort_table(self, sort_col, ascending=True):
        self.raw_df = self.elem_df.copy()

        def _range_mean_key(v):
            # 将类似 "lo~hi" 的范围转为均值；其他情况尽量转 float；失败则 NaN
            if pd.isna(v):
                return np.nan
            # 已是数字
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            # 可能是字符串
            if isinstance(v, str):
                s = v.strip()
                if "~" in s:
                    parts = s.split("~", 1)                    
                    a = float(parts[0].strip())
                    b = float(parts[1].strip())
                    # 容错：无序时取中点
                    lo, hi = (a, b) if a <= b else (b, a)
                    return (lo + hi) / 2.0


        # 使用 key 将范围映射到排序键，但不改变实际显示值
        self.elem_df = self.elem_df.sort_values(
            sort_col,
            ascending=ascending,
            key=lambda s: s.apply(_range_mean_key),
        )

    def state_as_long_str(self, df=None):
        if df is None:
            return self.elem_df.to_string(index=True)
        else:
            return df.to_string(index=True)

    def att_table(self, attr_name):
        """
        生成属性值的表格可视化字符串
        参数:
            attr_name: 要显示的属性列名 
        返回:
            对齐的表格字符串，空白位置用空字符串表示
        """
        # 检查是否有元素已填入表格
        if self.elem_df[['col', 'row']].isna().all().all():
            return 'No element in table currently.'
        
        # 获取表格的行列范围
        max_row = int(self.elem_df['row'].max())
        max_col = int(self.elem_df['col'].max())
        
        # 创建空表格 (row+1行 x col+1列，因为从0或1开始计数)
        table = np.empty((max_row, max_col), dtype=object)
        table.fill('')
        
        # 填充已填入的元素
        filled = self.elem_df.dropna(subset=['col', 'row'])
        for _, row in filled.iterrows():
            r = int(row['row']) - 1  # 转换为0-based索引
            c = int(row['col']) - 1
            if 0 <= r < max_row and 0 <= c < max_col:
                attr_value = row[attr_name]
                if pd.api.types.is_float(attr_value) and not pd.isna(attr_value):
                    table[r, c] = f"{float(attr_value):.1f}"
                else:
                    table[r, c] = str(attr_value)
        
        # 计算每列最大宽度
        col_widths = [max(len(str(cell)) for cell in col) for col in table.T]
        
        # 生成表格字符串
        table_lines = []
        for r in range(max_row):
            line = '|' + ', '.join(
                f" {table[r, c]:^{col_widths[c]}} " for c in range(max_col)
            ) + '|'
            table_lines.append(line)
        
        return '\n'.join(table_lines)


    def gap_filling(self, predict_element, main_attr, ascending, selected_pairs):
        new_rows = []
        n_to_add = min(predict_element, len(selected_pairs)) 
        for idx in range(n_to_add):
            le, re = selected_pairs[idx]
            # 邻居的主属性值
            lv = pd.to_numeric(self.elem_df.loc[le, main_attr], errors='coerce') if le in self.elem_df.index else None
            rv = pd.to_numeric(self.elem_df.loc[re, main_attr], errors='coerce') if re in self.elem_df.index else None
            if pd.isna(lv) or pd.isna(rv):
                range_str = ''
            else:
                a, b = float(lv), float(rv)
                lo, hi = (a, b) if a <= b else (b, a)
                range_str = f"{lo:.3f}~{hi:.3f}"

            row_vals = []
            for col in self.elem_df.columns:
                if col == main_attr:
                    row_vals.append(range_str)
                else:
                    row_vals.append(None)
            new_name = f"NewElem{idx}"
            new_rows.append([new_name] + row_vals)

        
        self.append_new_elem(new_rows)
        # 重新更新状态文本以带上新元素
        self.sort_table(main_attr, ascending=ascending)
        sorted_state = self.get_complete_state()
        return sorted_state
    
    def find_matched_elements(self, df1):

        """
        以 df1 的顺序逐个样本在 test_df 中寻找最相似样本，匹配后不放回（贪心匹配）。
        相似度 = 交集列上逐列"是否匹配"的加权平均（使用 space_shares 作为权重），满分 1.0。
        - 数值列: |v1 - v2| <= attribute.tolerance 视为匹配
        - 非数值列: v1 == v2 视为匹配
        返回
        matches_df: 逐样本匹配报告（包含 df1 索引、匹配到的 test_df 索引、每列是否匹配、总体得分）
        col_match_rate: 各列平均匹配率（只统计参与比较的样本对；若某样本该列有 NaN 则跳过该对）
        约束
        - 要求 len(df1) == len(test_df)
        """
        if len(df1) != len(self.test_df):
            raise ValueError("df1 与 test_df 的样本量必须相等。")

        # 1) 列交集（并排除 row/col 等不参与列）
        common_cols = [c for c in df1.columns if c in self.test_df.columns]
        if not common_cols:
            raise ValueError("两个 DataFrame 没有可比较的公共列。")

        # 2) 创建属性映射字典和获取权重
        attr_dict = {attr.name: attr for attr in self.aset.attributes}
        weights = self.aset.space_shares()

        # 3) 打分函数（返回列级是否匹配的布尔字典 + 总分）
        def row_similarity(r1: pd.Series, r2: pd.Series):
            per_col_match = {}
            total_weight = 0
            weighted_sum = 0
            
            for col in common_cols:
                v1, v2 = r1[col], r2[col]
                # 缺失值：跳过该列
                if pd.isna(v1) or pd.isna(v2):
                    per_col_match[col] = np.nan  # 表示未纳入统计
                    continue

                # 使用 aset.attribute 获取属性信息
                if col in attr_dict:
                    attr = attr_dict[col]
                    if attr.kind == "numeric":
                        # 数值列：容差判断（允许数字字符串）
                        try:
                            f1, f2 = float(v1), float(v2)
                            ok = abs(f1 - f2) <= attr.tolerance
                        except Exception:
                            ok = (v1 == v2)
                    else:
                        # 字符/类别精确匹配
                        ok = (v1 == v2)
                else:
                    # 如果列不在属性中，默认精确匹配
                    ok = (v1 == v2)

                per_col_match[col] = bool(ok)
                
                # 加权计算
                if not pd.isna(ok):
                    weight = weights.get(col, 0)
                    weighted_sum += int(ok) * weight
                    total_weight += weight

            score = weighted_sum / total_weight if total_weight > 0 else 0.0
            return per_col_match, score

        # 4) 贪心匹配：为 df1 的每一行挑选 df2 中最优且未被占用的行
        unused_df2_idx = set(self.test_df.index.tolist())
        records = []

        for i1, r1 in df1.iterrows():
            best = (-1.0, None, None)  # (score, idx2, per_col_match)
            for i2 in list(unused_df2_idx):
                r2 = self.test_df.loc[i2]
                per_col_match, score = row_similarity(r1, r2)
                # 更高分优先；同分可按先遇到的 i2
                if score > best[0]:
                    best = (score, i2, per_col_match)

            score, i2, per_col_match = best
            if i2 is not None:
                unused_df2_idx.remove(i2)

            row_out = {
                "df1_index": i1,
                "matched_df2_index": i2,
                "match_score": float(max(score, 0.0)),
            }
            # 展开每列是否匹配
            for col, ok in (per_col_match or {}).items():
                row_out[f"match_{col}"] = ok
                
            records.append(row_out)

        matches_df = pd.DataFrame.from_records(records).set_index("df1_index")
        matches_df['AllMatch'] = matches_df.loc[:, matches_df.columns.str.contains("match_Attribute")].all(axis=1)
        matches_df.loc['mean'] = matches_df.iloc[:, 1:].mean()
        
        return matches_df

    def clear_new_elems(self):
        self.elem_df = self.elem_df[~self.elem_df.index.str.startswith('NewElem')]

    def get_complete_state(self):
        state_str = 'All elements:\n'
        state_str += self.state_as_long_str() + '\n\n'
        for attr in self.elem_attrs:
            state_str += f'{attr} of current elements in virtual periodic table\n'
            state_str += self.att_table(attr) + '\n\n'
        return state_str
    
    def calculate_minority_rate(
        self,
        top_k: Optional[int] = None,
    ) -> Dict[Any, Dict[str, MinorityReport]]:
        """Return minority reports for discrete/categorical attributes grouped by table column."""
        if self.aset is None:
            raise ValueError("Attribute set is not initialized; call infer_aset first.")

        if "col" not in self.elem_df.columns:
            raise KeyError("elem_df must contain a 'col' column")

        reports: Dict[Any, Dict[str, MinorityReport]] = {}
        grouped = self.elem_df.dropna(subset=["col"]).groupby("col")

        for col_value, group_df in grouped:
            attr_reports: Dict[str, MinorityReport] = {}
            for attr in self.aset.attributes:
                if attr.name not in group_df.columns:
                    continue
                if attr.name in {"col", "row"}:
                    continue
                if attr.kind == "categorical" or (attr.kind == "numeric" and attr.numeric_type == "discrete"):
                    report = minority_report_from_series(group_df[attr.name], top_k=top_k)
                    if report is not None:
                        attr_reports[attr.name] = report
            if attr_reports:
                reports[col_value] = attr_reports

        return reports


    def summarize_minority_report(self, top_k: Optional[int] = None) -> str:
        """Render the minority distribution report in a compact, LLM-friendly text format."""
        reports = self.calculate_minority_rate(top_k=top_k)
        if not reports:
            return "No column exhibits a clear majority/minority pattern."

        def _sort_key(value):
            if isinstance(value, (int, float)):
                return (0, float(value))
            return (1, str(value))

        lines: List[str] = []
        for col_value in sorted(reports.keys(), key=_sort_key):
            attr_reports = reports[col_value]
            if not attr_reports:
                continue
            lines.append(f"col={col_value}:")
            for attr_name, report in attr_reports.items():
                if report.minority_object:
                    entries = "; ".join(f"{idx}: {val}" for idx, val in report.minority_object)
                else:
                    entries = "None"
                lines.append(
                    f"  - {attr_name}: majority={report.majority_value} ({report.majority_rate:.1%}), "
                    f"minority share={report.minority_rate:.1%}, minority entries={entries}"
                )
        return "\n".join(lines) if lines else "No column exhibits a clear majority/minority pattern."
