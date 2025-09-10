from __future__ import annotations
import os
import sys
import re
from datetime import datetime
import json
from os.path import join
from viz_utils import create_periodic_table_plot
from table_agents_v2 import RecordAgent
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List, Tuple
import numpy as np
from pandas.api.types import is_numeric_dtype

def print_and_enter(content):
    print(content)
    print(' ')

class Logger:
    def __init__(self, log_path):
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_folder = join(log_path, timestamp)
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

        self.log_file = open(join(self.log_folder, 'log.txt'), 'w', encoding='utf-8')
        
        # 保存原始stdout
        self.original_stdout = sys.stdout
        
        # 设置同时输出到控制台和文件
        sys.stdout = self

    def write(self, message):
        self.original_stdout.write(message)  # 控制台输出
        self.log_file.write(message)         # 文件写入

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def new_part(self, part_name):
        syb_len = 40
        line_str = "\n" + "="*syb_len + f" {part_name} " + "="*syb_len
        return line_str

    def log_table_as_csv(self, df):
        df.to_csv(join(self.log_folder, 'table.csv'))

    def log_table_as_img(self, df, iteration=None):
        for attribute in list(df.columns):
            if attribute in ['row', 'col']:
                continue
            key_idx = -1 if attribute == 'KnownAndMatched' else 0
            file_name = f'table_{attribute}.png' if iteration is None else f'table_{iteration}iter_{attribute}.png'
            create_periodic_table_plot(df, attribute, join(self.log_folder, file_name), key_idx=key_idx)                    

class History(object):
    def __init__(self):
        super(History, self).__init__()
        self.records = ['']

    def update_records(self, hypothesis, evaluation, match_rate, eval_score, attribute=None, ascending=None, abduction_code=None, deduction_code=None):
        self.records.append({
            'hypothesis': hypothesis, 
            'evaluation': evaluation,
            'match_rate': match_rate,
            'eval_score': eval_score,
            'attribute': attribute,
            'ascending': ascending,
            'abduction_code': abduction_code,
            'deduction_code': deduction_code,
        })

    def show_records(self):
        if len(self.records) == 0:
            return 'empty history'
        else:
            hist_str = ''
            for i, record in enumerate(self.records):
                # 确保记录是字典类型
                if not isinstance(record, dict):
                    continue
                    
                hypothesis = record.get('hypothesis', 'N/A')
                evaluation = record.get('evaluation', 'N/A')
                eval_score = record.get('eval_score', 'N/A')
                match_rate = record.get('match_rate', 'N/A')
                abduction_code = record.get('abduction_code', 'N/A')
                deduction_code = record.get('deduction_code', 'N/A')
                hist_str += f'Iteration #{i+1}\n'
                hist_str += f'Hypothesis:\n{hypothesis}\n\n'
                hist_str += f'Evaluation:\n{evaluation}\n\n'
                hist_str += f'Eval Score:\n{eval_score}\n\n'
                hist_str += f'Match Rate: {match_rate}\n\n'
                hist_str += f'Abduction Code:\n{abduction_code}\n\n'
                hist_str += f'Deduction Code:\n{deduction_code}\n\n'
            return hist_str
        
    def load_records_from_log(self, log_path, iteration=-1):
        """
        Load records from a log file
        
        Args:
            log_path: Path to the log directory
            iteration: Specific iteration to load (-1 for all iterations)
        
        Returns:
            List of record dictionaries
        """
        log_file = join(log_path, 'log.txt')
        if not os.path.exists(log_file):
            print(f"Log file not found: {log_file}")
            return []
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by iteration sections
        iteration_sections = re.split(r'======================================== Iteration (\d+) ========================================', content)
        
        records = []
        for i in range(1, len(iteration_sections), 2):  # Skip empty first section
            iter_num = int(iteration_sections[i])
            iter_content = iteration_sections[i + 1] if i + 1 < len(iteration_sections) else ""
            
            # If specific iteration requested, skip others
            if iteration != -1 and iter_num != iteration:
                continue
            

            hypothesis_match = re.search(r'Hypothesis:\n(.*?)(?=\nCode:|$)', iter_content, re.DOTALL)
            
            hypothesis = hypothesis_match.group(1).strip() if hypothesis_match else ""
            
            # Extract evaluation from Induction Process
            evaluation_match = re.search(r'evaluation:\n(.*?)(?=\n\ndecision:|$)', iter_content, re.DOTALL)
            evaluation = evaluation_match.group(1).strip() if evaluation_match else ""
            
            # Extract match rate from Induction Process
            match_rate = 0.0
            match_rate_match = re.search(r'Match Rate: (\d+\.\d+)', iter_content)
            if match_rate_match:
                match_rate = float(match_rate_match.group(1))

            eval_score_match = re.search(r"eval score:\s*(.*?)\s*Name:\s*mean", iter_content, flags=re.S)

            # to get eval score from logs
            if eval_score_match:
                eval_score = eval_score_match.group(1)
            # Extract main attribute and ascending from Abduction Process
            main_attr_match = re.search(r'Main attribute:\n(.*?)(?=\n|$)', iter_content)
            attribute = ""
            ascending = True
            if main_attr_match:
                attr_line = main_attr_match.group(1).strip()
                attr_parts = attr_line.split()
                if len(attr_parts) >= 2:
                    attribute = attr_parts[0]
                    ascending = "ascending=True" in attr_line
            abduction_code_match = re.search(r'Code:\n([\s\S]*?)(?=^={5,}|\Z)', iter_content, re.MULTILINE)
            abduction_code = abduction_code_match.group(1).rstrip() if abduction_code_match else ""
            deduction_code_match = re.search(r'Inverse code:\n([\s\S]*?)(?=All|\Z)', iter_content, re.MULTILINE)
            deduction_code = deduction_code_match.group(1).rstrip() if deduction_code_match else ""
            record = {
                'hypothesis': hypothesis,
                'evaluation': evaluation,
                'match_rate': match_rate,
                'eval_score': eval_score,
                'attribute': attribute,
                'ascending': ascending,
                'abduction_code': abduction_code,
                'deduction_code': deduction_code,
            }
            records.append(record)
        self.records = records
        return records
    
    def select_record(self):
        if len(self.records) < 2:
            return json.dumps(self.records[-1], ensure_ascii=False)
        else:
            max_match_rate = max([record['match_rate'] for record in self.records[1:]])
            max_match_rate_record = [record for record in self.records[1:] if record['match_rate'] == max_match_rate]
            last_record = [self.records[-1]]
            merged_records = max_match_rate_record + last_record
            record_agent = RecordAgent()
            json_response = record_agent.merge_records(merged_records)
            return json_response
            

@dataclass(frozen=True)
class Attribute:
    """
    表示单个属性（数值/离散），并能计算自身的属性空间大小。
    - 数值属性：按 bin = floor(value / tolerance) 离散（锚定 0）
    - 离散属性：非空唯一值计数
    """
    name: str
    kind: str  # "numeric" | "categorical"
    tolerance: Optional[float] = None  # 仅 numeric 使用
    # 对数值型做进一步标注："continuous" | "discrete"（可选，不影响现有逻辑）
    numeric_type: Optional[str] = None

    @staticmethod
    def numeric(name: str, tolerance: float, numeric_type: Optional[str] = None) -> "Attribute":
        if tolerance is None or tolerance <= 0:
            raise ValueError(f"tolerance for numeric attribute '{name}' must be > 0")
        return Attribute(name=name, kind="numeric", tolerance=tolerance, numeric_type=numeric_type)

    @staticmethod
    def categorical(name: str) -> "Attribute":
        return Attribute(name=name, kind="categorical", tolerance=None)

    # --------- 计算空间大小 ---------
    def space_size(self, series: pd.Series) -> int:
        """
        计算该属性在给定数据（通常是训练子集）上的空间大小。
        对于不存在/空列，返回 0。
        """
        if series is None or len(series) == 0:
            return 0

        if self.kind == "numeric":
            tol = self.tolerance
            if tol is None or tol <= 0:
                raise ValueError(f"numeric attribute '{self.name}' requires a positive tolerance")

            # 过滤 NaN/非有限值
            s = pd.to_numeric(series, errors="coerce")
            s = s[np.isfinite(s.values)]
            if s.empty:
                return 0

            # 按 floor(x / tol) 分箱
            bins = np.floor(s.values / tol).astype(np.int64, copy=False)
            # 唯一 bin 数
            return int(pd.Series(bins).nunique(dropna=True))

        elif self.kind == "categorical":
            s = series[pd.notna(series)]
            if s.empty:
                return 0
            # 直接唯一值计数（空字符串视为有效类别）
            return int(pd.Series(s).astype(object).nunique(dropna=True))

        else:
            raise ValueError(f"unknown attribute kind: {self.kind}")


class AttributeSet:
    """
    属性集合：负责
      - 推断 DF 中的数值/离散属性并构造 Attribute 列表
      - 计算每个属性的空间、总空间、以及占比
    支持“数据绑定”：先 bind(df) 或 infer_from_df(..., bind_df=True)，
    后续可不再传 df 参数。
    """

    def __init__(self, attributes: List[Attribute], test_flag: str = "Test", bound_df: Optional[pd.DataFrame] = None):
        self.attributes: List[Attribute] = attributes
        self.test_flag: str = test_flag
        self._bound_df: Optional[pd.DataFrame] = bound_df

    @staticmethod
    def infer_from_df(
        df: pd.DataFrame,
        test_flag: str = "Test",
        numeric_tolerances: Optional[Dict[str, float]] = None,
        default_tolerance: float = 0.1,
        bind_df: bool = False,
        exclude: List[str] = [],
    ) -> "AttributeSet":
        """
        根据 DF 的 dtype 推断数值/离散列（排除 test_flag），
        为数值列应用 numeric_tolerances 或 default_tolerance。
        若 bind_df=True，将把 df 绑定进返回的 AttributeSet，之后可直接调用无参计算方法。
        """
        if numeric_tolerances is None:
            numeric_tolerances = {}

        def _classify_numeric_unique(series: pd.Series, threshold: float = 0.7) -> str:
            """
            仅使用唯一值占比(unique_ratio)来区分：
              - unique_ratio > threshold -> continuous
              - 否则 -> discrete
            空/无效样本时，按离散处理（保守）。
            """
            s = pd.to_numeric(series, errors="coerce")
            s = s[np.isfinite(s.values)]
            n = int(len(s))
            if n <= 0:
                return "discrete"
            unique_ratio = float(pd.Series(s).nunique(dropna=True)) / n
            return "continuous" if unique_ratio > threshold else "discrete"

        attributes: List[Attribute] = []
        for col in df.columns:
            if col == test_flag:
                continue
            if col in exclude:
                continue
            if is_numeric_dtype(df[col]):
                tol = numeric_tolerances.get(col, default_tolerance)
                numeric_type = _classify_numeric_unique(df[col])
                attributes.append(Attribute.numeric(col, tol, numeric_type=numeric_type))
            else:
                attributes.append(Attribute.categorical(col))

        return AttributeSet(attributes=attributes, test_flag=test_flag, bound_df=(df if bind_df else None))

    # --------- 绑定数据（可链式调用）---------
    def bind(self, df: pd.DataFrame) -> "AttributeSet":
        """
        绑定一个 DataFrame 到当前实例。绑定后，space_sizes/total_space/space_shares
        可不再传入 df 参数。
        注意：这里保存的是引用。如果希望“快照”，请自行传入 df.copy()。
        """
        self._bound_df = df
        return self

    # --------- 工具：解析 df 来源 ---------
    def _resolve_df(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if df is not None:
            return df
        if self._bound_df is not None:
            return self._bound_df
        raise ValueError("No DataFrame provided. Call .bind(df) first or pass df as an argument.")

    # --------- 取训练子集 ---------
    def _training_subset(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.test_flag in df.columns:
            mask = df[self.test_flag] == False  # noqa: E712
            return df.loc[mask]
        return df

    
    # --------- 计算：每列空间大小 ---------
    def space_sizes(self, df: Optional[pd.DataFrame] = None) -> Dict[str, int]:
        df = self._resolve_df(df)
        train_df = self._training_subset(df)
        result: Dict[str, int] = {}
        for attr in self.attributes:
            series = train_df[attr.name] if attr.name in train_df.columns else pd.Series(dtype="float64")
            result[attr.name] = attr.space_size(series)
        return result

    # --------- 计算：总空间大小（加和） ---------
    def total_space(self, df: Optional[pd.DataFrame] = None) -> int:
        sizes = self.space_sizes(df)
        total = 0
        for size in sizes.values():
            total += int(size)
            if total == 0:
                return 0
        return int(total)

    # --------- 计算：各属性空间占比 ---------
    def space_shares(self, df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        sizes = self.space_sizes(df)
        total = self.total_space(df)
        if total == 0:
            return {k: 0.0 for k in sizes.keys()}
        return {k: (v / total) for k, v in sizes.items()}

def execute_function(code_str, func_name, df):
    namespace = {}
    exec(code_str, namespace)
    function = namespace[func_name]
    result = function(df.copy()) 
    return result
