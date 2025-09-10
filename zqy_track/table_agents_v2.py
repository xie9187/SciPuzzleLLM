import openai
import time
import re
import json
import os
import getpass
from typing import Any, Dict, List, Optional
import pandas as pd
from interpolator import Interpolator

class Agent(object):

    def __init__(self):
        super(Agent, self).__init__()
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        if not os.environ.get("BASE_URL"):
            os.environ["BASE_URL"] = getpass.getpass("Enter BASE URL: ")
        self.key = os.environ.get("OPENAI_API_KEY") #我的key
        self.url = os.environ.get("BASE_URL")
        self.model = 'claude-sonnet-4-20250514' # 'deepseek-r1', 'deepseek-v3', 'o1', 'gpt-4o', 'gpt-5'
        self.max_retries = 3
        self.base_delay = 2
        self.max_delay = 30
        self.timeout = 100

    def _exponential_backoff(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        return delay

    def _is_retryable_error(self, error: Exception) -> bool:
        s = str(error).lower()
        retryables = [
            "timeout", "timed out", "rate limit", "overloaded", "temporarily unavailable",
            "502", "503", "504", "gateway", "connection reset", "connection aborted",
        ]
        return any(r in s for r in retryables)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,  # "auto", "required", or None
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        timeout: Optional[int] = None,
        stream: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        return_raw: bool = True,
    ) -> Dict[str, Any]:
        """Unified chat request supporting tools and structured output.

        Returns a dict with: content, tool_calls, finish_reason, usage, response_time, response(raw).
        """
        client = openai.OpenAI(api_key=self.key, base_url=self.url, timeout=(timeout or self.timeout))
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        if temperature is not None:
            params["temperature"] = temperature
        if top_p is not None:
            params["top_p"] = top_p
        if extra:
            params.update(extra)

        start = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if stream:
                    # Aggregate streamed deltas
                    content_parts: List[str] = []
                    tool_calls: List[Dict[str, Any]] = []
                    response_obj = None
                    for event in client.chat.completions.create(stream=True, **params):
                        response_obj = event  # keep last event object
                        if not event.choices:
                            continue
                        delta = event.choices[0].delta  # type: ignore[attr-defined]
                        if getattr(delta, "content", None):
                            content_parts.append(delta.content)
                        if getattr(delta, "tool_calls", None):
                            # Some SDKs stream tool_calls incrementally; collect as-is
                            tool_calls.extend(delta.tool_calls)
                    content = ("".join(content_parts)).strip()
                    finish_reason = None
                    usage = None
                    response = response_obj
                else:
                    response = client.chat.completions.create(**params)
                    choice = response.choices[0]
                    msg = choice.message
                    content = (msg.content or "").strip()
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    finish_reason = getattr(choice, "finish_reason", None)
                    usage = getattr(response, "usage", None)

                elapsed = time.time() - start
                print(f'Got response from {self.model} in {elapsed:.1f} sec.')
                return {
                    "content": content,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "response_time": elapsed,
                    "response": response if return_raw else None,
                }
            except Exception as e:
                last_err = e
                print(f"Attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt >= self.max_retries - 1 or not self._is_retryable_error(e):
                    raise
                delay = self._exponential_backoff(attempt)
                time.sleep(delay)

    def get_LLM_response(self, prompt_msgs, response_format: Optional[Dict[str, Any]] = None) -> str:
        """Backward-compatible wrapper for plain or structured output.

        - Pass response_format={"type": "json_object"} for JSON mode.
        - For tools, prefer calling chat(..., tools=..., tool_choice="auto").
        """
        res = self.chat(prompt_msgs, response_format=response_format)
        return res.get("content", "")
    
    def get_LLM_structured_response(self, prompt_msgs) -> str:
        res = self.chat(prompt_msgs, response_format={"type": "json_object"})
        return res.get("content", "")
    
    def get_LLM_tools_response(self, prompt_msgs, tools):
        res = self.chat(prompt_msgs, tools=tools, tool_choice="required ")
        return res.get("content", "")
    
    # 解析响应内容
    def extract_content(self, tag, text):
        pattern = re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL)
        match = pattern.search(text)
        if tag == 'code':
            if not match:
                return '\n'.join([line for line in text.splitlines() 
                                      if '```' not in line])
        return match.group(1).strip() if match else None

    def reflaction(self, task, raw_response):
        pass

class AbductionAgent(Agent):

    def __init__(self):
        super(AbductionAgent, self).__init__()  # 调用父类的初始化方法
        # 定义角色提示词
        self.system_prompt = """
        你是一位虚拟元素周期律研究专家，需要根据表格状态以及以往历史记录提出或修改虚拟元素周期性规律假设。
        """
        self.state_introduction = """
        1. 元素表格，包括元素名称（无任何实际含义）、元素属性、元素在虚拟周期表格的位置（若已被填入）
        2. 以虚拟元素表格形式呈现的所有已被填入表格元素属性值

        """

    def select_main_attribute(self, state, history):

        task = """
        1. 选择一个属性作为元素排布的主属性，使得其他属性能尽可能呈现周期变化规律
        2. 确定主属性是按升序还是降序排列
        """

        user_prompt = f"""
        当前虚拟元素周期表状态包括：
        <state introduction>
        {self.state_introduction}
        </state introduction>

        当前虚拟元素周期表状态为：
        <state>
        {state}
        </state>

        请按照如下要求完成：
        <task>
        {task}
        </task>

        以往的历史记录（包含假设、评价与预测元素成功匹配率）为：
        <history>
        {history.select_record()}
        </history>

        你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程，要求对比每个属性作为主属性的优劣并选出最有利于归纳虚拟元素周期律的主属性
        </reasoning>

        <attribute>
        此处给出主属性的名字
        </attribute>

        <ascending>
        此处给出是否按升序排列主属性，仅回复True或False
        </ascending>
        """

        # 构造符合API要求的messages列表
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = self.get_LLM_response(prompt_msgs)

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "attribute": self.extract_content("attribute", raw_response),
            "ascending": self.extract_content("ascending", raw_response),
            "raw_response": raw_response  # 保留原始响应以防解析失败
        }
    
    
    def generate_hypothesis(self, state, attribute, ascending, history):    
        
        task = f"""
        你的任务仅是确定周期表“形状”（每行/每列的长度或换行规则），以便在不改变主属性顺序的前提下，将元素依序填入表格：
        1. 形状可以不规则（不同周期长度不同），但必须满足：按主属性{'升序' if str(ascending).lower()=='true' else '降序'}排列时，沿着你定义的遍历顺序（例如逐行从左到右、按行依次向下），主属性值单调变化（不逆序）。
        2. 请论证该形状如何使“同一列的元素具有较相似的属性”，以及“同一行呈现某种周期性的变化”。
        3. 结合当前数据（注意其中一些 NewElem 的主属性为字符串范围如 "lo~hi"），明确给出你的形状描述（如各行长度列表，或可计算的换行规则）。
        4. 根据该形状，给出将所有元素映射到 (row, col) 的规则，且行列均为正整数，且位置不重叠。
        """

        code_requrement = f"""
        1. 输入: 
            current_df: DataFrame  # 包含全部元素属性以及当前所在表格的行(row)和列(col), index_col 是Elem 名称。
        2. 约束：
            - 使用主属性 {attribute} 且按 {'升序' if str(ascending).lower()=='true' else '降序'} 排列进行放置；若主属性为范围字符串 "lo~hi"，请使用 (lo+hi)/2 的均值参与排序；其他不可解析值跳到队尾。
            - 仅根据你确定的“形状”与顺序分配 (row, col)，不得与已分配位置重叠；所有位置均为正整数；可存在空位（尚未发现的元素）。
            - 代码需自包含、可直接运行；不要调用外部未定义函数；必要时可导入基础库（如 pandas、numpy、math）。
        3. 输出：
            results: [(elem_name, row, col), ...]  # Python 列表，所有元素的位置，row/col 为正整数, 其中elem_name 与current_df 中的name对应。
        """

        user_prompt = f"""
        当前虚拟元素周期表状态包括：
        <state introduction>
        {self.state_introduction}
        </state introduction>

        当前虚拟元素周期表状态为：
        <state>
        {state}
        </state>

        当前选择的主属性及其顺序为：
        <attribute>
        {attribute}, ascending={ascending}
        </attribute>

        任务要求：
        <task>
        {task}
        </task>

        代码要求：
        <code_requrement>
        {code_requrement}
        </code_requrement>

        以往的历史记录（包含假设、评价与预测元素成功匹配率）为：
        <history>
        {history.select_record()} 
        </history>

        你的回复必须严格遵循以下格式：
        <reasoning>
        给出确定“形状”的思路、依据和对列相似性/行周期性的解释；说明遍历顺序与保持主属性单调的方式。
        </reasoning>

        <hypothesis>
        1. 周期表形状的明确描述（如每行列数/换行规则）
        2. 逐元素放置的规则（保证主属性单调、无重叠、尽量紧凑）
        </hypothesis>

        <code>
        放置所有元素位置的函数代码
        </code>

        <func_name>
        将上述代码中函数的名称写在此处
        </func_name>
        """
        
        # 构造符合API要求的messages列表
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        raw_response = self.get_LLM_response(prompt_msgs)

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "hypothesis": self.extract_content("hypothesis", raw_response),
            "code": self.extract_content("code", raw_response),
            "func_name": self.extract_content("func_name", raw_response),
            "raw_response": raw_response  # 保留原始响应以防解析失败
        }

    def select_gap_candidates(self, state, main_attribute, ascending, outlier_table_text, n: int = 10):
        """
        基于多尺度空隙投票的候选表（仅包含 is_outlier 的行的字符串表示），选择最可能的 n 个空位。

        输入：
        - state: 当前表格文本（包含所有元素、属性、位置表格）
        - main_attribute: 主属性名
        - ascending: 是否升序
        - outlier_table_text: 候选空位文本表（包含 gap_index,left_elem,right_elem 等列）
        - n: 要选的空位数

        输出：
        - 一个 CSV 风格的段落：gap_index,left_elem,right_elem，每行一个记录，最多 n 行。
        """
        task = f"""
        1. 阅读候选空位表（仅包含 is_outlier=True 的条目），优先选择分数高、票数多、位于已知元素密集区的空位。
        2. 若候选数量少于 {n}，请结合主属性 {main_attribute} 的排序方向（ascending={ascending}）与当前表格分布，自行推断补充分布合理且不重复的空位（与已列候选不同）。
        3. 输出不超过 {n} 个，避免重复，尽量覆盖不同区段。
        4. 输出严格按 CSV 三列：gap_index,left_elem,right_elem；若为自行推断的额外空位，gap_index 可填 NA。
        """

        user_prompt = f"""
        当前虚拟元素周期表状态：
        <state>
        {state}
        </state>

        候选空位（由多尺度空隙投票筛出的 is_outlier 条目）：
        <outlier_table>
        {outlier_table_text}
        </outlier_table>

        任务要求：
        <task>
        {task}
        </task>

        输出格式：
        <top_gaps>
        gap_index,left_elem,right_elem
        ...（不超过 {n} 行）
        </top_gaps>
        """

        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        raw_response = self.get_LLM_response(prompt_msgs)

        top_gaps_text = self.extract_content("top_gaps", raw_response) or ""
        selected = []
        for line in top_gaps_text.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("gap_index"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue
            gi, le, re = parts
            selected.append((gi, le, re))

        return {
            "selected": selected[:n],
            "raw_response": raw_response,
        }

class DeductionAgent(Agent):
    
    def __init__(self):
        super(DeductionAgent, self).__init__()  # 调用父类的初始化方法
        # 定义角色提示词
        self.system_prompt = """
        你是一位虚拟元素周期表研究专家，需要根据给定的周期律假设和已填入元素的虚拟周期表预测潜在的未知元素。
        """
        self.state_introduction = """
        1. 元素表格，包括元素名称（无任何实际含义）、元素属性、元素在虚拟周期表格的位置（若已被填入）
        2. 以虚拟元素表格形式呈现的所有已被填入表格元素属性值
        """
    
    def predict_numeric(self, table):
        try :
            for attr in table.aset.attributes:
                if attr.numeric_type == 'continuous':
                    itp = Interpolator(table.elem_df, attr.name)
                    for idx, row in table.elem_df.iterrows():
                        if row.name.startswith('NewElem'):
                            table.elem_df.at[idx, attr.name] = itp.predict_at(row['row'], row['col'])
        except Exception as e:
            print(f"Error predicting numeric attribute: {e}")


    def predict_discrete_categorical_attribute(self, state, hypothesis, history, n=10):

        task = f"""
        1. 根据提供的虚拟元素周期律假设，写出一个预测“离散或类别属性”的函数。
        2. 函数必须严格使用如下签名：def predict_element_attributes(current_df: pd.DataFrame) -> list
           - 输入 current_df 为完整表（含已知与 NewElem），包含列：所有属性 + row + col
           - 输出 results 为 Python 列表：[(elem_name, attr1, attr2, ..., row, col), ...]
           - 已有的数值/字符串属性若已给定则直接复制；仅对离散/类别属性做规则推断
        3. 注意元素属性值的变量类型，可能是数字也可能是字符串。 
        4. 不得调用外部未定义的函数；可导入 pandas/numpy/math。
        """

        code_requrement = f"""
        1. 输入: 
            current_df: 当前元素表格DataFrame（index 为元素名，含所有属性及 row/col）
        2. 输出：
            results: [(elem_name, attr1, attr2, ..., row, col), ...] # 所有元素属性值与位置的Python list
        3. 函数签名必须是：predict_element_attributes(current_df)
        4. 不要依赖外部未定义函数；如需库请在代码顶部显式 import
        """

        user_prompt = f"""
        当前虚拟元素周期表状态包括：

        <state introduction>
        {self.state_introduction}
        </state introduction>

        当前虚拟元素周期表状态为：
        <state>
        {state}
        </state>

        当前的虚拟元素周期律假设为：
        <hypothesis>
        {hypothesis}
        </hypothesis>


        已知预测函数为：
        <code>
        {history.records[-1]['deduction_code'] if len(history.records) > 1 else ''}
        </code>

        任务要求：
        <task>
        {task}
        </task>

        代码要求：
        <core_requrement>
        {code_requrement}
        </core_requrement>

        你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程
        </reasoning>


        <code>
        此处给出代码
        </code>

        <func_name>
        将上述代码中函数的名称写在此处
        </func_name>
        """
        
        # 获取大模型响应
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = self.get_LLM_response(prompt_msgs)

        func_name = self.extract_content("func_name", raw_response) or "predict_element_attributes"

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "code": self.extract_content("code", raw_response),
            "func_name": func_name,
            "raw_response": raw_response
        }


class InductionAgent(Agent):
    def __init__(self):
        super(InductionAgent, self).__init__()
        # 定义角色提示词
        self.system_prompt = """
        你是一位虚拟元素周期律评审专家，需要根据对应的虚拟元素周期表格状态评价当前虚拟元素周期性规律，并最终做出决定。
        """
        self.state_introduction = """
        1. 元素表格，包括元素名称（无任何实际含义）、元素属性、元素在虚拟周期表格的位置（若已被填入）
        2. 以虚拟元素表格形式呈现的所有已被填入表格元素属性值

        """
        self.options = {
            'P': 'pass the hypothesis',
            'A': 'adjust the hypothesis without changing main attribute',
            'C': 'change the main attribute' 
        }

    def evaluate_hypothesis(self, state, hypothesis, matched_elem_str, match_rate):
        task = """
        1. 评价当前的虚拟元素周期表中所有元素的各属性是否呈现明显的周期规律
        2. 评价所预测潜在元素（以NewElem命名）的与测试集匹配情况，例如导致有不匹配的元素的可能原因
        3. 是否需要修改假设，若需要应该如何修改，包括是否需要更换主元素以及修改假设内容
        4. 在提供的选项中选择一个决策
        """

        user_prompt = f"""
        当前虚拟元素周期表状态包括：
        <state introduction>
        {self.state_introduction}
        </state introduction>

        当前虚拟元素周期表状态为：
        <state>
        {state}
        </state>

        当前的虚拟元素周期律假设为：
        <hypothesis>
        {hypothesis}
        </hypothesis>

        与测试集元素能匹配上的潜在新元素：
        <matched_elem>
        {matched_elem_str}
        匹配成功率:
        {match_rate}
        </new_elem>

        任务要求：
        <task>
        {task}
        </task>

        决策选项：
        <options>
        {self.options}
        </options>

        你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程
        </reasoning>

        <evaluation>
        此处给出对假设的评价以及修改建议
        </evaluation>

        <decision>
        此处给出决策结果，仅输出对应选项的字典key
        </decision>
        """
        
        # 获取大模型响应
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = self.get_LLM_response(prompt_msgs)

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "evaluation": self.extract_content("evaluation", raw_response),
            "decision": self.extract_content("decision", raw_response),
            "raw_response": raw_response
        }
    
class RecordAgent(Agent):
    def __init__(self):
        super(RecordAgent, self).__init__()
        # 定义角色提示词
        self.system_prompt = """
        你是一个假设总结评价专家，需要根据历史记录总结出最优假设。
        """
        self.history_introduction = """
        包含一个或多个假设，每个假设包含假设、评价与预测元素成功匹配率，以及主属性、主属性是否升序，填表函数代码，逆函数代码。
        
        """

    def merge_records(self, records):
        task = '''
        1. 根据提供的最优假设，以及最后的假设总结并优化假设。
        2. 假设尽可能清晰，便于后续转化为python代码。
        3. 输出的格式与单个历史记录一致，包括假设、评价与预测元素成功匹配率，以及主属性、主属性是否升序, 填表函数代码，逆函数代码。
        4. 请以JSON格式返回结果。
        '''
        user_prompt = f'''
        当前的历史记录为：
        <history introduction>
        {self.history_introduction}
        </history introduction> 
        
        任务要求：
        <task>
        {task}
        </task>
        
        历史记录：
        <history>
        {json.dumps(records, ensure_ascii=False)}
        </history>


        '''
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        json_response = self.get_LLM_structured_response(prompt_msgs)
        return json_response

