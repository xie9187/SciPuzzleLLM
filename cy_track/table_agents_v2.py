import openai
import time
import re
import json
import os
import getpass
import random

class Agent(object):

    def __init__(self):
        super(Agent, self).__init__()

        #self.key = "sk-PumVcl4i7eUKuTl75c47E62774B1487b8cDb056b1d92D201" #我的key
        #self.url = "https://api.bltcy.ai/v1"
        #self.model = 'claude-opus-4-1-20250805-thinking' # 'deepseek-r1', 'deepseek-V3', 'o1', 'GPT-4o'


        self.key = "sk-v0LHdPEABytbnbbGvyy2WR9QLemde9EdYu52dzyiwrYg563L" #我的key
        self.url = "https://happyapi.org/v1"
        self.model = 'claude-3-7-sonnet-20250219-thinking' # 'deepseek-r1', 'deepseek-v3', 'o1', 'gpt-4o'
         # 重试配置
        self.max_retries = 5
        self.base_delay = 5
        self.max_delay = 60
        self.timeout = 150

    def _exponential_backoff(self, attempt: int) -> float:
        """指数退避算法"""
        delay = min(self.base_delay * (2 ** attempt) + random.uniform(0, 1), self.max_delay)
        return delay

    def _is_retryable_error(self, error) -> bool:
        """判断错误是否可重试"""
        retryable_errors = [
            '502 Bad Gateway',
            '503 Service Unavailable', 
            '504 Gateway Timeout',
            'ConnectionError',
            'TimeoutError',
            'InternalServerError',
            'new_api_error'
        ]
        
        error_str = str(error).lower()
        return any(retryable in error_str for retryable in retryable_errors)

    def get_LLM_response(self, prompt_msgs):    
        """ # Set API key and base URL
        client = openai.OpenAI(api_key=self.key, base_url=self.url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_msgs
        )
        response_time = time.time() - start_time
        print(f'Got response from {self.model} in {response_time:.1f} sec.')
        answer = response.choices[0].message.content.strip()
        return answer   """
        """获取LLM响应，具有重试和错误处理"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 设置API客户端
                client = openai.OpenAI(
                    api_key=self.key, 
                    base_url=self.url,
                    timeout=self.timeout
                )
                
                start_time = time.time()
                
                # 发送请求
                response = client.chat.completions.create(
                    model=self.model,
                    messages=prompt_msgs
                )
                
                response_time = time.time() - start_time
                print(f'✅ 从 {self.model} 获得响应，用时 {response_time:.1f} 秒')
                
                answer = response.choices[0].message.content.strip()
                return answer
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                print(f'❌ 尝试 {attempt + 1}/{self.max_retries} 失败: {error_msg}')
                
                # 检查是否可重试
                if not self._is_retryable_error(e):
                    print(f'❌ 不可重试的错误，直接抛出: {error_msg}')
                    raise e
                
                # 如果还有重试机会
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    print(f'⏳ 等待 {delay:.1f} 秒后重试...')
                    time.sleep(delay)
                else:
                    print(f'❌ 已达到最大重试次数 ({self.max_retries})')
        
        # 所有重试都失败了
        raise Exception(f"API请求失败，已重试 {self.max_retries} 次。最后错误: {last_error}")

    def get_LLM_response_with_tools(self, prompt_msgs, tools):
        """获取LLM响应，支持function calling"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 设置API客户端
                client = openai.OpenAI(
                    api_key=self.key, 
                    base_url=self.url,
                    timeout=self.timeout
                )
                
                start_time = time.time()
                
                # 发送请求，包含工具定义
                response = client.chat.completions.create(
                    model=self.model,
                    messages=prompt_msgs,
                    tools=tools,
                    tool_choice="auto"  # 让模型自动决定是否使用工具
                )
                
                response_time = time.time() - start_time
                print(f'✅ 从 {self.model} 获得带工具的响应，用时 {response_time:.1f} 秒')
                
                # 检查是否有工具调用
                if response.choices[0].message.tool_calls:
                    tool_call = response.choices[0].message.tool_calls[0]
                    print(f'🔧 检测到工具调用: {tool_call.function.name}')
                    
                    # 如果调用了make_decision函数，直接返回响应内容
                    if tool_call.function.name == "make_decision":
                        return response.choices[0].message.content or ""
                
                # 如果没有工具调用，返回普通响应
                answer = response.choices[0].message.content.strip()
                return answer
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                print(f'❌ 尝试 {attempt + 1}/{self.max_retries} 失败: {error_msg}')
                
                # 检查是否可重试
                if not self._is_retryable_error(e):
                    print(f'❌ 不可重试的错误，直接抛出: {error_msg}')
                    raise e
                
                # 如果还有重试机会
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    print(f'⏳ 等待 {delay:.1f} 秒后重试...')
                    time.sleep(delay)
                else:
                    print(f'❌ 已达到最大重试次数 ({self.max_retries})')
        
        # 所有重试都失败了
        raise Exception(f"API请求失败，已重试 {self.max_retries} 次。最后错误: {last_error}")
  

    def get_LLM_structured_response(self, prompt_msgs):
        """ client = openai.OpenAI(api_key=self.key, base_url=self.url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_msgs,
            response_format={
                "type": "json_object"
            }
        )
        response_time = time.time() - start_time
        print(f'Got response from {self.model} in {response_time:.1f} sec.')
        answer = response.choices[0].message.content.strip()
        return answer  """
        """获取结构化LLM响应，具有重试和错误处理"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # 设置API客户端
                client = openai.OpenAI(
                    api_key=self.key, 
                    base_url=self.url,
                    timeout=self.timeout
                )
                
                start_time = time.time()
                
                # 发送请求
                response = client.chat.completions.create(
                    model=self.model,
                    messages=prompt_msgs,
                    response_format={
                        "type": "json_object"
                    }
                )
                
                response_time = time.time() - start_time
                print(f'✅ 从 {self.model} 获得结构化响应，用时 {response_time:.1f} 秒')
                
                answer = response.choices[0].message.content.strip()
                return answer
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                print(f'❌ 尝试 {attempt + 1}/{self.max_retries} 失败: {error_msg}')
                
                # 检查是否可重试
                if not self._is_retryable_error(e):
                    print(f'❌ 不可重试的错误，直接抛出: {error_msg}')
                    raise e
                
                # 如果还有重试机会
                if attempt < self.max_retries - 1:
                    delay = self._exponential_backoff(attempt)
                    print(f'⏳ 等待 {delay:.1f} 秒后重试...')
                    time.sleep(delay)
                else:
                    print(f'❌ 已达到最大重试次数 ({self.max_retries})')
        
        # 所有重试都失败了
        raise Exception(f"API请求失败，已重试 {self.max_retries} 次。最后错误: {last_error}")
       
    
    # 解析响应内容
    def extract_content(self, tag, text):
        pattern = re.compile(f'<{tag}>(.*?)</{tag}>', re.DOTALL)
        match = pattern.search(text)
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

    def select_main_attribute(self, table,state, history):

        task = """
        1. 选择一个属性作为元素排布的主属性，使得其他属性能尽可能呈现按行周期变化、按列相似或渐变的规律
        2. 主属性必须严格按升序排列，不能有重复值，每个元素的主属性值必须唯一
        3. 周期表的行和列必须严格按照主属性值的大小顺序排列，确保主属性值从左到右、从上到下严格递增
        4. 如果发现主属性有重复值，必须重新选择主属性
        """

        user_prompt = f"""
        当前虚拟元素周期表状态包括：
        <state introduction>
        {self.state_introduction}
        </state introduction>

        虚拟元素及属性信息：
        <elem_df>
        {table.elem_df}
        </elem_df>

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

    def generate_hypothesis(self, table,state, attribute, history):    
        
        task = f"""
        1. 识别在主属性顺序排列时其他属性是否呈现以及呈现何种周期规律
        2. 提出最可能的虚拟元素周期律假设，使尽可能多的元素满足
        3. 根据假设分析每个周期的元素数，建立将所有元素填入虚拟元素周期表格的规则
        4. 根据所建立的规则以python代码形式编写一段将元素填入虚拟周表格的函数
        5. 编写代码要求表格中各元素按照主属性从小到大顺序排列，行列均不可乱序
        6. 要求每个元素都填入表中，根据元素属性为每个元素确定合适且唯一的表格位置(row,col)
        7. 允许每行或每列元素数量不同，但必须保证元素位置没有重叠
        8. 允许元素间存在间隔，因为表格中存在部分尚未发现的元素
        9. 当前虚拟元素周期表状态不是参考状态，不一定正确
        """

        code_requrement = f"""
        1. 输入: 
            current_df: DataFrame # 包含元素所有属性以及当前所在表格的行(row)和列(col)
        2. 输出：
            results: [(elem_name, row, col), ...] # 所有元素位置的Python list, rol 和 col 为正整数
        3. 注意导入函数所需的package
        """


        user_prompt = f"""
        当前选择的主属性为：
        <attribute>
        {attribute}
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

        当前虚拟元素周期表状态包括：
        <state introduction>
        {self.state_introduction}
        </state introduction>

        虚拟元素及属性信息：
        <elem_df>
        {table.elem_df}
        </elem_df>

        当前虚拟元素周期表状态为(上一轮效果不好时无需考虑)：
        <state>
        {state}
        </state>


        你的回复必须严格遵循以下格式：
        <reasoning>
        此处给出推理过程
        </reasoning>

        <hypothesis>
        1. 较为精简的虚拟元素周期律假设
        2. 将元素填入虚拟元素周期表的规则
        </hypothesis>

        <code>
        此处给出函数代码
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

    def predict_elements(self, state, hypothesis, code, history,n=1):

        task = f"""
        1. 根据主属性的缺失情况，预测可能缺失的元素，提出{n}个主属性不同、位置不同的最有可能的潜在未知元素，元素命名为NewElem1, NewElem2, ...
        2. 根据提供的虚拟元素周期律假设、当前的元素周期表状态和历史记录，预测{n}个潜在未知元素的各属性值
        3. 给出所有潜在未知元素在元素周期表中的行(row)和列(col)，潜在元素间不能有位置重复
        4. 必须确保未知元素的主属性值严格按升序排列，不能有重复值、负值和离群值
        5. 考虑历史记录中先前的预测元素，考虑induction中给的评估结果，进行潜在元素的比较和优化
        6. 必须确保元素之间以及与已知元素在元素周期表的位置没有重叠
        7. 周期表的排列必须遵循：主属性值小的元素在左上角，主属性值大的元素在右下角
        8. 避免生成多个高度相似且映射到同一真实测试元素的预测；若多个新元素指向同一个测试目标，只计一个覆盖，冗余预测应减少
        9. 根据提供的虚拟元素周期律假设和填表函数代码（输入为元素属性，输出为表中位置），写出其对应的逆函数代码，逆函数命名为inverse_func
        10. 逆函数要求以潜在元素在元素周期表中位置为输入，输出其所有属性值
        11. 注意元素属性值的变量类型，可能是数字也可能是字符串，注意避免使用中文标点，注意括号的闭合性
        """

        code_requrement = f"""
        1. 输入: 
            element_positions: [(elem_name, row, col), ...] # 所有元素位置的Python list
            current_df: 当前元素表格DataFrame
        2. 输出：
            results: [(elem_name, attr1, attr2, ..., row, col), ...] # 所有元素属性与位置的Python list
        3. 注意导入函数所需的package
        4. 注意不要调用外部未定义的函数
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

        已知填表函数为：
        <code>
        {code}
        </code>

        任务要求：
        <task>
        {task}
        </task>

        代码要求：
        <core_requrement>
        {code_requrement}
        </core_requrement>

        以往的历史记录（包含假设、评价与预测元素成功匹配率）为：
        <history>
        {history.select_record()} 
        </history>

        你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程
        </reasoning>

        <new_elem>
        NewElem1, row1, col1
        NewElem2, row2, col2
        ... 
        </new_elem>

        <inverse_code>
        此处给出代码
        </inverse_code>

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

        # 提取并处理new_elem
        new_elem_text = self.extract_content("new_elem", raw_response)
        new_elems_posi = []
        if new_elem_text:
            for line in new_elem_text.split('\n'):
                if line.strip():
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) == 3:
                        elem, row, col = parts
                        new_elems_posi.append((elem, int(row), int(col)))

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "new_elems_posi": new_elems_posi,
            "inverse_code": self.extract_content("inverse_code", raw_response),
            "func_name": self.extract_content("func_name", raw_response),
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
            #'C': 'change the main attribute',
            'R': 'rollback to previous version due to constraint violation or poor performance'
        }
        
        # 定义决策函数的工具描述
        self.decision_tools = [
            {
                "type": "function",
                "function": {
                    "name": "make_decision",
                    "description": "根据匹配率和假设表现做出决策",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "match_rate": {
                                "type": "number",
                                "description": "当前匹配率（0-1之间）"
                            },
                            "previous_match_rate": {
                                "type": "number",
                                "description": "之前的匹配率（0-1之间），用于比较"
                            },
                            "main_attribute_ordered": {
                                "type": "boolean",
                                "description": "主属性是否严格有序"
                            },
                            "constraint_violation": {
                                "type": "boolean",
                                "description": "是否存在约束违反"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "决策推理过程"
                            }
                        },
                        "required": ["match_rate", "previous_match_rate", "main_attribute_ordered", "constraint_violation", "reasoning"]
                    }
                }
            }
        ]

    def make_decision(self, match_rate, previous_match_rate, main_attribute_ordered, constraint_violation, reasoning):
        """
        决策函数：根据匹配率和假设表现做出决策
        
        Args:
            match_rate: 当前匹配率
            previous_match_rate: 之前的匹配率
            main_attribute_ordered: 主属性是否严格有序
            constraint_violation: 是否存在约束违反
            reasoning: 决策推理过程
            
        Returns:
            dict: 包含决策结果和详细信息的字典
        """
        # 决策逻辑
        if constraint_violation and match_rate < 0.1:
            # 存在约束违反，选择回滚
            decision = "R"
            decision_reason = "存在约束违反，需要回滚"
        elif match_rate > 0.8:
            # 主属性严格有序且匹配率>0.8，选择通过
            decision = "P"
            decision_reason = "主属性严格有序且表现优秀（匹配率>0.8），通过假设"
        elif  match_rate > 0.6:
            # 主属性严格有序且匹配率>0.6，选择通过
            decision = "P"
            decision_reason = "主属性严格有序且表现及格（匹配率>0.6），通过假设"
        elif match_rate > previous_match_rate:
            # 匹配率有提升，选择调整
            decision = "A"
            decision_reason = f"匹配率有提升（从{previous_match_rate:.3f}提升到{match_rate:.3f}），选择调整"
        elif match_rate < previous_match_rate:
            # 匹配率下降，选择回滚
            decision = "R"
            decision_reason = f"匹配率下降（从{previous_match_rate:.3f}下降到{match_rate:.3f}），选择回滚"
        else:
            # 匹配率无变化，根据当前表现决定
            if match_rate > 0.3:
                decision = "A"
                decision_reason = "匹配率无变化但表现尚可，选择调整"
            else:
                decision = "R"
                decision_reason = "匹配率无变化且表现不佳，选择回滚"
        
        return {
            "decision": decision,
            "decision_reason": decision_reason,
            "match_rate": match_rate,
            "previous_match_rate": previous_match_rate,
            "main_attribute_ordered": main_attribute_ordered,
            "constraint_violation": constraint_violation,
            "reasoning": reasoning
        }

    def evaluate_hypothesis(self, state, hypothesis, matched_elem_str, match_rate, avg_matched_score, previous_match_rate=0.0):
        task = """
        1. 评价当前的虚拟元素周期律是否符合约束条件：
           - 主属性是否严格按升序排列，除新添加的元素外不能有重复值
           - 元素是否按主属性值从小到大严格顺序排列
           - 行列是否按照主属性值严格递增排列
        2. 评价当前的虚拟元素周期表非主属性是否呈现明显按行周期变化、按列相似或渐变的规律
        3. 评价所预测潜在元素（以NewElem命名）的与测试集匹配情况，例如导致有不匹配的元素的可能原因
        4. 评估当前假设的整体表现（匹配率、规律性等）
        5. 评估是否有出现多个新元素与同一个目标匹配的情况，给出提醒
        6. 使用make_decision函数做出最终决策：
           - 如果违背约束条件（如除新添加的元素外主属性重复、排序错误等），选择R（回滚）
           - 如果匹配率有提升，即使表现仍不够理想，小幅调整假设，选择A（调整）
           - 如果当前假设表现变得更差（匹配率下降，规律性不明显），选择R（回滚）
           - 如果假设通过且表现良好（及格：匹配率>0.6,良好：匹配率>0.8，主属性严格有序），选择P（通过）
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

        平均匹配分数：
        <avg_matched_score>
        {avg_matched_score}
        </avg_matched_score>

        与测试集元素能匹配上的潜在新元素：
        <matched_elem>
        {matched_elem_str}
        匹配成功率{match_rate}
        </matched_elem>

        之前的匹配率：
        <previous_match_rate>
        {previous_match_rate}
        </previous_match_rate>

        任务要求：
        <task>
        {task}
        </task>

        决策选项：
        <options>
        {self.options}
        </options>

        请使用make_decision函数来做出最终决策。你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程，包括对约束条件、规律性和匹配率的评估
        </reasoning>

        <evaluation>
        此处给出对假设的评价以及修改建议
        </evaluation>

        <constraint_analysis>
        此处分析主属性是否严格有序以及是否存在约束违反
        </constraint_analysis>

        <decision_call>
        调用make_decision函数，传入以下参数：
        - match_rate: {match_rate}
        - previous_match_rate: {previous_match_rate}
        - main_attribute_ordered: [根据约束分析结果填写true或false]
        - constraint_violation: [根据约束分析结果填写true或false]
        - reasoning: [总结你的推理过程]
        </decision_call>
        """
        
        # 获取大模型响应
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # 使用function calling获取响应
        raw_response = self.get_LLM_response_with_tools(prompt_msgs, self.decision_tools)

        # 提取约束分析结果
        constraint_analysis = self.extract_content("constraint_analysis", raw_response)
        
        # 根据约束分析结果调用决策函数
        main_attribute_ordered = "true" in constraint_analysis.lower() if constraint_analysis else False
        constraint_violation = "violation" in constraint_analysis.lower() or "违反" in constraint_analysis if constraint_analysis else False
        
        # 调用决策函数
        decision_result = self.make_decision(
            match_rate=match_rate,
            previous_match_rate=previous_match_rate,
            main_attribute_ordered=main_attribute_ordered,
            constraint_violation=constraint_violation,
            reasoning=self.extract_content("reasoning", raw_response) or ""
        )

        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "evaluation": self.extract_content("evaluation", raw_response),
            "constraint_analysis": constraint_analysis,
            "decision": decision_result["decision"],
            "decision_reason": decision_result["decision_reason"],
            "decision_details": decision_result,
            "raw_response": raw_response
        }
    
class RecordAgent(Agent):
    def __init__(self):
        super(RecordAgent, self).__init__()
        # 定义角色提示词
        self.system_prompt = """
        你是一个假设总结评价专家，需要根据历史记录选择并总结最优假设。
        """
        self.history_introduction = """
        包含一个或多个假设，每个假设包含假设、评价与预测元素成功匹配率，以及主属性信息。
        
        """

    def merge_records(self, records):
        task = '''
        1. 根据提供的最优假设，以及最后的假设总结并优化假设。
        2. 假设尽可能清晰，便于后续转化为形式语言，如python代码。
        3. 输出的格式与单个历史记录一致，包括假设、评价与预测元素成功匹配率，以及主属性、主属性是否升序。
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
        {records}
        </history>

        '''
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        json_response = self.get_LLM_structured_response(prompt_msgs)
        return json_response

