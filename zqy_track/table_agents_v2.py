import openai
import time
import re
import json
import os
import getpass

class Agent(object):

    def __init__(self):
        super(Agent, self).__init__()
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        if not os.environ.get("BASE_URL"):
            os.environ["BASE_URL"] = getpass.getpass("Enter BASE URL: ")
        self.key = os.environ.get("OPENAI_API_KEY") #我的key
        self.url = os.environ.get("BASE_URL")
        self.model = 'gpt-5' # 'deepseek-r1', 'deepseek-v3', 'o1', 'gpt-4o'

    def get_LLM_response(self, prompt_msgs):    
        # Set API key and base URL
        client = openai.OpenAI(api_key=self.key, base_url=self.url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_msgs,
            timeout=60,  # 添加60秒超时
        )
        response_time = time.time() - start_time
        print(f'Got response from {self.model} in {response_time:.1f} sec.')
        answer = response.choices[0].message.content.strip()
        return answer    

    def get_LLM_structured_response(self, prompt_msgs):
        client = openai.OpenAI(api_key=self.key, base_url=self.url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_msgs,
            response_format={
                "type": "json_object"
            },
            timeout=60,  # 添加60秒超时
        )
        response_time = time.time() - start_time
        print(f'Got response from {self.model} in {response_time:.1f} sec.')
        answer = response.choices[0].message.content.strip()
        return answer    
    
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

    def generate_hypothesis(self, state, attribute, history):    
        
        task = f"""
        1. 识别在主属性顺序排列时其他属性是否呈现以及呈现何种周期规律
        2. 提出最可能的虚拟元素周期律假设，使尽可能多的元素满足
        3. 根据假设建立将所有元素填入虚拟元素周期表格的规则
        4. 根据所建立的规则以python代码形式编写一段将元素填入虚拟周表格的函数
        5. 要求代码根据元素属性为每个元素确定合适且唯一的表格位置(row,col)
        6. 要求所得行(row)和列(col)必须为大于0的正整数
        7. 要求所填表格尽量紧凑，但允许元素间存在间隔（尚未发现的元素）
        8. 必须确保元素位置没有重叠
        """

        code_requrement = f"""
        1. 输入: 
            current_df: DataFrame # 包含元素所有属性以及当前所在表格的行(row)和列(col)
        2. 输出：
            results: [(elem_name, row, col), ...] # 所有元素位置的Python list, rol 和 col 为正整数
        3. 注意导入函数所需的package
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

    def predict_elements(self, state, hypothesis, code, history,n=10):

        task = f"""
        1. 根据提供的虚拟元素周期律假设以及当前的元素周期表状态，提出{n}个最有可能的潜在未知元素，元素命名为NewElem1, NewElem2, ...
        2. 给出所有潜在未知元素在元素周期表中的行(row)和列(col)，尽量在已知元素附近
        3. 必须确保未知元素之间以及与已知元素在元素周期表的位置没有重叠
        4. 根据提供的虚拟元素周期律假设和填表函数代码（输入为元素属性，输出为表中位置），写出其对应的逆函数代码
        5. 逆函数要求以潜在元素在元素周期表中位置为输入，输出其所有属性值
        6. 注意元素属性值的变量类型，可能是数字也可能是字符串 
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

        已知逆函数为：
        <inverse_code>
        {history.records[-1]['deduction_code'] if len(history.records) > 1 else ''}
        </inverse_code>

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
        匹配成功率{match_rate}
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

