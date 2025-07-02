import openai
import time
import re

class Agent(object):

    def __init__(self):
        super(Agent, self).__init__()
        self.key = "sk-PumVcl4i7eUKuTl75c47E62774B1487b8cDb056b1d92D201" #我的key
        self.url = "https://api.bltcy.ai/v1"
        self.model = 'deepseek-v3' # 'deepseek-r1', 'deepseek-V3', 'o1', 'GPT-4o'

    def get_LLM_response(self, prompt_msgs):    
        # Set API key and base URL
        client = openai.OpenAI(api_key=self.key, base_url=self.url)
        start_time = time.time()
        response = client.chat.completions.create(
            model=self.model,
            messages=prompt_msgs
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
        你是一位虚拟元素周期律研究专家，需要根据表格状态以及以往历史记录（若有）提出可能的虚拟元素周期性规律假设。
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

        以往的历史记录（包含假设与评价）为：

        <history>
        {history}
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
        2. 提出最可能的1条虚拟元素周期律假设，使尽可能多的元素满足
        3. 根据假设建立将所有元素填入虚拟元素周期表格的规则
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

        以往的历史记录（包含假设与评价）为：

        <history>
        {history}
        </history>

        你的回复必须严格遵循以下格式：

        <reasoning>
        此处给出推理过程
        </reasoning>

        <hypothesis>
        1. 较为精简的虚拟元素周期律假设
        2. 将元素填入虚拟元素周期表的规则
        </hypothesis>
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
            "raw_response": raw_response  # 保留原始响应以防解析失败
        }

class DeductionAgent(Agent):
    
    def __init__(self):
        super(DeductionAgent, self).__init__()  # 调用父类的初始化方法
        # 定义角色提示词
        self.system_prompt = """
        你是一位虚拟元素周期表填充专家，需要根据给定的周期律假设将元素填入表格的适当位置。
        """
        self.state_introduction = """
        1. 元素表格，包括元素名称（无任何实际含义）、元素属性、元素在虚拟周期表格的位置（若已被填入）
        2. 以虚拟元素表格形式呈现的所有已被填入表格元素属性值

        """

    def fill_table(self, state, hypothesis):

        task = """
        1. 根据指定的虚拟元素周期律假设，为每个元素设定其在虚拟元素周期表格的行(row)、列(col)位置
        2. 为每个未放置元素(row和col为None的元素)确定合适且唯一的位置(row,col)
        3. 行(row)和列(col)必须为大于0的正整数，确保元素间位置无冲突，且无元素遗漏
        4. 允许元素间存在间隔（尚未发现的元素），但要避免整个表格过于稀疏
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

        任务要求：

        <task>
        {task}
        </task>

        你的回复必须严格遵循以下格式：

        <reasoning>
        1. 解释如何应用给定的周期律假设
        2. 说明每个元素位置的确定依据
        3. 分析可能存在的冲突及解决方案
        </reasoning>

        <action>
        Elem1, row1, col1
        Elem2, row2, col2
        ...
        </action>
        """
        
        # 获取大模型响应
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        raw_response = self.get_LLM_response(prompt_msgs)

        # 提取并处理actions
        action_text = self.extract_content("action", raw_response)
        actions = []
        if action_text:
            for line in action_text.split('\n'):
                if line.strip():
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) == 3:
                        elem, row, col = parts
                        actions.append((elem, int(row), int(col)))
        
        return {
            "task": task,
            "reasoning": self.extract_content("reasoning", raw_response),
            "actions": actions,
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

    def evaluate_hypothesis(self, state, hypothesis):
        task = """
        1. 当前的虚拟元素周期表中所有元素的各属性是否呈现明显的周期规律
        2. 表格中不符合当前假设规律的虚拟元素个数
        3. 是否需要修改假设
        4. 若需要修改假设，是否需要更换元素主属性
        5. 在提供的选项中选择一个决策
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
        此处给出较为精简的假设评价
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