import unittest
from pydantic import BaseModel
from table_agents_v2 import Agent
from code_executor import ComplexityVisitor
import ast
import subprocess
import tempfile
import os
import sys
'''
通过测试驱动开发来生成假设对应的代码
1. 通过假设和状态生成fail2pass代码
    #TODO:

'''


class TestAgent(Agent):
    """测试代理"""
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        你是一位专业的Python单元测试专家。你的任务是为给定的代码生成全面的单元测试，
        验证代码是否正确实现了预期功能。
        """

    def evaluate_complexity(self, code: str): 
        # McCabe
        tree = ast.parse(code)
        cv = ComplexityVisitor()
        cv.visit(tree)
        cyclomatic = 1 + cv.decisions
        return cyclomatic

    def generate_fail2pass(self, backgroud: str, 
                           hypothesis: str, 
                           max_retries: int = 3): 

        '''
        根据背景信息和假设信息生成fail2pass 代码
        '''
        user_prompt = f'''
请基于以下假设与数据，
编写 fail2pass 单元测试代码，目标是确保即使代码变异也能被检测出来。

<hypothesis>
{hypothesis}
</hypothesis>

<data>
{backgroud}
</data>

要求：
1. 高覆盖率：测试应覆盖主要函数逻辑、边界条件、异常情况。
2. 回归敏感性：确保对代码轻微修改（如逻辑符号/边界条件变动）能够触发测试失败。
3. 输入唯一性：测试函数的输入参数仅限于提供的 pandas DataFrame。
4. 可维护性：代码需结构清晰、包含必要注释，方便后续扩展。
5. 断言粒度：断言结果需具体，避免仅判断是否报错。

输出 python unittest 测试代码，按以下格式回复:
<code>
# 单元测试代码
</code>
        '''

        prompt_msgs = self.prompt(user_prompt)
        trial = 0
        while trial < max_retries:
            try:
                raw_response = self.get_LLM_response(prompt_msgs)
                if code:=self.extract_content('code', raw_response):
                    pass
                if '```' in raw_response:
                    code = '\n'.join([line for line in raw_response.splitlines() 
                                      if '```' not in line])
                return code
            except Exception as e:
                trial += 1
                print(f'{e}, 重试第{trial}次')
                     
        
    def exec_unittest(self, code:str, timeout: float = 120.0): 
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as f: 
            f.write(code)
            fname = f.name

        try:
            proc = subprocess.run(
                [sys.executable, fname],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                "return code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "timeout": False
            }
        except subprocess.TimeoutExpired as e: 
            return {
                "return code": -1,
                "stdout": e.stdout or "",
                "stderr": e.stderr or "",
                "timeout": True
            }
        finally:
            os.unlink(fname)

    def prompt(self, user_prompt): 
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]


            
class TestTDD(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None: 
        super().__init__(methodName)
        self.test_agent = TestAgent()
    
    def test_generate_fail2pass_response(self): 
        backgroud = '''
         Attribute1  Attribute2 Attribute3 Attribute4   row   col
Element                                                          
Elem859   35.453000          -1     State1      Type1  None  None
Elem609  112.411000           2     State2      Type2  None  None
Elem768   87.620000           2     State2      Type2  None  None
Elem345  137.327000           2     State2      Type2  None  None
Elem241   47.867000           4     State2      Type2  None  None
Elem835  140.907650           3     State2      Type2  None  None
Elem441   72.640000           4     State2      Type3  None  None
Elem733   69.723000           3     State2      Type2  None  None
Elem615   91.224000           4     State2      Type2  None  None
Elem904   51.996100           3     State2      Type2  None  None
Elem726   30.973762           5     State2      Type1  None  None
Elem916   14.006700          -3     State1      Type1  None  None
Elem435  106.420000           2     State2      Type2  None  None
Elem400   22.989769           1     State2      Type2  None  None
Elem935   10.811000           3     State2      Type3  None  None
Elem509   18.998403          -1     State1      Type1  None  None
Elem872  121.760000           3     State2      Type3  None  None
Elem418  138.905470           3     State2      Type2  None  None
Elem565  140.116000           4     State2      Type2  None  None
Elem366   40.078000           2     State2      Type2  None  None
Elem515   24.305000           2     State2      Type2  None  None
Elem782   44.955912           3     State2      Type2  None  None
Elem245   65.409000           2     State2      Type2  None  None
Elem631   95.940000           6     State2      Type2  None  None
Elem259   88.905850           3     State2      Type2  None  None
'''
        hypothesis = '''
Hypothesis:
1. 虚拟元素周期律假设：
   - 元素按主属性(Attribute1)值递增排列，每行包含一定范围的Attribute1值
   - 元素在列上的位置由其Attribute2值决定，Attribute2代表元素的"化合价"特性
   - 同一Attribute2值的元素在同一列，按Attribute1递增排列

2. 填入规则：
   - 行号确定：将Attribute1值划分为10个等宽区间，每个区间对应一行
   - 列号确定：列号 = Attribute2值 + 5（将负值转换为正数）
   - 冲突解决：如果位置已被占用，则向下移动一行
   - 确保所有位置(row,col)为正整数
'''
        response = self.test_agent.generate_fail2pass(backgroud, hypothesis)
        print(response)
        self.assertIsNotNone(response)

    def test_Failunitest(self): 
        code = '''
# 单元测试代码
import unittest
import pandas as pd
from your_module import fail2pass  # 假设fail2pass是你实现的函数名

class TestFail2Pass(unittest.TestCase):

    def setUp(self):
        # 初始化测试数据
        data = {
            'Attribute1': [
                35.453, 112.411, 87.620, 137.327, 47.867, 140.90765, 72.64, 69.723, 91.224,
                51.9961, 30.973762, 14.0067, 106.42, 22.989769, 10.811, 18.998403, 121.76,
                138.90547, 140.116, 40.078, 24.305, 44.955912, 65.409, 95.94, 88.90585
            ],
            'Attribute2': [
                -1, 2, 2, 2, 4, 3, 4, 3, 4, 3, 5, -3, 2, 1, -1, 3, 3, 4, 2, 2, 3, 2, 6, 3
            ],
            'Attribute3': [
                'State1', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 
                'State2', 'State2', 'State2', 'State1', 'State2', 'State2', 'State1', 'State2',
                'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2'
            ],
            'Attribute4': [
                'Type1', 'Type2', 'Type2', 'Type2', 'Type2', 'Type2', 'Type3', 'Type2', 'Type2',
                'Type2', 'Type1', 'Type1', 'Type2', 'Type2', 'Type1', 'Type3', 'Type3', 'Type2',
                'Type2', 'Type2', 'Type2', 'Type2', 'Type2', 'Type2'
            ],
            'row': [None]*24,
            'col': [None]*24
        }
        self.df = pd.DataFrame(data, index=[f"Elem{i}" for i in range(1, 25)])

    def test_calculate_row_and_col(self):
        # 期望的row和col的计算结果
        expected_row = [
            4, 11, 8, 13, 5, 14, 7, 6, 9, 7, 6, 2, 11, 3, 4, 10, 9, 12, 6, 7, 8, 7, 13, 8
        ]
        expected_col = [
            4, 7, 7, 7, 9, 8, 9, 8, 9, 8, 10, 2, 7, 6, 4, 8, 8, 9, 7, 7, 8, 7, 11, 8
        ]

        # 调用 fail2pass 函数进行计算并赋值给 'row' 和 'col'
        result_df = fail2pass(self.df.copy())

        # 断言行列值
        self.assertEqual(list(result_df['row']), expected_row, "Row values are incorrect!")
        self.assertEqual(list(result_df['col']), expected_col, "Column values are incorrect!")

    def test_no_conflict_in_positions(self):
        # 确保没有行列位置冲突
        result_df = fail2pass(self.df.copy())

        # 检查每个元素的行列位置是否都唯一
        positions = list(zip(result_df['row'], result_df['col']))
        self.assertEqual(len(positions), len(set(positions)), "There are duplicate positions!")

    def test_invalid_attribute2_values(self):
        # 测试 Attribute2 的非法值，检查是否正确转换成合法的列号
        invalid_df = self.df.copy()
        invalid_df['Attribute2'] = [-10, -5, -1, 0, 1, 10]  # 给出一些无效值
        
        result_df = fail2pass(invalid_df)
        
        # 检查所有列值是否大于等于 1 (列号应该是正整数)
        self.assertTrue((result_df['col'] >= 1).all(), "Invalid column numbers found!")

    def test_empty_dataframe(self):
        # 测试空DataFrame
        empty_df = pd.DataFrame(columns=['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'row', 'col'])
        
        result_df = fail2pass(empty_df)
        
        # 确保返回值也是空的
        self.assertTrue(result_df.empty, "Result should be an empty dataframe!")

    def test_large_dataframe(self):
        # 测试大数据量情况
        large_data = {
            'Attribute1': [i for i in range(1, 1001)],
            'Attribute2': [i % 10 for i in range(1, 1001)],
            'Attribute3': ['State1']*1000,
            'Attribute4': ['Type1']*1000,
            'row': [None]*1000,
            'col': [None]*1000
        }
        large_df = pd.DataFrame(large_data, index=[f"Elem{i}" for i in range(1, 1001)])

        # 检查函数是否能处理大数据
        result_df = fail2pass(large_df)
        self.assertEqual(len(result_df), 1000, "The size of the result dataframe is incorrect!")

if __name__ == '__main__':
    unittest.main()

### 测试内容说明：

1. **`test_calculate_row_and_col`**: 确保根据规则计算出的 `row` 和 `col` 列值正确。
2. **`test_no_conflict_in_positions`**: 检查 `row` 和 `col` 的组合没有重复，确保没有冲突。
3. **`test_invalid_attribute2_values`**: 测试 `Attribute2` 可能包含的无效值（例如负数或超出范围的值）并确保列号被正确转换为正整数。
4. **`test_empty_dataframe`**: 测试空数据框，确保函数不会出错并返回空数据框。
5. **`test_large_dataframe`**: 测试大规模数据集，验证函数能处理大数据并正确返回结果。

每个测试都检查了特定的边界条件和异常情况，确保即使代码有所变动，逻辑错误也会被及时发现。
'''
        self.assertEqual(self.test_agent.exec_unittest(code)['return code'], 1)
    
    def test_Successtest(self): 

        code = '''
import unittest
import pandas as pd
import numpy as np

# 假设的函数，待测试的函数逻辑
def fill_element_positions(df):
    """
    根据元素的属性(Attribute1 和 Attribute2)计算每个元素的行号(row)和列号(col)。
    计算规则如下：
    - 行号由Attribute1值划分为10个等宽区间，每个区间对应一行
    - 列号由Attribute2值 + 5，确保列号为正整数
    - 如果某个位置(row, col)已占用，则向下移动一行
    """
    df['row'] = pd.cut(df['Attribute1'], bins=10, labels=False, include_lowest=True)
    df['col'] = df['Attribute2'] + 5

    # 检查冲突并向下移动
    occupied = set()
    for index, row in df.iterrows():
        r, c = row['row'], row['col']
        while (r, c) in occupied:
            r += 1  # 向下移动一行
        occupied.add((r, c))
        df.at[index, 'row'] = r
        df.at[index, 'col'] = c

    return df


class TestFillElementPositions(unittest.TestCase):
    
    def setUp(self):
        # 准备测试数据
        self.data = {
            'Element': ['Elem526', 'Elem529', 'Elem982', 'Elem261', 'Elem855', 'Elem819', 'Elem158', 'Elem341', 
                        'Elem535', 'Elem751', 'Elem516', 'Elem721', 'Elem684', 'Elem229', 'Elem562', 'Elem514', 
                        'Elem886', 'Elem139', 'Elem533', 'Elem905', 'Elem680', 'Elem756', 'Elem561', 'Elem231', 
                        'Elem607'],
            'Attribute1': [1, 5, 16, 1, 18, 3, 11, 3, 12, 13, 11, 3, 2, 3, 13, 6, 2, 9, 14, 8, 3, 17, 1, 1, 15],
            'Attribute2': [22.989769, 50.941500, 127.600000, 6.941000, 20.179700, 144.242000, 63.546000, 44.955912, 
                           65.409000, 69.723000, 107.868200, 140.116000, 24.305000, 140.907650, 10.811000, 
                           51.996100, 40.078000, 58.933195, 28.085500, 55.845000, 138.905470, 35.453000, 
                           39.098300, 85.467800, 121.760000],
            'Attribute3': [1, 5, -2, 1, 0, 3, 2, 3, 2, 3, 1, 4, 2, 3, 3, 3, 2, 2, 4, 3, 3, -1, 1, 1, 3],
            'Attribute4': ['State2', 'State2', 'State2', 'State2', 'State1', 'State2', 'State2', 'State2', 'State2',
                           'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2', 'State2',
                           'State2', 'State2', 'State2', 'State1', 'State2', 'State2', 'State2'],
            'Attribute5': ['Type2', 'Type2', 'Type3', 'Type2', 'Type1', 'Type2', 'Type2', 'Type2', 'Type2', 'Type2',
                           'Type2', 'Type2', 'Type2', 'Type2', 'Type3', 'Type2', 'Type2', 'Type2', 'Type3', 'Type2',
                           'Type2', 'Type1', 'Type2', 'Type2', 'Type3'],
            'row': [None] * 25,
            'col': [None] * 25
        }

        self.df = pd.DataFrame(self.data)

    def test_fill_element_positions(self):
        # 执行填充操作
        result = fill_element_positions(self.df.copy())

        # 检查行和列的填充结果是否正确
        self.assertTrue(result['row'].isnull().sum() == 0, "Row values should not be null")
        self.assertTrue(result['col'].isnull().sum() == 0, "Col values should not be null")
        
        # 确保行列位置为正整数
        self.assertTrue((result['row'] >= 0).all(), "Row values must be positive integers")
        self.assertTrue((result['col'] >= 0).all(), "Col values must be positive integers")
        
        # 检查是否有冲突
        occupied_positions = set(zip(result['row'], result['col']))
        self.assertEqual(len(occupied_positions), len(result), "There should be no overlapping positions")

    def test_edge_cases(self):
        # 检查边界条件，Attribute1最大值和最小值
        max_attr1_df = self.df.copy()
        max_attr1_df['Attribute1'] = 99999  # 设置最大值，看看是否能正确处理
        
        min_attr1_df = self.df.copy()
        min_attr1_df['Attribute1'] = -99999  # 设置最小值，看看是否能正确处理
        
        # 执行填充并验证
        max_attr1_result = fill_element_positions(max_attr1_df)
        min_attr1_result = fill_element_positions(min_attr1_df)
        
        self.assertTrue((max_attr1_result['row'] > 0).all(), "Max Attribute1 test: Row values must be positive integers")
        self.assertTrue((min_attr1_result['row'] > 0).all(), "Min Attribute1 test: Row values must be positive integers")
    
    def test_column_assignment(self):
        # 测试列号赋值逻辑
        result = fill_element_positions(self.df.copy())
        
        # 确保列号遵循 col = Attribute2 + 5 的规则
        for _, row in result.iterrows():
            self.assertEqual(row['col'], row['Attribute2'] + 5, "Column value is not correct")
    
    def test_conflict_resolution(self):
        # 测试冲突解决逻辑
        df_conflict = self.df.copy()
        df_conflict['Attribute1'] = [1]*25  # 所有元素的Attribute1相同，可能产生冲突
        
        result = fill_element_positions(df_conflict)
        
        # 检查是否所有位置(row, col)都唯一
        occupied_positions = set(zip(result['row'], result['col']))
        self.assertEqual(len(occupied_positions), len(result), "Conflict resolution failed: Positions are overlapping")

if __name__ == '__main__':
    unittest.main()
'''
        self.assertEqual(self.test_agent.exec_unittest(code)['return code'], 0)


