import multiprocessing
import time
import types
import unittest
import traceback
import json
import pandas as pd
from typing import Dict, List, Tuple, Any
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import tempfile
import os
from table_agents_v2 import Agent

# 确保Windows上的multiprocessing正常工作
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

class CodeExecutorService:
    """独立进程代码执行服务"""
    
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.debug_agent = DebugAgent()
        self.test_agent = TestAgent()
    
    def execute_code_in_process(self, code: str, func_name: str, input_data: Any) -> Dict:
        """在独立进程中执行代码"""
        # 创建进程间通信的队列
        result_queue = multiprocessing.Queue()
        
        # 创建并启动执行进程
        process = multiprocessing.Process(
            target=self._execute_code_worker,
            args=(code, func_name, input_data, result_queue)
        )
        
        process.start()
        
        try:
            # 等待结果，设置超时
            result = result_queue.get(timeout=self.timeout)
            process.join(timeout=1)
            
            if process.is_alive():
                process.terminate()
                process.join()
                
            return result
            
        except Exception as e:
            process.terminate()
            process.join()
            return {
                'success': False,
                'error': f'执行超时或出错: {str(e)}',
                'result': None
            }
    
    def _execute_code_worker(self, code: str, func_name: str, input_data: Any, result_queue: multiprocessing.Queue):
        """工作进程中执行代码"""
        try:
            # 捕获输出
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # 执行代码
                namespace = {}
                exec(code, namespace)
                
                # 获取函数并执行
                if func_name not in namespace:
                    raise ValueError(f"函数 '{func_name}' 未在代码中定义")
                
                function = namespace[func_name]
                
                # 处理输入参数 - 如果是tuple，则解包作为多个参数
                if isinstance(input_data, tuple):
                    result = function(*input_data)
                else:
                    result = function(input_data)
                
            # 返回结果
            result_queue.put({
                'success': True,
                'result': result,
                'stdout': stdout_buffer.getvalue(),
                'stderr': stderr_buffer.getvalue(),
                'error': None
            })
            
        except Exception as e:
            result_queue.put({
                'success': False,
                'result': None,
                'error': str(e),
                'traceback': traceback.format_exc()
            })
    
    def execute_with_debug_and_test(self, code: str, func_name: str, input_data: Any, 
                                   hypothesis: str, max_retries: int = 3) -> Dict:
        """执行代码，如果出错则使用大语言模型进行调试和测试"""
        
        original_code = code
        current_code = code
        test_result = {'overall_passed': False}
        for attempt in range(max_retries + 1):
            print(f"\n{'='*50}")
            print(f"代码执行尝试 {attempt + 1}/{max_retries + 1}")
            print(f"{'='*50}")
            
            # 1. 尝试执行代码
            execution_result = self.execute_code_in_process(current_code, func_name, input_data)
            
            if execution_result['success']:
                print("✅ 代码执行成功!")
                
                # 2. 生成和运行单元测试
                
                test_result = self.test_agent.generate_and_run_tests(
                        current_code, func_name, input_data, execution_result['result'], hypothesis
                )
                
                if test_result['test_execution']['all_passed'] == True:
                    return {
                        'success': True,
                        'result': execution_result['result'],
                        'code': current_code,
                        'execution_details': execution_result,
                        'test_result': test_result,
                        'attempts': attempt + 1
                    }
                
            
            if attempt < max_retries:
                # 4. 使用调试代理修复代码
                if execution_result['success'] == False:
                    print(f"❌ 代码执行失败: {execution_result['error']}")
                    debug_result = self.debug_agent.debug_error_code(
                    current_code, func_name, execution_result['error'], 
                    execution_result.get('traceback', ''), hypothesis
                )
                if test_result['test_execution']['all_passed'] == False:
                    print(f"❌ 单元测试失败: {test_result['test_execution']['test_summary']}")
                    debug_result = self.debug_agent.debug_unittest_code(
                    current_code, func_name, test_result['test_generation']['test_strategy'], 
                    test_result['test_execution']['stdout'], hypothesis)
                

                
                if debug_result['success']:
                    current_code = debug_result['fixed_code']
                    print(f"🔧 调试代理已修复代码，准备重新执行...")
                else:
                    print(f"❌ 调试代理无法修复代码: {debug_result['error']}")
                    break
            else:
                print(f"❌ 达到最大重试次数，代码执行失败")
        
        return {
            'success': False,
            'error': execution_result['error'],
            'original_code': original_code,
            'last_attempted_code': current_code,
            'attempts': max_retries + 1
        }
    



class DebugAgent(Agent):
    """代码调试代理"""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        你是一位专业的Python代码调试专家。你的任务是分析代码执行错误，并提供修复后的代码。
        
        你需要：
        1. 分析错误原因和堆栈跟踪
        2. 理解代码的预期功能
        3. 提供修复后的完整代码
        4. 确保修复后的代码符合原始hypothesis的要求
        """
    
    def debug_error_code(self, code: str, func_name: str, error: str, 
                          traceback_info: str, hypothesis: str) -> Dict:
        """调试并修复代码"""
        
        user_prompt = f"""
        以下Python代码在执行时出现了错误，请分析错误原因并提供修复后的代码。

        <original_code>
        {code}
        </original_code>

        <function_name>
        {func_name}
        </function_name>

        <error_message>
        {error}
        </error_message>

        <traceback>
        {traceback_info}
        </traceback>


        <hypothesis>
        {hypothesis}
        </hypothesis>

        请分析错误原因并提供修复后的代码。修复后的代码必须：
        1. 保持原始函数名不变
        2. 保持函数的输入输出接口不变
        3. 符合hypothesis中描述的功能要求
        4. 修复所有语法和逻辑错误
        5. 确保返回格式为[(elem_name, row, col), ...]的列表

        请按以下格式回复：

        <analysis>
        分析错误原因和修复思路
        </analysis>

        <fixed_code>
        修复后的完整代码
        </fixed_code>
        """
        
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            raw_response = self.get_LLM_response(prompt_msgs)
            
            analysis = self.extract_content("analysis", raw_response)
            fixed_code = self.extract_content("fixed_code", raw_response)
            
            if not fixed_code:
                return {
                    'success': False,
                    'error': '无法从响应中提取修复后的代码',
                    'raw_response': raw_response
                }
            
            return {
                'success': True,
                'analysis': analysis,
                'fixed_code': fixed_code,
                'raw_response': raw_response
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'调试过程中出错: {str(e)}',
                'traceback': traceback.format_exc()
            }

    def debug_unittest_code(self, code: str, func_name: str, test_strategy: str, 
                          stdout: str, hypothesis: str) -> Dict:
        """调试并修复代码"""
        
        user_prompt = f"""
        以下Python代码在执行时出现了错误，请分析错误原因并提供修复后的代码。

        <original_code>
        {code}
        </original_code>

        <function_name>
        {func_name}
        </function_name>

        <test_strategy>
        {test_strategy}
        </test_strategy>

        <stdout>
        {stdout}
        </stdout>


        <hypothesis>
        {hypothesis}
        </hypothesis>

        请分析错误原因并提供修复后的代码。修复后的代码必须：
        1. 保持原始函数名不变
        2. 保持函数的输入输出接口不变
        3. 符合hypothesis中描述的功能要求
        4. 修复单元测试失败的原因
        5. 确保返回格式为[(elem_name, row, col), ...]的列表

        请按以下格式回复：

        <analysis>
        分析错误原因和修复思路
        </analysis>

        <fixed_code>
        修复后的完整代码
        </fixed_code>
        """
        
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            raw_response = self.get_LLM_response(prompt_msgs)
            
            analysis = self.extract_content("analysis", raw_response)
            fixed_code = self.extract_content("fixed_code", raw_response)
            
            if not fixed_code:
                return {
                    'success': False,
                    'error': '无法从响应中提取修复后的代码',
                    'raw_response': raw_response
                }
            
            return {
                'success': True,
                'analysis': analysis,
                'fixed_code': fixed_code,
                'raw_response': raw_response
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'调试过程中出错: {str(e)}',
                'traceback': traceback.format_exc()
            }





class TestAgent(Agent):
    """测试代理"""
    
    def __init__(self):
        super().__init__()
        self.system_prompt = """
        你是一位专业的Python单元测试专家。你的任务是为给定的代码生成全面的单元测试，
        验证代码是否正确实现了预期功能。
        """
    
    def generate_and_run_tests(self, code: str, func_name: str, input_data: Any, 
                              execution_result: Any, hypothesis: str) -> Dict:
        """生成并运行单元测试"""
        
        # 生成测试用例
        test_generation_result = self.generate_test_cases(
            code, func_name, input_data, execution_result, hypothesis
        )
        
        if not test_generation_result['success']:
            return test_generation_result
        
        # 运行测试用例
        test_execution_result = self.run_test_cases(
            code, test_generation_result['test_code']
        )
        
        return {
            'success': True,
            'test_generation': test_generation_result,
            'test_execution': test_execution_result,
            'overall_passed': test_execution_result['success'] and test_execution_result['all_passed']
        }
    
    def generate_test_cases(self, code: str, func_name: str, input_data: Any, 
                           execution_result: Any, hypothesis: str) -> Dict:
        """生成测试用例"""
        
        user_prompt = f"""
        请为以下代码生成全面的单元测试用例。

        <code_to_test>
        {code}
        </code_to_test>

        <function_name>
        {func_name}
        </function_name>

        <sample_input>
        {str(input_data)[:1000]}...
        </sample_input>

        <sample_output>
        {str(execution_result)[:1000]}...
        </sample_output>

        <hypothesis>
        {hypothesis}
        </hypothesis>

        请生成测试用例来验证：
        1. 函数的基本功能是否正确
        2. 输出格式是否符合要求[(elem_name, row, col), ...]
        3. 所有row和col都是正整数
        4. 没有位置重叠
        5. 结果是否符合hypothesis中描述的规律

        请按以下格式回复：

        <test_strategy>
        测试策略和测试用例设计思路
        </test_strategy>

        <test_code>
        import unittest
        import pandas as pd
        
        # 在此处包含被测试的代码
        {code}
        
        class TestGeneratedFunction(unittest.TestCase):
            # 生成的测试用例
            pass
        if __name__ == "__main__":
            # 使用TestRunner来控制输出流
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(DemoTest)
            runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
            result = runner.run(suite)
            </test_code>
        """
        
        prompt_msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            raw_response = self.get_LLM_response(prompt_msgs)
            
            test_strategy = self.extract_content("test_strategy", raw_response)
            test_code = self.extract_content("test_code", raw_response)
            
            if not test_code:
                return {
                    'success': False,
                    'error': '无法从响应中提取测试代码',
                    'raw_response': raw_response
                }
            
            return {
                'success': True,
                'test_strategy': test_strategy,
                'test_code': test_code,
                'raw_response': raw_response
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'生成测试用例时出错: {str(e)}',
                'traceback': traceback.format_exc()
            }
    
    def run_test_cases(self, original_code: str, test_code: str) -> Dict:
        """运行测试用例"""
        
        try:
            # 创建临时文件来运行测试
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                test_file = f.name
            
            # 捕获测试输出
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # 在独立的命名空间中运行测试
                namespace = {"__name__": "__main__"}
                exec(test_code, namespace)

            # 清理临时文件
            os.unlink(test_file)
            
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            
            # 简单分析测试结果
            if 'FAILED' in stdout_content or 'ERROR' in stdout_content:
                all_passed = False
                test_summary = "部分测试失败"
            elif 'OK' in stdout_content:
                all_passed = True
                test_summary = "所有测试通过"
            else:
                all_passed = False
                test_summary = "测试结果不明确"
            
            return {
                'success': True,
                'all_passed': all_passed,
                'test_summary': test_summary,
                'stdout': stdout_content,
                'stderr': stderr_content
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'运行测试时出错: {str(e)}',
                'traceback': traceback.format_exc()
            }


def enhanced_code_execution(code: str, func_name: str, input_data: Any, 
                          hypothesis: str, max_retries: int = 3) -> Dict:
    """增强的代码执行函数，包含调试和测试功能"""
    
    executor = CodeExecutorService()
    return executor.execute_with_debug_and_test(
        code, func_name, input_data, hypothesis, max_retries
    )


if __name__ == '__main__':
    # 测试示例
    test_code = """
import pandas as pd

def test_function(df):
    results = []
    for i, (elem, row) in enumerate(df.iterrows()):
        results.append((elem, i+1, 1))
    return results
"""
    
    test_input = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    
    test_hypothesis = "根据元素属性将元素排列在周期表格中"
    
    result = enhanced_code_execution(test_code, 'test_function', test_input, test_hypothesis)
    print(json.dumps(result, indent=2, ensure_ascii=False)) 