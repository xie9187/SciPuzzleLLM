import pandas as pd
import random
import os
from os.path import join
import sys
from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from table_agents_v2 import *
from data_utils import *
from code_executor import enhanced_code_execution
from evaluation import EarlyStopper
def generate_table():
    # 初始化数据列表
    element_data = []

    # 门捷列夫当时已知元素
    mendeleev_known_elements = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 37, 38,
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56,
        57, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 92
    }   

    # 先只考虑使用前60个元素，降低难度
    n_elem = 60
    # 使用10个元素测试
    n_sample = 10
    # stratified random sampling
    group_size = n_elem // n_sample  

    test = []
    for i in range(n_sample):
        start = i * group_size
        end = (i+1) * group_size if i < n_sample - 1 else n_elem  
        test.append(random.choice(range(start, end)))
    
    for Z in range(1, n_elem+1):
        e = Element.from_Z(Z)

        # 金属性质判断（只用文档中实际存在的字段）
        if e.is_metal:
            metal_type = "Metal"
        elif e.is_metalloid:
            metal_type = "Metalloid"
        # elif e.is_noble_gas or e.is_halogen or e.is_chalcogen:
        #     metal_type = "Nonmetal"
        else:
            metal_type = "Nonmetal"

        # 常温状态判断（用 boiling_point 和 melting_point）
        if e.boiling_point and e.boiling_point < 298:
            state = "Gas"
        elif e.melting_point and e.melting_point < 298 < (e.boiling_point or 9999):
            state = "Liquid"
        else:
            state = "Solid"
        
        element_data.append({
            "AtomicNumber": e.Z,
            "Element": e.long_name,
            "Symbol": e.symbol,
            "group": e.group, 
            "AtomicMass": float(e.atomic_mass) if e.atomic_mass else None,
            "OxidationStates": ", ".join(map(str, e.common_oxidation_states)) if e.common_oxidation_states else 0,
            "StateAtRoomTemp": state,
            "MetalType": metal_type,
            "Test": e.Z in test
        })

    df = pd.DataFrame(element_data)

    most_common_oxidation_states = {
        # 第一周期
        "H": +1, "He": 0,
        # 第二周期
        "Li": +1, "Be": +2, "B": +3, "C": +4, "N": -3, "O": -2, "F": -1, "Ne": 0,
        # 第三周期
        "Na": +1, "Mg": +2, "Al": +3, "Si": +4, "P": +5, "S": -2, "Cl": -1, "Ar": 0,
        # 第四周期
        "K": +1, "Ca": +2, "Sc": +3, "Ti": +4, "V": +5, "Cr": +3, "Mn": +2,
        "Fe": +3, "Co": +2, "Ni": +2, "Cu": +2, "Zn": +2, "Ga": +3, "Ge": +4,
        "As": +5, "Se": -2, "Br": -1, "Kr": 0,
        # 第五周期
        "Rb": +1, "Sr": +2, "Y": +3, "Zr": +4, "Nb": +5, "Mo": +6, "Tc": +7,
        "Ru": +3, "Rh": +3, "Pd": +2, "Ag": +1, "Cd": +2, "In": +3, "Sn": +4,
        "Sb": +3, "Te": -2, "I": -1, "Xe": 0,
        # 第六周期
        "Cs": +1, "Ba": +2, "La": +3, "Ce": +4, "Pr": +3, "Nd": +3, "Pm": +3,
        "Sm": +3, "Eu": +3, "Gd": +3, "Tb": +3, "Dy": +3, "Ho": +3, "Er": +3,
        "Tm": +3, "Yb": +3, "Lu": +3,
        "Hf": +4, "Ta": +5, "W": +6, "Re": +7, "Os": +4, "Ir": +3, "Pt": +2,
        "Au": +3, "Hg": +2, "Tl": +3, "Pb": +2, "Bi": +3, "Po": +2, "At": -1,
        "Rn": 0,
        # 第七周期
        "Fr": +1, "Ra": +2, "Ac": +3, "Th": +4, "Pa": +5, "U": +6, "Np": +5,
        "Pu": +4, "Am": +3, "Cm": +3, "Bk": +3, "Cf": +3, "Es": +3, "Fm": +3,
        "Md": +3, "No": +2, "Lr": +3,
        "Rf": +4, "Db": +5, "Sg": +6, "Bh": +7, "Hs": +4, "Mt": +1,
        "Ds": +0, "Rg": +1, "Cn": +2, "Nh": +1, "Fl": +2, "Mc": +1,
        "Lv": +2, "Ts": -1, "Og": 0
    }
    df["OxidationStates"] = df["Symbol"].map(most_common_oxidation_states)

    return df

def mask_table(df: pd.DataFrame, known_to_mendeleev: bool = True) -> pd.DataFrame:
    # 1. 剔除 AtomicNumber 和 Symbol 列
    df = df.drop(columns=["AtomicNumber", "Symbol"])

    # 2. 生成 3 位随机且不重复的数字，用于替换元素名称
    num_elements = len(df)
    random_numbers = random.sample(range(100, 1000), num_elements)  # 生成 3 位不重复随机数
    df["Element"] = [f"Elem{num}" for num in random_numbers]

    # # 3. 将 AtomicMass 归一化到 [1, 100]
    # scaler = MinMaxScaler(feature_range=(1, 100))
    # df["AtomicMass"] = scaler.fit_transform(df[["AtomicMass"]])

    # 5. StateAtRoomTemp 映射为 State1, State2...
    state_mapping = {state: f"State{i+1}" for i, state in enumerate(df["StateAtRoomTemp"].unique())}
    df["StateAtRoomTemp"] = df["StateAtRoomTemp"].map(state_mapping)

    # 6. MetalType 映射为 Type1, Type2...
    type_mapping = {typ: f"Type{i+1}" for i, typ in enumerate(df["MetalType"].unique())}
    df["MetalType"] = df["MetalType"].map(type_mapping)
    # 7. group
    
    # 8. 转换列名为 Attribute1, Attribute2,...
    columns = list(df.columns)
    
    # 第一列命名为 Element
    new_column_names = {columns[0]: "Element"}
    
    # 其余列命名为 Attribute1, Attribute2, ...
    for i, col in enumerate(columns[1:-1], start=1):
        new_column_names[col] = f"Attribute{i}"
    
    df = df.rename(columns=new_column_names)

    # 9. 按照 Element 排序（按 ElemXXX 的数字部分升序排列）
    df["TempSortKey"] = df["Element"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("TempSortKey").drop(columns="TempSortKey")

    # 10. 根据 Known 分割数据
    train_df = df[df["Test"] == False].drop(columns=["Test"])
    test_df = df[df["Test"] == True].drop(columns=["Test"])

    return train_df, test_df

class TableState(object):
    def __init__(self, elem_df, test_df):
        super(TableState, self).__init__()
        self.elem_df = elem_df
        self.elem_attrs = list(elem_df.columns)
        self.elem_df['row'] = None
        self.elem_df['col'] = None
        self.test_df = test_df

    def fill_elem_posi(self, actions):
        for elem, row, col in actions:
            self.elem_df.loc[elem, ['col', 'row']] = [row, col]

    def append_new_elem(self, elems):
        for elem_attr in elems:
            # elem_attr = [str(val) for val in elem_attr]
            self.elem_df.loc[elem_attr[0]] = elem_attr[1:] 
        
    def sort_table(self, sort_col, ascending=True):
        self.raw_df = self.elem_df.copy()
        self.elem_df = self.elem_df.sort_values(sort_col, ascending=ascending)

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



    def find_matched_elements(self, df1, df2, tolerance=1.0):
        """
        通过行循环匹配两个 DataFrame 的所有共有属性列（除了 row 和 col）
        
        参数:
            df1: 待匹配的 DataFrame（可能包含 row 和 col）
            df2: 目标 DataFrame
            tolerance: Attribute1 允许的误差范围（默认 1.0）
        
        返回:
            匹配成功的 df1 的子集
        """
        # 1. 找出两个 DataFrame 共有的列（排除 df1 的 row 和 col）
        common_cols = [col for col in df1.columns 
                       if col in df2.columns and col not in ["row", "col"]]
        
        if not common_cols:
            return pd.DataFrame()  # 如果没有共有列，返回空 DataFrame
        
        matched_indices = []
        
        # 2. 遍历 df1 的每一行
        for idx1, row1 in df1.iterrows():
            # 3. 对每个 df1 的行，遍历 df2 的所有行寻找匹配
            for idx2, row2 in df2.iterrows():
                match = True
                # 4. 检查所有共有列是否匹配
                for col in common_cols:
                    if pd.isna(row1[col]) or pd.isna(row2[col]):
                        match = False
                        break
                    if col == "Attribute1":
                        # Attribute1 允许误差
                        if abs(row1[col] - row2[col]) >= tolerance:
                            match = False
                            break
                    else:
                        # 其他属性必须完全相等
                        if row1[col] != row2[col]:
                            match = False
                            break
                
                # 5. 如果找到匹配，记录 df1 的索引并跳出内层循环
                if match:
                    matched_indices.append(idx1)
                    break
        
        # 6. 返回匹配成功的 df1 行
        return df1.loc[matched_indices]

    def clear_new_elems(self):
        self.elem_df = self.elem_df[~self.elem_df.index.str.startswith('NewElem')]

    def get_complete_state(self):
        state_str = 'All elements:\n'
        state_str += self.state_as_long_str() + '\n\n'
        for attr in self.elem_attrs:
            state_str += f'{attr} of current elements in virtual periodic table\n'
            state_str += self.att_table(attr) + '\n\n'
        return state_str

def hypo_gen_and_eval(table, agents, history, decision, logger, max_retries=2):
    table.clear_new_elems()
    state = table.get_complete_state()

    # abduction
    print(logger.new_part('Abduction Process'))
    ab_agent = agents['ab_agent']
    if decision == 'C':
        
        attribute_result = ab_agent.select_main_attribute(state, history)
        main_attr = attribute_result['attribute']
        ascending = attribute_result['ascending'].lower() == 'true'
        table.sort_table(main_attr, ascending=ascending)
        sorted_state = table.get_complete_state()

        print('Reasoning:')
        print_and_enter(attribute_result["reasoning"])
        print('Main attribute:')
        print_and_enter(main_attr + f' ascending={ascending}')
    else:

        main_attr = history.records[-1]['attribute']
        ascending = history.records[-1]['ascending']
        table.sort_table(main_attr, ascending=ascending)
        sorted_state = table.get_complete_state()



    hypothesis_result = ab_agent.generate_hypothesis(sorted_state, main_attr, history)


    hypothesis = hypothesis_result['hypothesis']
    code = hypothesis_result['code']
    func_name = hypothesis_result['func_name']
    
    # 使用增强的代码执行系统

    execution_result = enhanced_code_execution(
        code, func_name, table.elem_df.copy(), hypothesis, max_retries, threshold=0.5
    )
    
    if not execution_result['success']:
        print_and_enter(f"代码执行失败: {execution_result['error']}")
        raise Exception(f"无法生成可执行的代码: {execution_result['error']}")
    
    actions = execution_result['result']
    
    # 显示执行详情
    print_and_enter(f"✅ 代码执行成功，尝试次数: {execution_result['attempts']}")

    
    success = True

    print('Reasoning:')
    print_and_enter(hypothesis_result["reasoning"])
    print('Hypothesis:')
    print_and_enter(hypothesis)
    print('Code:')
    print_and_enter(code)

    table.fill_elem_posi(actions)
    state = table.get_complete_state()

    # deduction
    print(logger.new_part('Deduction Process'))
    de_agent = agents['de_agent']

    pred_result = de_agent.predict_elements(state, hypothesis, code, history, n=10)
    new_elems_posi = pred_result['new_elems_posi']
    inverse_code = pred_result['inverse_code']
    func_name = pred_result['func_name']
    
    # 使用增强的代码执行系统
    execution_result = enhanced_code_execution(
        inverse_code, func_name, (new_elems_posi, table.elem_df.copy()), 
        f"根据元素位置预测元素属性，符合假设: {hypothesis}", max_retries, threshold=0.5
    )
    
    if not execution_result['success']:
        print(f"反向代码执行失败: {execution_result['error']}")
        raise Exception(f"无法生成可执行的反向代码: {execution_result['error']}")
    
    new_elems = execution_result['result']
    
    # 显示执行详情
    print(f"✅ 反向代码执行成功，尝试次数: {execution_result['attempts']}")

    
    success = True

    print('Reasoning:')
    print_and_enter(pred_result["reasoning"])
    print('New elem position:')
    print(new_elems_posi)
    print('Inverse code:')
    print_and_enter(inverse_code)

    table.append_new_elem(new_elems)
    state = table.get_complete_state()
    print(state)

    # induction
    print(logger.new_part('Induction Process'))
    in_agent = agents['in_agent']

    new_elem_df = table.elem_df[table.elem_df.index.str.contains("NewElem")]
    matched_df = table.find_matched_elements(new_elem_df, table.test_df)
    print('matched elements')
    print_and_enter(matched_df)
    n_matched = matched_df.shape[0]
    matched_elem_str = table.state_as_long_str(matched_df)
    match_rate = n_matched / new_elem_df.shape[0]
    print_and_enter(f'Match Rate: {match_rate}')
    eval_result = in_agent.evaluate_hypothesis(state, hypothesis, matched_elem_str, match_rate)
    evaluation = eval_result['evaluation']
    decision = eval_result['decision']

    print('Reasoning:')
    print_and_enter(eval_result["reasoning"])
    print('evaluation:')
    print_and_enter(evaluation)
    print('decision:')
    print_and_enter(in_agent.options[decision])

    history.update_records(hypothesis, evaluation, match_rate, main_attr, ascending, code, inverse_code)

    return table, history, decision, matched_df

if __name__ == '__main__':
    if sys.platform == 'win32':
        data_path = r'D:\data\SciPuzzleLLM'
    else:
        data_path = r'/data/SciPuzzleLLM'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # make table data and object
    table_file = join(data_path, 'PeriodicTable.csv')
    if not os.path.exists(table_file):
        df = generate_table()
        df.to_csv(table_file, index=False)
    else:
        df = pd.read_csv(table_file, index_col=None)

    train_df, test_df = mask_table(df)
    train_df.to_csv(join(data_path, 'train_df.csv'), index=False)
    train_df = train_df.set_index('Element', drop=True)
    test_df.to_csv(join(data_path, 'test_df.csv'), index=False)
    test_df = test_df.set_index('Element', drop=True)

    train_df = pd.read_csv(join(data_path, 'train_df.csv'), index_col='Element')
    test_df = pd.read_csv(join(data_path, 'test_df.csv'), index_col='Element')

    table = TableState(train_df, test_df)
    
    agents = {
        'ab_agent': AbductionAgent(),
        'de_agent': DeductionAgent(),
        'in_agent': InductionAgent()
    }

    history = History()
    
    # Optionally load previous records from a specific log
    # history.load_records_from_log(join(data_path, 'logs', '2025-07-04-13-17-03'), iteration=1)
    
    logger = Logger(join(data_path, 'logs'))
    max_iter = 5
    max_retries = 3
    decision = 'C'
    patience = 3
    min_delta = 0
    min_iters = 3
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, min_iters=min_iters)
    best_score = -float("inf")
    try:
        for i in range(max_iter):

            print(logger.new_part(f'Iteration {i+1}'))

            table, history, decision, matched_df = hypo_gen_and_eval(
                table, agents, history, decision, logger, max_retries)
            # early stop
            # TODO: Add code complexity penalty
            match_score = history.records[-1]['match_rate']
            should_continue = stopper.update(i, match_score, (table, history, decision, matched_df))
            best_score = max(best_score, match_score)
            if not should_continue:
                print("Stop as the iteration stops improving (early stopping).")
                break    
            if decision == 'P':
                print('Stop as the hypothesis is accepted!')
                break
            if i == max_iter - 1:
                print(f'Stop as the max iteration {i+1} is reached!')

        print('\nhistory:')
        print(history.show_records())
        if stopper.best_payload is not None:
            table, history, decision, matched_df = stopper.best_payload
        final_df = table.elem_df.copy()
        # 新增 KnownAndMatched 列
        matched_elements = set(matched_df.index) if matched_df is not None else set()
        final_df['KnownAndMatched'] = final_df.index.map(
            lambda elem: 'Known' if not str(elem).startswith('NewElem') 
            else 'New&Matched' if elem in matched_elements 
            else 'New&Unmatched'
        )

        logger.log_table_as_csv(final_df)
        logger.log_table_as_img(final_df)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        logger.close()
