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
from  valence_detection import robust_multiscale_gap_voting
from TableState import TableState
import traceback

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
    # 7. group 打乱顺序
    group = df['group'].unique().copy()
    random_groups = random.sample(range(10, 100), len(group))
    group_mapping = dict(zip(df['group'].unique(), [f'group{i}' for i in random_groups]))
    df['group'] = df['group'].map(group_mapping)
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



def hypo_gen_and_eval(table, agents, history, decision, logger, max_retries=2, predict_element=10):
    table.clear_new_elems()
    state = table.get_complete_state()

    # abduction
    print(logger.new_part('Abduction Process'))
    ab_agent = agents['ab_agent']
    if decision == 'C':
        
        # attribute_result = ab_agent.select_main_attribute(state, history)
        # main_attr = attribute_result['attribute']
        # # 按照 Abduction 的判断决定升降序
        # ascending = attribute_result['ascending'].lower() == 'true'
        main_attr = 'Attribute2'
        ascending = True
        table.sort_table(main_attr, ascending=ascending)
        sorted_state = table.get_complete_state()

        # print('Reasoning:')
        # print_and_enter(attribute_result["reasoning"])
        # print('Main attribute:')
        # print_and_enter(main_attr + f' ascending={ascending}')
    else:

        main_attr = history.records[-1]['attribute']
        ascending = history.records[-1]['ascending']
        table.sort_table(main_attr, ascending=ascending)
        sorted_state = table.get_complete_state()

    # 在生成假设之前：基于主属性进行多尺度空隙投票，挑选空位并在表中插入 NewElem0-9

    series = pd.to_numeric(table.elem_df[main_attr], errors='coerce')
    series = series.dropna()
    ordered_names = list(series.index)
    ordered_vals = series.values

    selected_pairs = []  # list[(left_elem, right_elem)]
    if len(ordered_vals) >= 3:
        gap_df = robust_multiscale_gap_voting(
            ordered_vals,
            k=1.0,
            trim=0.1,
            exclude_self=True,
            weighted=True,
            multi_scales=[3, 5, 7],
            min_ctx=3,
            min_votes=2,
        )
        # 附上左右元素名称，gap_index 从 1 开始
        left_names = [ordered_names[i] for i in range(len(ordered_vals) - 1)]
        right_names = [ordered_names[i+1] for i in range(len(ordered_vals) - 1)]
        gap_df['left_elem'] = left_names
        gap_df['right_elem'] = right_names

        # 仅保留 is_outlier 行，转为文本供 agent 选择
        out_df = gap_df[gap_df.get('is_outlier', False) == True].copy()
        cols = ['gap_index', 'left_elem', 'left_value', 'right_value', 'right_elem', 'gap_value', 'score', 'votes', 'best_win']
        cols = [c for c in cols if c in out_df.columns]
        outlier_table_text = out_df[cols].to_string(index=False) if not out_df.empty else ""

        # 交给 Agent 挑选；若返回不足 10 个，后续我们再补齐
        sel_ret = ab_agent.select_gap_candidates(sorted_state, main_attr, ascending, outlier_table_text, n=predict_element)
        selected = sel_ret.get('selected', [])
        # 规范化为 (left_elem, right_elem)
        gap_alignment(predict_element, selected_pairs, gap_df, selected)

    # 将选出的空位插入到 elem_df 中，命名为 NewElem0..9
    table.gap_filling(predict_element, main_attr, ascending, selected_pairs)


    hypothesis_result = ab_agent.generate_hypothesis(table.elem_df.__str__(), main_attr, ascending, history)
    

    hypothesis = hypothesis_result['hypothesis']
    code = hypothesis_result['code']
    func_name = hypothesis_result['func_name']
    
    # 使用增强的代码执行系统

    execution_result = enhanced_code_execution(
        code, func_name, table.elem_df.copy(), hypothesis, max_retries, threshold=0.2
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
    

    # deduction
    print(logger.new_part('Deduction Process'))
    de_agent = agents['de_agent']
    pred_numeric_result = de_agent.predict_numeric(table)
    state = table.get_complete_state()
    pred_result = de_agent.predict_discrete_categorical_attribute(
        state, hypothesis, history, n=predict_element
    )
    deduct_code = pred_result['code']
    func_name = pred_result['func_name']
    
    # 使用增强的代码执行系统
    execution_result = enhanced_code_execution(
        deduct_code, func_name, (table.elem_df.copy()), 
        f"根据元素位置预测元素属性，符合假设: {hypothesis}", max_retries, threshold=0.5
    )
    
    if not execution_result['success']:
        print(f"反向代码执行失败: {execution_result['error']}")
        raise Exception(f"无法生成可执行的反向代码: {execution_result['error']}")
    
    new_elems = execution_result['result']

    # 将预测的属性值填回表中（覆盖已有 NewElem 行的空值/占位）
    try:
        if isinstance(new_elems, list) and len(new_elems) > 0:
            table.append_new_elem(new_elems)
    except Exception as e:
        print(f"填充预测值到表时出错: {e}")

    # 显示执行详情
    print(f"✅ 反向代码执行成功，尝试次数: {execution_result['attempts']}")

    
    success = True

    print('Reasoning:')
    print_and_enter(pred_result["reasoning"])
    print('Deduction code:')
    print_and_enter(deduct_code)

    state = table.get_complete_state()
    print(state)

    # induction
    print(logger.new_part('Induction Process'))
    in_agent = agents['in_agent']

    new_elem_df = table.elem_df[table.elem_df.index.str.contains("NewElem")]
    
    matched_df = table.find_matched_elements(new_elem_df)
    print('matched elements')
    print_and_enter(matched_df)
    n_matched = matched_df.shape[0]
    matched_elem_str = table.state_as_long_str(matched_df)
    mean_rate = matched_df.loc['mean']
    AllPass_rate = mean_rate['AllMatch']
    eval_score = mean_rate.iloc[2:]
    match_rate = mean_rate['match_score']
    print_and_enter(f'Match Rate: {match_rate}')
    print_and_enter(f'eval score:\n{eval_score.__str__()}')
    
    eval_result = in_agent.evaluate_hypothesis(state, hypothesis, matched_elem_str, eval_score.__str__())
    evaluation = eval_result['evaluation']
    decision = eval_result['decision']

    print('Reasoning:')
    print_and_enter(eval_result["reasoning"])
    print('evaluation:')
    print_and_enter(evaluation)
    print('decision:')
    print_and_enter(in_agent.options[decision])

    history.update_records(hypothesis, evaluation, match_rate, eval_score.__str__(), main_attr, ascending, code, deduct_code)

    return table, history, decision, matched_df



def gap_alignment(predict_element, selected_pairs, gap_df, selected):
    for _, le, re in selected:
        if le and re:
            selected_pairs.append((str(le), str(re)))

        # 若不足 10 个，用剩余 gap 依据分数/间隙大小补齐，避免重复
    if len(selected_pairs) < predict_element:
        rest = gap_df.copy()
            # 首选 is_outlier、score 高、gap_value 大
        sort_cols = [c for c in ['is_outlier', 'score', 'gap_value'] if c in rest.columns]
        if sort_cols:
            ascending_flags = [False for _ in sort_cols]
            rest = rest.sort_values(sort_cols, ascending=ascending_flags)
        for _, r in rest.iterrows():
            le = str(r.get('left_elem', ''))
            re = str(r.get('right_elem', ''))
            if not le or not re:
                continue
            pair = (le, re)
            if pair not in selected_pairs:
                selected_pairs.append(pair)
            if len(selected_pairs) >= predict_element:
                break

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
    table.infer_aset(numeric_tolerances={"Attribute2": 4, "Attribute3": 1}, exclude=['Element', 'row', 'col'])
    agents = {
        'ab_agent': AbductionAgent(),
        'de_agent': DeductionAgent(),
        'in_agent': InductionAgent()
    }

    history = History()
    
    # Optionally load previous records from a specific log
    # history.load_records_from_log(join(data_path, 'logs', '2025-07-04-13-17-03'), iteration=1)
    
    logger = Logger(join(data_path, 'logs', '30x10x5attribute'))
    max_iter = 10
    max_retries = 3
    decision = 'C'
    patience = 5
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
        matched_df = matched_df[matched_df['AllMatch'] == True]
        matched_elements = set(matched_df.index) if matched_df is not None else set()
        final_df['KnownAndMatched'] = final_df.index.map(
            lambda elem: 'Known' if not str(elem).startswith('NewElem') 
            else 'New&Matched' if elem in matched_elements 
            else 'New&Unmatched'
        )

        logger.log_table_as_csv(final_df)
        logger.log_table_as_img(final_df)
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
    finally:
        logger.close()
