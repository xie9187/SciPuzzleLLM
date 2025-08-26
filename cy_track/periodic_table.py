import pandas as pd
import random
import os
from os.path import join

from pymatgen.core.periodic_table import Element
from sklearn.preprocessing import MinMaxScaler

from table_agents_v2 import *
from data_utils import *
from code_executor import enhanced_code_execution, only_code_execution


def get_element_group(symbol):
    """
    获取元素的化学族信息
    基于PeriodicTable_mjlf.csv的数据
    """
    group_mapping = {
        # 第1族 - 碱金属
        "Li": "Alkali metal", "Na": "Alkali metal", "K": "Alkali metal", "Rb": "Alkali metal", "Cs": "Alkali metal",
        # 第2族 - 碱土金属
        "Be": "Alkaline earth", "Mg": "Alkaline earth", "Ca": "Alkaline earth", "Sr": "Alkaline earth", "Ba": "Alkaline earth",
        # 第13族 - 硼族
        "B": "Boron group", "Al": "Boron group", "Ga": "Boron group", "In": "Boron group", "Tl": "Boron group",
        # 第14族 - 碳族
        "C": "Carbon", "Si": "Carbon", "Ge": "Carbon", "Sn": "Carbon", "Pb": "Carbon",
        # 第15族 - 氮族
        "N": "Nitrogen", "P": "Nitrogen", "As": "Nitrogen", "Sb": "Nitrogen", "Bi": "Nitrogen",
        # 第16族 - 氧族（硫族）
        "O": "Chalcogen", "S": "Chalcogen", "Se": "Chalcogen", "Te": "Chalcogen", "Po": "Chalcogen",
        # 第17族 - 卤素
        "F": "Halogen", "Cl": "Halogen", "Br": "Halogen", "I": "Halogen", "At": "Halogen",
        # 第18族 - 惰性气体
        "He": "Noble gas", "Ne": "Noble gas", "Ar": "Noble gas", "Kr": "Noble gas", "Xe": "Noble gas", "Rn": "Noble gas",
        # 过渡金属族
        "Sc": "Transition metal", "Ti": "Transition metal", "V": "Transition metal", "Cr": "Transition metal", "Mn": "Transition metal",
        "Fe": "Iron", "Co": "Iron", "Ni": "Iron", "Cu": "Copper", "Zn": "Zinc",
        "Y": "Transition metal", "Zr": "Transition metal", "Nb": "Transition metal", "Mo": "Transition metal", "Tc": "Transition metal",
        "Ru": "Platinum", "Rh": "Platinum", "Pd": "Platinum", "Ag": "Copper", "Cd": "Zinc",
        "La": "Lanthanide", "Ce": "Lanthanide", "Pr": "Lanthanide", "Nd": "Lanthanide", "Pm": "Lanthanide",
        "Sm": "Lanthanide", "Eu": "Lanthanide", "Gd": "Lanthanide", "Tb": "Lanthanide", "Dy": "Lanthanide",
        "Ho": "Lanthanide", "Er": "Lanthanide", "Tm": "Lanthanide", "Yb": "Lanthanide", "Lu": "Lanthanide",
        "Hf": "Transition metal", "Ta": "Transition metal", "W": "Transition metal", "Re": "Transition metal", "Os": "Transition metal",
        "Ir": "Transition metal", "Pt": "Platinum", "Au": "Transition metal", "Hg": "Zinc", "Tl": "Transition metal",
        "Pb": "Carbon", "Bi": "Nitrogen", "Po": "Chalcogen", "At": "Halogen", "Rn": "Noble gas",
        # 特殊元素
        "H": "Hydrogen"
    }
    return group_mapping.get(symbol, "unknow")

def get_elemental_affinity(symbol):
    """
    获取元素的亲和性信息
    基于PeriodicTable_mjlf.csv的数据
    """
    affinity_mapping = {
        # 亲石元素 (Lithophile)
        "Li": "Lithophile", "Na": "Lithophile", "K": "Lithophile", "Rb": "Lithophile", "Cs": "Lithophile",
        "Be": "Lithophile", "Mg": "Lithophile", "Ca": "Lithophile", "Sr": "Lithophile", "Ba": "Lithophile",
        "B": "Lithophile", "Al": "Lithophile", "Ga": "Lithophile", "In": "Lithophile", "Tl": "Lithophile",
        "Si": "Lithophile", "Ge": "Lithophile", "Sn": "Lithophile", "Pb": "Lithophile",
        "P": "Lithophile", "As": "Lithophile", "Sb": "Nitrogen", "Bi": "Nitrogen",
        "S": "Chalcophile", "Se": "Chalcophile", "Te": "Chalcophile", "Po": "Chalcogen",
        "F": "Lithophile", "Cl": "Lithophile", "Br": "Lithophile", "I": "Lithophile", "At": "Halogen",
        "Sc": "Lithophile", "Ti": "Lithophile", "V": "Lithophile", "Cr": "Lithophile", "Mn": "Siderophile",
        "Fe": "Siderophile", "Co": "Siderophile", "Ni": "Siderophile", "Cu": "Chalcophile", "Zn": "Chalcophile",
        "Y": "Lithophile", "Zr": "Lithophile", "Nb": "Lithophile", "Mo": "Siderophile", "Tc": "unknow",
        "Ru": "Siderophile", "Rh": "Siderophile", "Pd": "Siderophile", "Ag": "Chalcophile", "Cd": "Chalcophile",
        "La": "Lithophile", "Ce": "Lithophile", "Pr": "Lithophile", "Nd": "Lithophile", "Pm": "Lanthanide",
        "Sm": "Lanthanide", "Eu": "Lanthanide", "Gd": "Lanthanide", "Tb": "Lanthanide", "Dy": "Lanthanide",
        "Ho": "Lanthanide", "Er": "Lanthanide", "Tm": "Lanthanide", "Yb": "Lanthanide", "Lu": "Lanthanide",
        "Hf": "Lithophile", "Ta": "Lithophile", "W": "Lithophile", "Re": "Lithophile", "Os": "Siderophile",
        "Ir": "Siderophile", "Pt": "Siderophile", "Au": "Chalcophile", "Hg": "Chalcophile", "Tl": "Chalcophile",
        # 亲气元素 (Atmophile)
        "H": "Atmophile", "He": "Atmophile", "Ne": "Atmophile", "Ar": "Atmophile", "Kr": "Atmophile", "Xe": "Atmophile",
        "N": "Atmophile", "O": "Lithophile", "C": "Atmophile"
    }
    return affinity_mapping.get(symbol, "unknow")

def get_properties_of_oxides(symbol):
    """
    获取元素的氧化物性质信息
    基于PeriodicTable_mjlf.csv的数据
    """
    oxide_mapping = {
        # 强碱性氧化物
        "Li": "Strongly basic oxide", "Na": "Strongly basic oxide", "K": "Strongly basic oxide", 
        "Rb": "Strongly basic oxide", "Cs": "Strongly basic oxide",
        "Be": "Amphoteric oxide", "Mg": "Basic oxide", "Ca": "Strongly basic oxide", 
        "Sr": "Strongly basic oxide", "Ba": "Strongly basic oxide",
        # 两性氧化物
        "Al": "Amphoteric oxide", "Ga": "Amphoteric oxide", "In": "Amphoteric oxide", "Tl": "Amphoteric oxide",
        "Si": "Weakly acidic oxide", "Ge": "Amphoteric oxide", "Sn": "Amphoteric oxide", "Pb": "Amphoteric oxide",
        "Ti": "Amphoteric oxide", "V": "Acidic oxide", "Cr": "Amphoteric oxide", "Mn": "Strongly acidic oxide",
        "Fe": "Weakly basic oxide", "Co": "Basic oxide", "Ni": "Basic oxide", "Cu": "Weakly basic oxide", "Zn": "Amphoteric oxide",
        "Zr": "Amphoteric oxide", "Nb": "Weakly acidic oxide", "Mo": "Acidic oxide", "Tc": "Strongly acidic oxide",
        "Ru": "Weakly basic oxide", "Rh": "Weakly basic oxide", "Pd": "Basic oxide", "Ag": "Weakly basic oxide", "Cd": "Basic oxide",
        "Hf": "Amphoteric oxide", "Ta": "Weakly acidic oxide", "W": "Weakly acidic oxide", "Re": "Weakly acidic oxide", "Os": "Weakly basic oxide",
        "Ir": "Weakly basic oxide", "Pt": "Weakly basic oxide", "Au": "Weakly basic oxide", "Hg": "Weakly basic oxide",
        # 酸性氧化物
        "B": "Weakly acidic oxide", "C": "Weakly acidic oxide", "N": "Acidic oxide", "P": "Acidic oxide", "As": "Acidic oxide", "Sb": "Acidic oxide", "Bi": "Acidic oxide",
        "S": "Strongly acidic oxide", "Se": "Weakly acidic oxide", "Te": "Acidic oxide", "Po": "Acidic oxide",
        "F": "Strongly oxidizing oxide", "Cl": "Strongly acidic oxide", "Br": "Strongly acidic oxide", "I": "Strongly acidic oxide", "At": "Strongly acidic oxide",
        "V": "Acidic oxide", "Cr": "Amphoteric oxide", "Mn": "Strongly acidic oxide", "Mo": "Acidic oxide", "Tc": "Strongly acidic oxide",
        # 中性氧化物
        "H": "Neutral oxide", "He": "unknow", "Ne": "unknow", "Ar": "unknow", "Kr": "unknow", "Xe": "unknow", "Rn": "unknow",
        # 镧系元素
        "La": "Strongly basic oxide", "Ce": "Weakly acidic oxide", "Pr": "Basic oxide", "Nd": "Basic oxide", "Pm": "Basic oxide",
        "Sm": "Basic oxide", "Eu": "Basic oxide", "Gd": "Basic oxide", "Tb": "Basic oxide", "Dy": "Basic oxide",
        "Ho": "Basic oxide", "Er": "Basic oxide", "Tm": "Basic oxide", "Yb": "Basic oxide", "Lu": "Basic oxide"
    }
    return oxide_mapping.get(symbol, "unknow")

def parse_electronic_structure(electronic_structure):
    """
    解析电子结构字符串，提取价层信息
    例如: "[Ne].3s2.2p3" -> 价层编号=3, 价电子数=5
    """
    if not electronic_structure:
        return None, None, None
    
    # 移除稀有气体符号，如 [Ne], [Ar] 等
    parts = electronic_structure.split('.')
    
    # 找到最高主量子数（价层编号）
    max_n = 0
    valence_electrons = 0
    
    for part in parts:
        if part.startswith('[') and part.endswith(']'):
            continue  # 跳过稀有气体符号
        
        # 解析如 "3s2", "2p3" 等
        if len(part) >= 2:
            try:
                n = int(part[0])  # 主量子数
                max_n = max(max_n, n)
                
                # 计算电子数
                if len(part) > 2:
                    electron_count = int(part[2:])
                    valence_electrons += electron_count
                else:
                    valence_electrons += 1
            except (ValueError, IndexError):
                continue
    
    return max_n, valence_electrons

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
    
    # 随机选择10个测试集元素
    test = random.sample(range(n_elem), 10)
    
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

        # 获取电子结构信息
        # Block: 电子亚层类型 (s, p, d, f)
        block = e.block
        
        # ValenceShellN: 价层编号（最外层电子层的主量子数）
        # ValenceElectrons: 价电子数（最外层电子数）
        valence_shell_n, valence_electrons = parse_electronic_structure(e.electronic_structure)
        
        # 如果解析失败，使用备选方法
        if valence_shell_n is None:
            valence_shell_n = e.row  # 使用周期数作为备选
            valence_electrons = sum(e.valence) if e.valence else 0  # 使用化合价作为备选

        element_data.append({
            "AtomicNumber": e.Z,
            "Element": e.long_name,
            "Symbol": e.symbol,
            "AtomicMass": float(e.atomic_mass) if e.atomic_mass else None,
            "OxidationStates": ", ".join(map(str, e.common_oxidation_states)) if e.common_oxidation_states else 0,
            "StateAtRoomTemp": state,
            "MetalType": metal_type,
            "Block": block,
            "ValenceShellN": valence_shell_n,
            "ValenceElectrons": valence_electrons,
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

def generate_test_column(df: pd.DataFrame, num_test_elements: int = 10) -> pd.DataFrame:
    """
    为DataFrame添加Test列，随机选择指定数量的元素作为测试集
    
    参数:
        df: 输入的DataFrame
        num_test_elements: 要选择为测试集的元素数量，默认10个
    
    返回:
        添加了Test列的DataFrame
    """
    # 确保测试元素数量不超过总元素数量
    num_test_elements = min(num_test_elements, len(df))
    
    # 随机选择测试元素的索引
    test_indices = random.sample(range(len(df)), num_test_elements)
    
    # 初始化Test列为False
    df = df.copy()
    df['Test'] = False
    
    # 将选中的元素标记为True
    df.loc[test_indices, 'Test'] = True
      
    return df

def mask_table(df: pd.DataFrame, known_to_mendeleev: bool = True):
    # 0. 生成Test列，随机选择10个元素作为测试集
    df = generate_test_column(df, num_test_elements=10)
    
    # 1. 剔除 AtomicNumber 和 Symbol 列
    df = df.drop(columns=["AtomicNumber", "Symbol"])

    # 2. 生成 3 位随机且不重复的数字，用于替换元素名称
    num_elements = len(df)
    random_numbers = random.sample(range(100, 1000), num_elements)  # 生成 3 位不重复随机数
    df["Element"] = [f"Elem{num}" for num in random_numbers]

    # # 3. 将 AtomicMass 归一化到 [1, 100]
    # scaler = MinMaxScaler(feature_range=(1, 100))
    # df["AtomicMass"] = scaler.fit_transform(df[["AtomicMass"]])

    # 5. StateAtRoomTemp 映射为 State1, State2...（随机顺序，检查列是否存在）
    if "StateAtRoomTemp" in df.columns:
        unique_states = [state for state in df["StateAtRoomTemp"].unique() if state != "unknow"]
        # 随机打乱
        random.shuffle(unique_states)
        state_mapping = {state: f"State{i+1}" for i, state in enumerate(unique_states)}
        # 保持unknow不变
        if "unknow" in df["StateAtRoomTemp"].unique():
            state_mapping["unknow"] = "unknow"
        df["StateAtRoomTemp"] = df["StateAtRoomTemp"].map(state_mapping)
        

    # 6. MetalType 映射为 Type1, Type2...（随机顺序，检查列是否存在）
    if "MetalType" in df.columns:
        unique_types = [typ for typ in df["MetalType"].unique() if typ != "unknow"]
        random.shuffle(unique_types)
        type_mapping = {typ: f"Type{i+1}" for i, typ in enumerate(unique_types)}
        # 保持unknow不变
        if "unknow" in df["MetalType"].unique():
            type_mapping["unknow"] = "unknow"
        df["MetalType"] = df["MetalType"].map(type_mapping)
        
    # 7. TheGroup 映射为 Group1, Group2...（随机顺序，保持unknow不变，检查列是否存在）
    if "TheGroup" in df.columns:
        group_values = [group for group in df["TheGroup"].unique() if group != "unknow"]
        random.shuffle(group_values)
        group_mapping = {group: f"Group{i+1}" for i, group in enumerate(group_values)}
        # 保持unknow不变
        if "unknow" in df["TheGroup"].unique():
            group_mapping["unknow"] = "unknow"
        df["TheGroup"] = df["TheGroup"].map(group_mapping)
        

    # 8. ElementalAffinity 映射为 Affinity1, Affinity2...（随机顺序，保持unknow不变，检查列是否存在）
    if "ElementalAffinity" in df.columns:
        affinity_values = [affinity for affinity in df["ElementalAffinity"].unique() if affinity != "unknow"]
        random.shuffle(affinity_values)
        affinity_mapping = {affinity: f"Affinity{i+1}" for i, affinity in enumerate(affinity_values)}
        # 保持unknow不变
        if "unknow" in df["ElementalAffinity"].unique():
            affinity_mapping["unknow"] = "unknow"
        df["ElementalAffinity"] = df["ElementalAffinity"].map(affinity_mapping)
        

    # 9. PropertiesOfOxides 映射为 Oxide1, Oxide2...（随机顺序，保持unknow不变，检查列是否存在）
    """ if "PropertiesOfOxides" in df.columns:
        oxide_values = [oxide for oxide in df["PropertiesOfOxides"].unique() if oxide != "unknow"]
        random.shuffle(oxide_values)
        oxide_mapping = {oxide: f"Oxide{i+1}" for i, oxide in enumerate(oxide_values)}
        # 保持unknow不变
        if "unknow" in df["PropertiesOfOxides"].unique():
            oxide_mapping["unknow"] = "unknow"
        df["PropertiesOfOxides"] = df["PropertiesOfOxides"].map(oxide_mapping)
         """

    # 10. 转换列名为 Attribute1, Attribute2,...
    columns = list(df.columns)
    
    # 第一列命名为 Element
    new_column_names = {columns[0]: "Element"}
    
    # 其余列命名为 Attribute1, Attribute2, ...
    # 在重命名前，基于当前列顺序构建 attr_name_map（AttributeX -> 原始列名）
    attr_name_map = {}
    for i, col in enumerate(columns[1:-1], start=1):
        new_name = f"Attribute{i}"
        new_column_names[col] = new_name
        attr_name_map[new_name] = col
    
    df = df.rename(columns=new_column_names)

    # 8. 按照 Element 排序（按 ElemXXX 的数字部分升序排列）
    df["TempSortKey"] = df["Element"].str.extract(r"(\d+)").astype(int)
    df = df.sort_values("TempSortKey").drop(columns="TempSortKey")

    # 9. 根据 Known 分割数据
    train_df = df[df["Test"] == False].drop(columns=["Test"])
    test_df = df[df["Test"] == True].drop(columns=["Test"])

    # 返回训练集、测试集，以及属性映射表（用于匹配加权）
    return train_df, test_df, attr_name_map

class TableState(object):
    def __init__(self, elem_df, test_df):
        super(TableState, self).__init__()
        self.elem_df = elem_df
        self.elem_attrs = list(elem_df.columns)
        #self.elem_df['row'] = None
        #self.elem_df['col'] = None
        if 'row' not in self.elem_df.columns:
            self.elem_df['row'] = None
        if 'col' not in self.elem_df.columns:
            self.elem_df['col'] = None
        self.test_df = test_df

    def fill_elem_posi(self, actions):
        for elem, row, col in actions:
            self.elem_df.loc[elem, ['row', 'col']] = [row, col]

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
        table = [['' for _ in range(max_col)] for _ in range(max_row)]
        
        # 填充已填入的元素
        filled = self.elem_df.dropna(subset=['col', 'row'])
        for _, row in filled.iterrows():
            r = int(row['row']) - 1  # 转换为0-based索引
            c = int(row['col']) - 1
            if 0 <= r < max_row and 0 <= c < max_col:
                attr_value = row[attr_name]
                if pd.api.types.is_float(attr_value) and not pd.isna(attr_value):
                    table[r][c] = f"{float(attr_value):.1f}"
                else:
                    table[r][c] = str(attr_value)
        
        # 计算每列最大宽度
        col_widths = [max(len(str(table[r][c])) for r in range(max_row)) for c in range(max_col)]
        
        # 生成表格字符串
        table_lines = []
        for r in range(max_row):
            line = '|' + ', '.join(
                f" {table[r][c]:^{col_widths[c]}} " for c in range(max_col)
            ) + '|'
            table_lines.append(line)
        
        return '\n'.join(table_lines)

    def find_matched_elements(self, df1, df2, tolerance=1.0, attr_name_map=None):
        """
        在匿名化列名场景下，匹配 df1 与 df2 的元素行，返回带有匹配分数的子集。
        - 若提供 attr_name_map（列名 -> 原始属性名），按方案B对不同属性加权；
          否则对所有参与属性等权重处理。
        - 不同模式下属性数量可能不同；本函数会对参与评分的属性权重归一化，使满分恒为1.0。

        参数:
            df1: 待匹配的 DataFrame（可能包含 row 和 col）
            df2: 目标 DataFrame
            tolerance: 数值属性允许的误差范围（默认 1.0）
            attr_name_map: 可选，dict 映射 {列名: 原始属性名}，用于按方案B加权。

        返回:
            DataFrame: 包含 df1 中每行的最佳匹配分数（match_score）与对应目标索引（matched_with）。
        """
        # 共有列（排除 row/col）
        common_cols = [col for col in df1.columns if col in df2.columns and col not in ["row", "col"]]
        if not common_cols:
            return pd.DataFrame()

        # 方案B权重（原始属性名 -> 权重）
        scheme_b_weights = {
            'OxidationStates': 0.13,
            'PropertiesOfOxides': 0.1,
            'ElementalAffinity': 0.1,
            'MetalType': 0.1,
            'AtomicMass': 0.30,
            'StateAtRoomTemp': 0.07,
            'TheGroup': 0.2,
        }

        # 判断是否可以使用方案B加权：attr_name_map 覆盖所有需要参与的列
        use_scheme_b = isinstance(attr_name_map, dict) and all(
            (col in attr_name_map and attr_name_map[col] in scheme_b_weights)
            for col in common_cols
        )

        result_data = []

        # 遍历 df1 的每一行
        for idx1, row1 in df1.iterrows():
            best_score = 0.0
            best_match_idx = None

            for idx2, row2 in df2.iterrows():
                # 计算可用列（两侧均非缺失/非 'unknow'）
                considered_cols = []
                col_weights = []

                for col in common_cols:
                    v1 = row1[col]
                    v2 = row2[col]
                    if pd.isna(v1) or pd.isna(v2):
                        continue
                    if isinstance(v1, str) and v1 == 'unknow':
                        continue
                    if isinstance(v2, str) and v2 == 'unknow':
                        continue

                    considered_cols.append(col)
                    # 计算该列的基础权重
                    if use_scheme_b:
                        canonical = attr_name_map[col]
                        col_weights.append(scheme_b_weights.get(canonical, 0.0))
                    else:
                        col_weights.append(1.0)  # 等权，稍后归一化

                if not considered_cols:
                    current_score = 0.0
                else:
                    # 归一化列权重，使其和为1
                    total_w = sum(col_weights)
                    if total_w <= 0:
                        # 退化到等权
                        col_weights = [1.0] * len(considered_cols)
                        total_w = float(len(considered_cols))
                    norm_weights = [w / total_w for w in col_weights]

                    # 逐列计算匹配得分（数值属性用容差，类别属性用相等）
                    current_score = 0.0
                    for col, w in zip(considered_cols, norm_weights):
                        v1 = row1[col]
                        v2 = row2[col]
                        # 数值匹配（含整型/浮点）
                        if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                            try:
                                if abs(float(v1) - float(v2)) <= tolerance:
                                    current_score += w
                            except Exception:
                                # 解析失败时退化为严格相等
                                if v1 == v2:
                                    current_score += w
                        else:
                            # 类别匹配
                            if v1 == v2:
                                current_score += w

                if current_score > best_score:
                    best_score = current_score
                    best_match_idx = idx2

            if best_match_idx is not None:
                new_row = row1.copy()
                new_row['match_score'] = float(best_score)
                new_row['matched_with'] = best_match_idx
                new_row['total_possible_score'] = 1.0
                result_data.append(new_row)

        return pd.DataFrame(result_data) if result_data else pd.DataFrame()
    
    def clear_new_elems(self):
        # 清除新元素
        self.elem_df = self.elem_df[~self.elem_df.index.str.startswith('NewElem')]
    
    def show_matching_details(self, matched_df, test_df):
        """
        显示匹配详情，包括每个新元素的匹配分数和匹配的真实元素
        """
        if matched_df.empty:
            return "没有匹配的元素"
        
        details = []
        for idx, row in matched_df.iterrows():
            if 'matched_with' in row and 'match_score' in row:
                matched_elem = row['matched_with']
                score = row['match_score']
                
                # 获取匹配的真实元素信息
                if matched_elem in test_df.index:
                    real_elem_info = test_df.loc[matched_elem]
                    details.append(f"{idx}: 匹配 {matched_elem} (分数: {score:.2f}/1.00)")
                else:
                    details.append(f"{idx}: 匹配分数 {score:.2f}/1.00 (但匹配元素不存在)")
            else:
                details.append(f"{idx}: 无匹配信息")
        
        return "\n".join(details)
    
    def rollback_to_state(self, target_table_state):
        """
        回滚到指定的表格状态
        
        参数:
            target_table_state: 目标表格状态（TableState对象）
        """
        if target_table_state is not None:
            # 恢复表格数据
            self.elem_df = target_table_state.elem_df.copy()
            # 恢复属性列表
            self.elem_attrs = list(self.elem_df.columns)
            print("✅ 表格状态已回滚")
        else:
            print("⚠️  目标状态为空，无法回滚")
    
    def get_constraint_status(self):
        """
        获取当前表格的约束状态
        
        返回:
            dict: 包含各种约束检查结果的字典
        """
        status = {
            'has_position_conflicts': False,
            'has_duplicate_positions': False,
            'has_invalid_positions': False,
            'constraint_violations': []
        }
        
        # 检查位置冲突
        if 'row' in self.elem_df.columns and 'col' in self.elem_df.columns:
            filled_elements = self.elem_df.dropna(subset=['row', 'col'])
            if len(filled_elements) > 0:
                # 检查是否有重复位置
                positions = filled_elements[['row', 'col']].drop_duplicates()
                if len(positions) < len(filled_elements):
                    status['has_duplicate_positions'] = True
                    status['constraint_violations'].append("存在元素位置冲突")
                
                # 检查位置值是否合理
                if (filled_elements['row'] < 1).any() or (filled_elements['col'] < 1).any():
                    status['has_invalid_positions'] = True
                    status['constraint_violations'].append("存在无效位置值（小于1）")
        
        return status
    
    def copy(self):
        """
        深度复制当前表格状态
        
        返回:
            TableState: 新的表格状态对象
        """
        new_table = TableState(self.elem_df.copy(), self.test_df)
        new_table.elem_attrs = self.elem_attrs.copy()
        return new_table

    def get_complete_state(self):
        state_str = 'All elements:\n'
        state_str += self.state_as_long_str() + '\n\n'
        for attr in self.elem_attrs:
            state_str += f'{attr} of current elements in virtual periodic table\n'
            state_str += self.att_table(attr) + '\n\n'
        return state_str

def _infer_function_name_from_code(code: str, preferred_names=None):
    """Try to infer a function name from code. Prefer names in preferred_names if provided."""
    try:
        import ast
        tree = ast.parse(code)
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        if preferred_names:
            for name in preferred_names:
                if name in func_names:
                    return name
        if func_names:
            # pick the last defined function as default
            return func_names[-1]
    except Exception:
        pass
    # Regex fallback
    try:
        import re
        m = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
        if m:
            name = m.group(1)
            if not preferred_names or name in preferred_names:
                return name
    except Exception:
        pass
    if preferred_names:
        # As a last resort, return the first preferred name
        return preferred_names[0]
    return None

def _resolve_func_name(code: str, func_name: str, preferred_names=None) -> str:
    """Resolve a usable function name. If missing/None, infer from code, honoring preferred names."""
    if isinstance(func_name, str) and func_name and func_name.strip().lower() != 'none':
        return func_name.strip()
    inferred = _infer_function_name_from_code(code, preferred_names)
    if inferred and isinstance(inferred, str):
        return inferred
    # Avoid passing literal 'None' into executor
    return (preferred_names[0] if preferred_names else '')

def hypo_gen_and_eval(table, agents, history, decision, logger, max_retries=4, attr_name_map=None, mode='test', current_match_rate=0.0):
    # 清除新元素
    table.clear_new_elems()
    # 获取当前状态
    state = table.get_complete_state()

    # abduction
    print(logger.new_part('Abduction Process'))
    ab_agent = agents['ab_agent']
    if decision == 'C':
        
        attribute_result = ab_agent.select_main_attribute(table,state, history)
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



    hypothesis_result = ab_agent.generate_hypothesis(table,sorted_state, main_attr, history)


    hypothesis = hypothesis_result['hypothesis']
    code = hypothesis_result['code']
    func_name = _resolve_func_name(
        code,
        hypothesis_result.get('func_name'),
        preferred_names=['fill_table', 'solve', 'main', 'apply_hypothesis']
    )
    
    print(f"🧪 执行模式: {'测试模式(only_code_execution)' if mode=='test' else '推理模式(enhanced_code_execution)'}")
    # 执行生成的前向代码
    if mode == 'test':
        execution_result = only_code_execution(
            code, func_name, table.elem_df.copy(), max_retries
        )
    else:
        execution_result = enhanced_code_execution(
            code, func_name, table.elem_df.copy(), hypothesis, max_retries
        )
    
    if not execution_result['success']:
        print_and_enter(f"代码执行失败: {execution_result['error']}")
        raise Exception(f"无法生成可执行的代码: {execution_result['error']}")
    
    actions = execution_result['result']
    
    # 显示执行详情
    print_and_enter(f"✅ 代码执行成功，尝试次数: {execution_result['attempts']}")
    if execution_result.get('test_result'):
        print_and_enter(f"📋 单元测试结果: {execution_result['test_result']['overall_passed']}")

    
    success = True

    print('Reasoning:')
    print_and_enter(hypothesis_result["reasoning"])
    print('Hypothesis:')
    print_and_enter(hypothesis)
    print('Code:')
    print_and_enter(code)

    # 仅当actions非空时才填充位置，避免将有效坐标覆盖为NaN
    if actions and len(actions) > 0:
        table.fill_elem_posi(actions)
    state = table.get_complete_state()
    print(state)

    # deduction
    print(logger.new_part('Deduction Process'))
    de_agent = agents['de_agent']

    pred_result = de_agent.predict_elements(state, hypothesis, code, history,n=10)
    new_elems_posi = pred_result['new_elems_posi']
    inverse_code = pred_result['inverse_code']
    func_name = _resolve_func_name(
        inverse_code,
        pred_result.get('func_name'),
        preferred_names=['inverse_func', 'inverse', 'predict_attributes']
    )
    
    # 执行逆向代码
    if mode == 'test':
        execution_result = only_code_execution(
            inverse_code, func_name, (new_elems_posi, table.elem_df.copy()), max_retries
        )
    else:
        execution_result = enhanced_code_execution(
            inverse_code, func_name, (new_elems_posi, table.elem_df.copy()), 
            f"根据元素位置预测元素属性，符合假设: {hypothesis}", max_retries
        )
    
    if not execution_result['success']:
        print(f"反向代码执行失败: {execution_result['error']}")
        raise Exception(f"无法生成可执行的反向代码: {execution_result['error']}")
    
    new_elems = execution_result['result']
    
    # 显示执行详情
    print(f"✅ 反向代码执行成功，尝试次数: {execution_result['attempts']}")
    if execution_result.get('test_result'):
        print(f"📋 单元测试结果: {execution_result['test_result']['overall_passed']}")

    
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
    
    print('📋 匹配结果详情:')
    if not matched_df.empty:
        print('✅ 找到匹配的元素:')
        print(table.show_matching_details(matched_df, table.test_df))
    else:
        print('❌ 没有找到任何匹配的元素')
    
    # 计算新的匹配率（基于分数）
    if not matched_df.empty:
        # 计算平均匹配分数
        avg_match_score = matched_df['match_score'].mean()
        # 评分规则更新：完全匹配以“测试集元素”为准，多个新元素匹配同一个测试元素只计一次
        threshold = 0.7
        perfect_matched_df = matched_df[matched_df['match_score'] >= threshold]
        unique_matched_targets = set(perfect_matched_df['matched_with'])
        perfect_matches = len(unique_matched_targets)

        # 以测试集元素总数作为基准，衡量“覆盖率”
        total_test_elems = table.test_df.shape[0] if hasattr(table, 'test_df') else 1
        match_rate = perfect_matches / total_test_elems if total_test_elems > 0 else 0.0
        
        # 重复匹配统计：同一测试元素被多个新元素命中
        target_counts = perfect_matched_df['matched_with'].value_counts()
        duplicate_targets = target_counts[target_counts > 1]
        num_duplicate_groups = duplicate_targets.shape[0]
        num_redundant_preds = int(duplicate_targets.sum() - num_duplicate_groups) if num_duplicate_groups > 0 else 0
        
        print(f'\n📊 匹配统计:')
        print(f'   平均匹配分数: {avg_match_score:.3f}')
        print(f'   完全匹配（按测试集去重）: {perfect_matches}')
        print(f'   测试集元素总数: {total_test_elems}')
        print(f'   覆盖率(唯一目标): {match_rate:.3f}')
        if num_duplicate_groups > 0:
            print(f'   ⚠️ 重复匹配目标数: {num_duplicate_groups}，冗余新元素数: {num_redundant_preds}')
        
        # 约束检查和性能评估
        print(f'\n🔍 约束检查和性能评估:')
        constraint_violations = []
        
       
        
        # 检查匹配率是否下降（与历史记录比较）
        if len(history.records) > 1:  # 至少有一个历史记录
            # 过滤掉非字典的记录
            valid_records = [record for record in history.records if isinstance(record, dict)]
            if len(valid_records) > 1:
                # 获取上一次的匹配率
                last_match_rate = valid_records[-1].get('match_rate', 0.0)
                if match_rate < last_match_rate:
                    decline = last_match_rate - match_rate
                    constraint_violations.append(f"匹配率下降: {match_rate:.3f} < {last_match_rate:.3f} (下降 {decline:.3f})")
                    print(f"📉 匹配率下降检测: 当前 {match_rate:.3f} vs 上次 {last_match_rate:.3f}")
        
        # 针对重复匹配的约束提醒
        if num_duplicate_groups > 0:
            constraint_violations.append(
                f"发现多个新元素匹配到同一测试元素：{num_duplicate_groups} 个目标存在重复匹配；重复将按一次计分，冗余预测应减少"
            )
        
        # 使用新的约束检查方法
        constraint_status = table.get_constraint_status()
        if constraint_status['constraint_violations']:
            constraint_violations.extend(constraint_status['constraint_violations'])
        
        if constraint_violations:
            print(f"❌ 发现约束违反:")
            for violation in constraint_violations:
                print(f"   - {violation}")
           
        else:
            print("✅ 约束检查通过，性能表现良好")
            
    else:
        avg_match_score = 0.0
        match_rate = 0.0
        print("❌ 没有找到任何匹配的元素")
        print("💡 建议选择R选项进行回滚")
    
    matched_elem_str = table.state_as_long_str(matched_df)
    
    # 使用传入的current_match_rate作为previous_match_rate
    eval_result = in_agent.evaluate_hypothesis(
        state, hypothesis, matched_elem_str, match_rate, avg_match_score, current_match_rate
    )
    evaluation = eval_result['evaluation']
    decision = eval_result['decision']

    print('Reasoning:')
    print_and_enter(eval_result["reasoning"])
    print('evaluation:')
    print_and_enter(evaluation)
    print('decision:')
    print_and_enter(in_agent.options[decision])

    history.update_records(hypothesis, evaluation, match_rate, main_attr, ascending, decision)

    return table, history, decision, matched_df

if __name__ == '__main__':
    
    data_path = r'/Users/yuliachu/Documents/GitHub/SciPuzzleLLM_cy/cy_track/data'# make table data and object
    #table_file = join(data_path, 'PeriodicTable.csv')
    #table_file = join(data_path, 'PeriodicTable_mjlf33.csv')
    #table_file = join(data_path, 'PeriodicTable_complete.csv')
    #table_file = join(data_path, 'PeriodicTableBaseline.csv')
    table_file = join(data_path, 'PeriodicTable_puzzle2.csv')
    if not os.path.exists(table_file):
        df = generate_table()
        df.to_csv(table_file, index=False)
    else:
        df = pd.read_csv(table_file, index_col=None)

    train_df, test_df, attr_name_map = mask_table(df)
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
    #history.load_records_from_log(join(data_path, 'logs', '2025-07-04-13-17-03'), iteration=1)
    logger = Logger(join(data_path, 'logs', 'new_experiment'))
    
    #logger = Logger(join(data_path, 'logs'))
    max_iter = 7
    max_retries = 1
    decision = 'C'
    matched_df = None
    
    # 统计信息
    successful_iterations = 0
    failed_iterations = 0
    
    # 保存初始状态（作为回滚的基准点）
    initial_table_state = table
    initial_history_state = history
    initial_decision_state = decision
    initial_matched_df = matched_df
    
    # 取消 last_successful_* 策略，统一依赖回滚到 pre_induction_* 状态
    
    try:
        for i in range(max_iter):

            print(logger.new_part(f'Iteration {i+1}'))

            try:
                # 在induction阶段之前保存当前状态（深度复制）
                pre_induction_table = table.copy() if hasattr(table, 'copy') else table
                pre_induction_history = history.copy() if hasattr(history, 'copy') else history
                pre_induction_decision = decision
                pre_induction_matched_df = matched_df
                
                # 保存当前的匹配率，用于下次迭代比较
                current_match_rate = 0.0
                if len(history.records) > 0:
                    last_record = history.records[-1]
                    if isinstance(last_record, dict) and 'match_rate' in last_record:
                        current_match_rate = last_record['match_rate']
                
                print(f"📋 保存迭代前状态:")
                print(f"   表格元素数量: {len(table.elem_df) if hasattr(table, 'elem_df') else 'N/A'}")
                print(f"   历史记录数量: {len(history.records) if hasattr(history, 'records') else 'N/A'}")
                print(f"   当前决策: {decision}")
                print(f"   当前匹配率: {current_match_rate:.3f}")
                
                table, history, decision, matched_df = hypo_gen_and_eval(
                    table, agents, history, decision, logger, max_retries, attr_name_map, mode='work', current_match_rate=current_match_rate)
                
                print(f"✅ 第 {i+1} 轮迭代成功完成")
                successful_iterations += 1
                
                if decision == 'P':
                    print('Stop as the hypothesis is accepted!')
                    break
                elif decision == 'R':
                    print('🔄 Decision R: Rolling back to previous version due to constraint violation or poor performance')
                    # 回滚到induction阶段之前的状态
                    if pre_induction_table is not None:
                        table = pre_induction_table
                        history = pre_induction_history
                        decision = pre_induction_decision
                        matched_df = pre_induction_matched_df # 清空匹配结果
                        
                        # 删除上一条历史记录（当前失败的迭代记录）
                        if len(history.records) > 1:  # 确保至少有一条记录可以删除
                            removed_record = history.records.pop()
                            print(f"🗑️ 已删除需要回滚的迭代记录: {removed_record.get('hypothesis', 'N/A')[:50]}...")
                        
                        print("✅ 已回滚到induction阶段之前的状态")
                        print(f"🔄 回滚后的决策: {decision}")
                        
                        # 验证回滚是否成功
                        print(f"📋 回滚后状态验证:")
                        print(f"   表格元素数量: {len(table.elem_df) if hasattr(table, 'elem_df') else 'N/A'}")
                        print(f"   历史记录数量: {len(history.records) if hasattr(history, 'records') else 'N/A'}")
                        print(f"   决策状态: {decision}")
                        
                        # 继续下一次迭代，跳过当前失败的版本
                        continue
                    else:
                        print("⚠️  无法回滚，使用初始状态")
                        table = initial_table_state
                        history = initial_history_state
                        decision = initial_decision_state
                        matched_df = None
                        
                        print(f"📋 使用初始状态:")
                        print(f"   表格元素数量: {len(table.elem_df) if hasattr(table, 'elem_df') else 'N/A'}")
                        print(f"   历史记录数量: {len(history.records) if hasattr(history, 'records') else 'N/A'}")
                        print(f"   决策状态: {decision}")
                        
                        continue
                    
            except Exception as e:
                print(f"❌ 第 {i+1} 轮迭代出现错误: {e}")
                failed_iterations += 1
                print("🔄 跳过本次迭代，使用上次迭代的正常结果继续...")
                
                # 出错时回滚到本轮 induction 前保存的状态（注意：异常情况下记录未保存，不需要删除）
                if 'pre_induction_table' in locals() and pre_induction_table is not None:
                    table = pre_induction_table
                    history = pre_induction_history
                    decision = pre_induction_decision
                    matched_df = pre_induction_matched_df
                    
                    # 异常情况下记录未保存，不需要删除历史记录
                    print("✅ 已回滚到induction阶段之前的状态（错误处理）")
                    print("📝 注意：异常情况下记录未保存，无需删除历史记录")
                else:
                    print("⚠️  无法回滚，使用初始状态（错误处理）")
                    table = initial_table_state
                    history = initial_history_state
                    decision = initial_decision_state
                    matched_df = initial_matched_df
                
                # 记录错误到日志
                print(f"❌ 第 {i+1} 轮迭代错误记录: {e}")
                continue

        if i == max_iter - 1:
            print(f'Stop as the max iteration {i+1} is reached!')

        # 显示迭代统计信息
        print(f'\n📊 迭代统计信息:')
        print(f'   成功迭代次数: {successful_iterations}')
        print(f'   失败迭代次数: {failed_iterations}')
        print(f'   总迭代次数: {max_iter}')
        print(f'   成功率: {successful_iterations/max_iter*100:.1f}%')

        print('\nhistory:')
        print(history.show_records())

        """  # 选择最优假设并进行提炼总结
        print("\n" + "="*50)
        print("选择最优假设并进行提炼总结")
        print("="*50) 
        
        record_agent = RecordAgent()
        optimal_hypothesis_summary = record_agent.merge_records(history.records)
        
        if optimal_hypothesis_summary:
            try:
                # 解析JSON响应（健壮化处理）
                summary_data = None
                import json, re
                if isinstance(optimal_hypothesis_summary, dict):
                    summary_data = optimal_hypothesis_summary
                else:
                    raw_text = str(optimal_hypothesis_summary).strip()
                    if raw_text:
                        try:
                            summary_data = json.loads(raw_text)
                        except Exception:
                            # 尝试提取第一个JSON对象
                            m = re.search(r"\{[\s\S]*\}", raw_text)
                            if m:
                                try:
                                    summary_data = json.loads(m.group(0))
                                except Exception:
                                    summary_data = None
                
                if not isinstance(summary_data, dict):
                    print("⚠️ 最优假设返回格式非JSON，跳过应用最优假设。原始内容预览：")
                    print(str(optimal_hypothesis_summary)[:300])
                    summary_data = None
                
                if summary_data is not None:
                    print("✅ 最优假设总结:")
                    print(f"   假设: {summary_data.get('hypothesis', 'N/A')}")
                    print(f"   主属性: {summary_data.get('attribute', 'N/A')}")
                    print(f"   升序排列: {summary_data.get('ascending', 'N/A')}")
                    print(f"   匹配率: {summary_data.get('match_rate', 'N/A')}")
                
                    # 将最优假设写入log目录，便于复现和绘图
                    try:
                        summary_path = join(logger.log_folder, 'optimal_hypothesis.json')
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            json.dump(summary_data, f, ensure_ascii=False, indent=2)
                        print(f"📝 已保存最优假设到: {summary_path}")
                    except Exception as e:
                        print(f"⚠️ 最优假设保存失败: {e}")
                
                    # 将最优假设应用到最终表格
                    if 'attribute' in summary_data and summary_data['attribute']:
                        main_attr = summary_data['attribute']
                        ascending = summary_data.get('ascending', True)
                        if isinstance(ascending, str):
                            ascending = ascending.lower() == 'true'
                        
                        print(f"\n🔄 应用最优假设到最终表格:")
                        print(f"   主属性: {main_attr}")
                        print(f"   升序排列: {ascending}")
                        
                        # 重新排序表格
                        table.sort_table(main_attr, ascending=ascending)
                        final_state = table.get_complete_state()
                        print("✅ 表格已按最优假设重新排序")
                    
            except Exception as e:
                print(f"⚠️  解析最优假设总结时出错: {e}")
                print("使用当前表格状态进行可视化")
        else:
            print("⚠️  无法获取最优假设总结，使用当前表格状态") """

        final_df = table.elem_df.copy()
        # 新增 KnownAndMatched 列，根据匹配分数分类
        
        if matched_df is not None and not matched_df.empty:
            # 创建匹配分数映射和重复匹配处理
            threshold = 0.7
            match_score_map = {}
            target_matches = {}  # 记录每个测试目标的匹配情况
            
            # 统计每个测试目标的匹配情况
            for idx, row in matched_df.iterrows():
                try:
                    score = float(row.get('match_score', 0.0))
                    target = row.get('matched_with', None)
                    if score >= threshold and target:
                        match_score_map[idx] = score
                        if target not in target_matches:
                            target_matches[target] = []
                        target_matches[target].append((idx, score))
                except Exception:
                    continue
            
            # 处理重复匹配：为每个测试目标只保留一个最佳匹配，记录匹配数
            final_match_info = {}  # 记录最终保留的匹配信息
            for target, matches in target_matches.items():
                if len(matches) > 1:
                    # 按分数排序，保留最高分的
                    matches.sort(key=lambda x: x[1], reverse=True)
                    best_match = matches[0][0]
                    match_count = len(matches)
                    final_match_info[best_match] = {
                        'score': matches[0][1],
                        'match_count': match_count,
                        'target': target
                    }
                    # 移除其他重复匹配
                    for idx, _ in matches[1:]:
                        if idx in match_score_map:
                            del match_score_map[idx]
                else:
                    # 只有一个匹配
                    idx, score = matches[0]
                    final_match_info[idx] = {
                        'score': score,
                        'match_count': 1,
                        'target': target
                    }
            
            # 创建 KnownAndMatched 列
            final_df['KnownAndMatched'] = final_df.index.map(
                lambda elem: 'Known' if not str(elem).startswith('NewElem') 
                else 'KnownAndMatched' if (elem in match_score_map and match_score_map[elem] >= threshold)
                else 'New&Unmatched'
            )
            
            # 添加匹配数信息列
            final_df['MatchCount'] = final_df.index.map(
                lambda elem: final_match_info.get(elem, {}).get('match_count', 1) if elem in final_match_info else 1
            )

            # 调试输出：哪些新元素匹配上了（>=0.7），哪些没有
            matched_items = []
            for idx, info in final_match_info.items():
                matched_items.append((idx, info['target'], info['score'], info['match_count']))

            unmatched_items = [
                elem for elem in final_df.index
                if str(elem).startswith('NewElem') and not (
                    elem in match_score_map and match_score_map[elem] >= threshold
                )
            ]

            print('\n🧩 可视化分类检查:')
            print(f'   达到阈值(>= {threshold})的新元素数: {len(matched_items)}')
            if len(matched_items) > 0:
                preview = '\n'.join([f"      {e} -> {t} (score={s:.2f}, count={c})" for e,t,s,c in matched_items[:10]])
                print('   示例(前10):')
                print(preview)
            print(f'   未达到阈值的新元素数: {len(unmatched_items)}')
            if len(unmatched_items) > 0:
                preview2 = '\n'.join([f"      {e}" for e in unmatched_items[:10]])
                print('   示例(前10):')
                print(preview2)
        else:
            # 如果没有匹配数据，所有新元素都标记为未匹配
            final_df['KnownAndMatched'] = final_df.index.map(
                lambda elem: 'Known' if not str(elem).startswith('NewElem') 
                else 'New&Unmatched'
            )
            final_df['MatchCount'] = 1  # 默认匹配数为1

        logger.log_table_as_csv(final_df)
        
        # 为MatchCount可视化准备匹配数据
        if matched_df is not None and not matched_df.empty:
            # 创建匹配数据的副本，用于可视化
            viz_matched_df = matched_df.copy()
        else:
            viz_matched_df = None
        
        logger.log_table_as_img(final_df, matched_df=viz_matched_df)
        
        # 解析log文件，生成处理后的结果
        print("\n" + "="*50)
        print("开始解析log文件...")
        parsed_output_path = join(logger.log_folder, 'parsed_results.txt')
        parse_log_file(logger.log_folder, parsed_output_path)
        print(f"原始log文件: {join(logger.log_folder, 'log.txt')}")
        print(f"解析结果文件: {parsed_output_path}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        logger.close()
