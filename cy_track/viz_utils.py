import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

def get_luminance(color):
    """计算颜色的亮度值"""
    if isinstance(color, str) and color.startswith('#'):
        # 处理十六进制颜色字符串
        color = color.lstrip('#')
        r = int(color[0:2], 16) / 255.0
        g = int(color[2:4], 16) / 255.0
        b = int(color[4:6], 16) / 255.0
    elif isinstance(color, (tuple, list)) and len(color) >= 3:
        # 处理RGB元组或列表
        r, g, b = color[0], color[1], color[2]
        if max(r, g, b) > 1:  # 如果值大于1，假设是0-255范围
            r, g, b = r/255.0, g/255.0, b/255.0
    else:
        # 默认返回0.5（中等亮度）
        return 0.5
    
    return 0.299 * r + 0.587 * g + 0.114 * b

def create_periodic_table_plot(df, attribute, save_path=None, key_idx=0):
    # 检查数据有效性
    if df.empty:
        print("⚠️ 警告：数据框为空，无法创建可视化")
        return
    
    # 检查必要的列是否存在
    required_cols = ['row', 'col']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告：缺少必要的列: {missing_cols}")
        return
    
    # 过滤掉row或col为NaN的行
    valid_df = df.dropna(subset=['row', 'col'])
    if valid_df.empty:
        print("⚠️ 警告：所有行的row或col值都是NaN，无法创建可视化")
        return
    
    # 检查row和col是否为数值类型
    if not pd.api.types.is_numeric_dtype(valid_df['row']) or not pd.api.types.is_numeric_dtype(valid_df['col']):
        try:
            valid_df['row'] = pd.to_numeric(valid_df['row'], errors='coerce')
            valid_df['col'] = pd.to_numeric(valid_df['col'], errors='coerce')
            # 再次过滤NaN值
            valid_df = valid_df.dropna(subset=['row', 'col'])
            if valid_df.empty:
                print("⚠️ 警告：转换后所有行的row或col值都是NaN，无法创建可视化")
                return
        except Exception as e:
            print(f"❌ 错误：无法转换row或col列为数值类型: {e}")
            return
    
    # 设置周期表基本参数
    max_row = int(valid_df['row'].max())
    max_col = int(valid_df['col'].max())
    
    # 检查最大值是否合理
    if max_row <= 0 or max_col <= 0:
        print(f"⚠️ 警告：无效的行列值: max_row={max_row}, max_col={max_col}")
        return
    
   
    fig, ax = plt.subplots(figsize=(max_col + 1, max_row))
    
    # 创建格子
    box_size = 1.0
    
    # 预处理数据：按(row,col)分组
    grouped = valid_df.groupby(['row', 'col'])
    
    # 遍历所有可能的(row,col)位置
    for row_num in range(1, max_row + 1):
        for col_num in range(1, max_col + 1):
            if (row_num, col_num) not in grouped.groups:
                continue  # 跳过没有元素的位置
                
            # 获取该位置的所有元素
            elements = grouped.get_group((row_num, col_num))
            x = col_num - 0.5
            y = row_num - 0.5  # 翻转y轴使第1周期在上
            
            # 使用key元素的属性决定颜色
            key_elem = elements.iloc[key_idx]
            
            # 特殊处理MatchCount属性：区分匹配为1和没匹配到的元素
            if attribute == 'MatchCount':
                # 为MatchCount创建特殊的颜色映射
                colors = []
                for _, elem in elements.iterrows():
                    if 'KnownAndMatched' in elem:
                        if elem['KnownAndMatched'] == 'Known':
                            # 已知元素：绿色
                            colors.append('#90EE90')  # 浅绿色
                        elif elem['KnownAndMatched'] == 'New&Matched':
                            # 新元素且匹配成功：蓝色
                            colors.append('#87CEEB')  # 天蓝色
                        else:
                            # 新元素但未匹配：红色
                            colors.append('#FFB6C1')  # 浅红色
                    else:
                        # 如果没有KnownAndMatched列，根据MatchCount判断
                        match_count = elem.get('MatchCount', 1)
                        if match_count > 1:
                            colors.append('#87CEEB')  # 匹配成功：蓝色
                        elif match_count == 1:
                            if str(elem.name).startswith('NewElem'):
                                colors.append('#FFB6C1')  # 新元素但匹配数为1：红色
                            else:
                                colors.append('#90EE90')  # 已知元素：绿色
                        else:
                            colors.append('#FFB6C1')  # 匹配数为0：红色
                
                # 使用第一个元素的颜色作为背景色
                color = colors[0] if colors else '#FFFFFF'
                
            elif pd.api.types.is_numeric_dtype(valid_df[attribute]):
                norm = Normalize(vmin=valid_df[attribute].min(), vmax=valid_df[attribute].max())
                cmap = plt.cm.viridis
                color = cmap(norm(key_elem[attribute]))
            else:
                unique_values = valid_df[attribute].unique()
                cmap = ListedColormap(sns.color_palette("husl", len(unique_values)))
                color = cmap(list(unique_values).index(key_elem[attribute]))
            
            # 绘制元素格子
            rect = Rectangle((x, y), box_size, box_size, 
                            facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
                
            # 构建合并的文本
            text_lines = []
            for i, (_, elem) in enumerate(elements.iterrows()):
                # 安全地获取属性值，避免KeyError
                attr1 = elem.get('Attribute1', 'N/A')
                attr2 = elem.get('Attribute2', 'N/A')
                attr3 = elem.get('Attribute3', 'N/A')
                attr4 = elem.get('Attribute4', 'N/A')
                
                # 格式化数值
                if pd.api.types.is_numeric_dtype(type(attr1)):
                    attr1_str = f"{attr1:.2f}"
                else:
                    attr1_str = str(attr1)
                
                elem_text = f"{elem.name}\n{attr1_str}|{attr2}\n{attr3}|{attr4}"
                
                # 特殊处理MatchCount属性的文本显示
                if attribute == 'MatchCount':
                    if 'KnownAndMatched' in elem:
                        if elem['KnownAndMatched'] == 'Known':
                            elem_text += "\n(Known)"
                        elif elem['KnownAndMatched'] == 'New&Matched':
                            match_count = elem.get('MatchCount', 1)
                            elem_text += f"\n(Matched×{match_count})"
                        else:
                            elem_text += "\n(Unmatched)"
                    else:
                        # 如果没有KnownAndMatched列，根据MatchCount判断
                        match_count = elem.get('MatchCount', 1)
                        if match_count > 1:
                            elem_text += f"\n(×{match_count})"
                        elif match_count == 1:
                            if str(elem.name).startswith('NewElem'):
                                elem_text += "\n(Unmatched)"
                            else:
                                elem_text += "\n(Known)"
                        else:
                            elem_text += "\n(Unmatched)"
                else:
                    # 如果有匹配数信息且大于1，添加匹配数标注
                    if 'MatchCount' in elem and elem['MatchCount'] > 1:
                        elem_text += f"\n(×{elem['MatchCount']})"
                
                text_lines.append(elem_text)
            combined_text = '\n'.join(text_lines)
            
            # 根据元素数量调整字体大小
            n_elem = len(elements)
            fontsize = 7 if len(elements) <= 2 else 5 if len(elements) <= 4 else 3

            # 计算颜色的亮度并确定文字颜色
            text_color = 'white' if get_luminance(color) < 0.5 else 'black'
            
            ax.text(x + box_size/2, y + box_size/2, combined_text, 
                    ha='center', va='center', fontsize=fontsize, color=text_color)
    
    # 设置坐标轴
    ax.set_xlim(0.5, max_col + 0.5)
    ax.set_ylim(0.5, max_row + 0.5)
    ax.set_xticks(np.arange(1, max_col + 1))
    ax.set_yticks(np.arange(1, max_row + 1))
    ax.invert_yaxis()  # 使第1周期在上
    ax.set_aspect('equal')
    ax.grid(False)
    
    # 添加颜色条或图例
    if attribute == 'MatchCount':
        # 为MatchCount创建自定义图例
        legend_elements = [
            Rectangle((0,0),1,1, facecolor='#90EE90', edgecolor='black', label='Known Elements'),
            Rectangle((0,0),1,1, facecolor='#87CEEB', edgecolor='black', label='Matched New Elements'),
            Rectangle((0,0),1,1, facecolor='#FFB6C1', edgecolor='black', label='Unmatched New Elements')
        ]
        ax.legend(handles=legend_elements, title='Element Status', loc='best')
    elif pd.api.types.is_numeric_dtype(valid_df[attribute]):
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label(attribute, rotation=270, labelpad=15)
    else:
        # 对于分类变量创建图例
        handles = [Rectangle((0,0),1,1, color=cmap(i)) for i in range(len(unique_values))]
        ax.legend(handles, unique_values, title=attribute, loc='best')
    
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化已保存到: {save_path}")
        except Exception as e:
            print(f"保存可视化失败: {e}")
        finally:
            plt.close()  # 关闭图形以释放内存


def create_matchcount_visualization(df, matched_df=None, save_path=None):
    """
    创建MatchCount可视化，只显示训练元素和被匹配到的测试元素
    
    Args:
        df: 包含所有元素的数据框
        matched_df: 匹配结果数据框（可选）
        save_path: 保存路径
    """
    # 检查数据有效性
    if df.empty:
        print("⚠️ 警告：数据框为空，无法创建MatchCount可视化")
        return
    
    # 检查必要的列是否存在
    required_cols = ['row', 'col']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告：缺少必要的列: {missing_cols}")
        return
    
    # 过滤掉row或col为NaN的行
    valid_df = df.dropna(subset=['row', 'col'])
    if valid_df.empty:
        print("⚠️ 警告：所有行的row或col值都是NaN，无法创建MatchCount可视化")
        return
    
    # 检查row和col是否为数值类型
    if not pd.api.types.is_numeric_dtype(valid_df['row']) or not pd.api.types.is_numeric_dtype(valid_df['col']):
        try:
            valid_df['row'] = pd.to_numeric(valid_df['row'], errors='coerce')
            valid_df['col'] = pd.to_numeric(valid_df['col'], errors='coerce')
            valid_df = valid_df.dropna(subset=['row', 'col'])
            if valid_df.empty:
                print("⚠️ 警告：转换后所有行的row或col值都是NaN，无法创建MatchCount可视化")
                return
        except Exception as e:
            print(f"❌ 错误：无法转换row或col列为数值类型: {e}")
            return
    
    # 创建过滤后的数据框，只包含训练元素和被匹配到的测试元素
    filtered_df = valid_df.copy()
    
    if matched_df is not None and not matched_df.empty:
        # 如果有匹配数据，只保留训练元素和被匹配到的测试元素
        matched_elements = set(matched_df.index)
        filtered_df = filtered_df[
            (~filtered_df.index.str.startswith('NewElem')) |  # 保留所有训练元素
            (filtered_df.index.isin(matched_elements))  # 保留被匹配到的测试元素
        ]
    
    if filtered_df.empty:
        print("⚠️ 警告：过滤后没有数据，无法创建MatchCount可视化")
        return
    
    # 设置周期表基本参数
    max_row = int(filtered_df['row'].max())
    max_col = int(filtered_df['col'].max())
    
    if max_row <= 0 or max_col <= 0:
        print(f"⚠️ 警告：无效的行列值: max_row={max_row}, max_col={max_col}")
        return
    
    print(f"📊 创建MatchCount可视化: {max_row}行 × {max_col}列")
    print(f"   训练元素数: {len(filtered_df[~filtered_df.index.str.startswith('NewElem')])}")
    print(f"   被匹配的测试元素数: {len(filtered_df[filtered_df.index.str.startswith('NewElem')])}")

    fig, ax = plt.subplots(figsize=(max_col + 1, max_row))
    
    # 创建格子
    box_size = 1.0
    
    # 预处理数据：按(row,col)分组
    grouped = filtered_df.groupby(['row', 'col'])
    
    # 遍历所有可能的(row,col)位置
    for row_num in range(1, max_row + 1):
        for col_num in range(1, max_col + 1):
            if (row_num, col_num) not in grouped.groups:
                continue  # 跳过没有元素的位置
                
            # 获取该位置的所有元素
            elements = grouped.get_group((row_num, col_num))
            x = col_num - 0.5
            y = row_num - 0.5  # 翻转y轴使第1周期在上
            
            # 为MatchCount创建特殊的颜色映射
            colors = []
            for _, elem in elements.iterrows():
                if str(elem.name).startswith('NewElem'):
                    # 测试元素：根据匹配数量决定颜色
                    match_count = elem.get('MatchCount', 1)
                    if match_count > 1:
                        # 多次匹配：深蓝色
                        colors.append('#4169E1')  # 皇家蓝
                    elif match_count == 1:
                        # 单次匹配：天蓝色
                        colors.append('#87CEEB')  # 天蓝色
                    else:
                        # 未匹配：红色
                        colors.append('#FF6B6B')  # 珊瑚红
                else:
                    # 训练元素：绿色系
                    colors.append('#90EE90')  # 浅绿色
            
            # 使用第一个元素的颜色作为背景色
            color = colors[0] if colors else '#FFFFFF'
            
            # 绘制元素格子
            rect = Rectangle((x, y), box_size, box_size, 
                            facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
                
            # 构建合并的文本
            text_lines = []
            for _, elem in elements.iterrows():
                # 安全地获取属性值
                attr1 = elem.get('Attribute1', 'N/A')
                attr2 = elem.get('Attribute2', 'N/A')
                attr3 = elem.get('Attribute3', 'N/A')
                attr4 = elem.get('Attribute4', 'N/A')
                
                # 格式化数值
                if pd.api.types.is_numeric_dtype(type(attr1)):
                    attr1_str = f"{attr1:.2f}"
                else:
                    attr1_str = str(attr1)
                
                elem_text = f"{elem.name}\n{attr1_str}|{attr2}\n{attr3}|{attr4}"
                
                # 添加匹配信息
                if str(elem.name).startswith('NewElem'):
                    match_count = elem.get('MatchCount', 1)
                    if match_count > 1:
                        elem_text += f"\n(×{match_count})"
                    elif match_count == 1:
                        elem_text += "\n(Matched)"
                    else:
                        elem_text += "\n(Unmatched)"
                else:
                    elem_text += "\n(Train)"
                
                text_lines.append(elem_text)
            
            combined_text = '\n'.join(text_lines)
            
            # 根据元素数量调整字体大小
            fontsize = 7 if len(elements) <= 2 else 5 if len(elements) <= 4 else 3

            # 计算颜色的亮度并确定文字颜色
            text_color = 'white' if get_luminance(color) < 0.5 else 'black'
            
            ax.text(x + box_size/2, y + box_size/2, combined_text, 
                    ha='center', va='center', fontsize=fontsize, color=text_color)
    
    # 设置坐标轴
    ax.set_xlim(0.5, max_col + 0.5)
    ax.set_ylim(0.5, max_row + 0.5)
    ax.set_xticks(np.arange(1, max_col + 1))
    ax.set_yticks(np.arange(1, max_row + 1))
    ax.invert_yaxis()  # 使第1周期在上
    ax.set_aspect('equal')
    ax.grid(False)
    
    # 创建自定义图例
    legend_elements = [
        Rectangle((0,0),1,1, facecolor='#90EE90', edgecolor='black', label='Training Elements'),
        Rectangle((0,0),1,1, facecolor='#87CEEB', edgecolor='black', label='Single Match'),
        Rectangle((0,0),1,1, facecolor='#4169E1', edgecolor='black', label='Multiple Matches'),
        Rectangle((0,0),1,1, facecolor='#FF6B6B', edgecolor='black', label='Unmatched')
    ]
    ax.legend(handles=legend_elements, title='Element Status', loc='best')
    
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ MatchCount可视化已保存到: {save_path}")
        except Exception as e:
            print(f"❌ 保存MatchCount可视化失败: {e}")
        finally:
            plt.close()  # 关闭图形以释放内存


def create_matchcount_plot(df, matched_df, test_df, save_path=None):
    """
    专门用于绘制MatchCount的可视化，只显示训练元素和完全匹配的测试元素
    
    Args:
        df: 包含所有元素的数据框（训练元素+新元素）
        matched_df: 匹配结果数据框
        test_df: 测试集数据框
        save_path: 保存路径
    """
    # 检查数据有效性
    if df.empty:
        print("⚠️ 警告：数据框为空，无法创建可视化")
        return
    
    # 检查必要的列是否存在
    required_cols = ['row', 'col']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️ 警告：缺少必要的列: {missing_cols}")
        return
    
    # 过滤掉row或col为NaN的行
    valid_df = df.dropna(subset=['row', 'col'])
    if valid_df.empty:
        print("⚠️ 警告：所有行的row或col值都是NaN，无法创建可视化")
        return
    
    # 检查row和col是否为数值类型
    if not pd.api.types.is_numeric_dtype(valid_df['row']) or not pd.api.types.is_numeric_dtype(valid_df['col']):
        try:
            valid_df['row'] = pd.to_numeric(valid_df['row'], errors='coerce')
            valid_df['col'] = pd.to_numeric(valid_df['col'], errors='coerce')
            # 再次过滤NaN值
            valid_df = valid_df.dropna(subset=['row', 'col'])
            if valid_df.empty:
                print("⚠️ 警告：转换后所有行的row或col值都是NaN，无法创建可视化")
                return
        except Exception as e:
            print(f"❌ 错误：无法转换row或col列为数值类型: {e}")
            return
    
    # 设置周期表基本参数
    max_row = int(valid_df['row'].max())
    max_col = int(valid_df['col'].max())
    
    # 检查最大值是否合理
    if max_row <= 0 or max_col <= 0:
        print(f"⚠️ 警告：无效的行列值: max_row={max_row}, max_col={max_col}")
        return
    
    print(f"📊 创建MatchCount可视化: {max_row}行 × {max_col}列")

    fig, ax = plt.subplots(figsize=(max_col + 1, max_row))
    
    # 创建格子
    box_size = 1.0
    
    # 预处理数据：按(row,col)分组
    grouped = valid_df.groupby(['row', 'col'])
    
    # 定义颜色方案
    colors = {
        'training': '#90EE90',      # 浅绿色：训练元素
        'matched_1': '#87CEEB',     # 天蓝色：匹配1次
        'matched_2': '#4682B4',     # 钢蓝色：匹配2次
        'matched_3': '#191970',     # 午夜蓝：匹配3次
        'matched_4+': '#000080',    # 深蓝色：匹配4次及以上
    }
    
    # 遍历所有可能的(row,col)位置
    for row_num in range(1, max_row + 1):
        for col_num in range(1, max_col + 1):
            if (row_num, col_num) not in grouped.groups:
                continue  # 跳过没有元素的位置
                
            # 获取该位置的所有元素
            elements = grouped.get_group((row_num, col_num))
            x = col_num - 0.5
            y = row_num - 0.5  # 翻转y轴使第1周期在上
            
            # 确定该位置的颜色
            position_color = None
            element_texts = []
            
            for _, elem in elements.iterrows():
                elem_name = elem.name
                
                # 判断元素类型和匹配状态
                if not str(elem_name).startswith('NewElem'):
                    # 训练元素：绿色
                    elem_color = colors['training']
                    elem_text = f"{elem_name}\n(Training)"
                    position_color = elem_color
                else:
                    # 新元素：检查匹配状态
                    if matched_df is not None and not matched_df.empty:
                        # 查找匹配信息
                        matched_info = matched_df[matched_df.index == elem_name]
                        if not matched_info.empty:
                            match_score = matched_info.iloc[0].get('match_score', 0.0)
                            matched_with = matched_info.iloc[0].get('matched_with', '')
                            
                            if match_score >= 0.7:  # 完全匹配阈值
                                # 计算重复匹配数量
                                duplicate_count = 0
                                if matched_with:
                                    # 统计有多少个新元素匹配到同一个测试元素
                                    same_target = matched_df[matched_df['matched_with'] == matched_with]
                                    duplicate_count = len(same_target)
                                
                                # 根据重复匹配数量选择颜色
                                if duplicate_count == 1:
                                    elem_color = colors['matched_1']
                                    elem_text = f"{elem_name}\n(Matched×1)"
                                elif duplicate_count == 2:
                                    elem_color = colors['matched_2']
                                    elem_text = f"{elem_name}\n(Matched×2)"
                                elif duplicate_count == 3:
                                    elem_color = colors['matched_3']
                                    elem_text = f"{elem_name}\n(Matched×3)"
                                else:
                                    elem_color = colors['matched_4+']
                                    elem_text = f"{elem_name}\n(Matched×{duplicate_count})"
                                
                                position_color = elem_color
                            else:
                                # 未完全匹配：不显示
                                continue
                        else:
                            # 未匹配：不显示
                            continue
                    else:
                        # 没有匹配数据：不显示
                        continue
                
                element_texts.append(elem_text)
            
            # 如果该位置没有要显示的元素，跳过
            if not element_texts:
                continue
            
            # 绘制元素格子
            rect = Rectangle((x, y), box_size, box_size, 
                            facecolor=position_color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # 合并文本
            combined_text = '\n'.join(element_texts)
            
            # 根据元素数量调整字体大小
            fontsize = 7 if len(element_texts) <= 2 else 5 if len(element_texts) <= 4 else 3
            
            # 计算颜色的亮度并确定文字颜色
            text_color = 'white' if get_luminance(position_color) < 0.5 else 'black'
            
            ax.text(x + box_size/2, y + box_size/2, combined_text, 
                    ha='center', va='center', fontsize=fontsize, color=text_color)
    
    # 设置坐标轴
    ax.set_xlim(0.5, max_col + 0.5)
    ax.set_ylim(0.5, max_row + 0.5)
    ax.set_xticks(np.arange(1, max_col + 1))
    ax.set_yticks(np.arange(1, max_row + 1))
    ax.invert_yaxis()  # 使第1周期在上
    ax.set_aspect('equal')
    ax.grid(False)
    
    # 创建自定义图例
    legend_elements = [
        Rectangle((0,0),1,1, facecolor=colors['training'], edgecolor='black', label='Training Elements'),
        Rectangle((0,0),1,1, facecolor=colors['matched_1'], edgecolor='black', label='Matched×1'),
        Rectangle((0,0),1,1, facecolor=colors['matched_2'], edgecolor='black', label='Matched×2'),
        Rectangle((0,0),1,1, facecolor=colors['matched_3'], edgecolor='black', label='Matched×3'),
        Rectangle((0,0),1,1, facecolor=colors['matched_4+'], edgecolor='black', label='Matched×4+')
    ]
    ax.legend(handles=legend_elements, title='Element Status', loc='best')
    
    plt.tight_layout()

    if save_path is None:
        plt.show()
    else:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ MatchCount可视化已保存到: {save_path}")
        except Exception as e:
            print(f"❌ 保存MatchCount可视化失败: {e}")
        finally:
            plt.close()  # 关闭图形以释放内存


if __name__ == '__main__':
    from os.path import join

    # 示例数据 (替换为你的实际数据)
    log_path = rf'D:\Data\SciPuzzleLLM\logs\{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
    df = pd.read_csv(join(log_path, 'table.csv'), index_col='Element')

    matched_df = df.copy().drop(df.index)

    matched_elements = set(matched_df.index) if matched_df is not None else set()
    df['KnownAndMatched'] = df.index.map(
        lambda elem: 'Known' if not str(elem).startswith('NewElem') 
        else 'New&Matched' if elem in matched_elements 
        else 'New&Unmatched'
    )

    # Attribute1 (数值型)
    for attribute in ['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'KnownAndMatched']:
        key_idx = -1 if attribute == 'KnownAndMatched' else 0
        create_periodic_table_plot(df, attribute, join(log_path, f'table_{attribute}.png'), key_idx=key_idx)
