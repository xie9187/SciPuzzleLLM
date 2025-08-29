import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

def get_luminance(color):
    r, g, b = color[:3]
    return 0.299 * r + 0.587 * g + 0.114 * b

def create_periodic_table_plot(df, attribute, save_path=None, key_idx=0):
    # 设置周期表基本参数
    max_row = df['row'].max()
    max_col = df['col'].max()

    fig, ax = plt.subplots(figsize=(max_col + 1, max_row))
    
    # 创建格子
    box_size = 1.0
    
    # 预处理数据：按(row,col)分组
    grouped = df.groupby(['row', 'col'])
    
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
            if pd.api.types.is_numeric_dtype(df[attribute]):
                norm = Normalize(vmin=df[attribute].min(), vmax=df[attribute].max())
                cmap = plt.cm.viridis
                color = cmap(norm(key_elem[attribute]))
            else:
                unique_values = df[attribute].unique()
                cmap = ListedColormap(sns.color_palette("husl", len(unique_values)))
                color = cmap(list(unique_values).index(key_elem[attribute]))
            
            # 绘制元素格子
            rect = Rectangle((x, y), box_size, box_size, 
                            facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
                
            # 构建合并的文本
            text_lines = []
            for _, elem in elements.iterrows():
                attributes = [f"{elem[col]}" for col in elem.index if col.startswith("Attribute")]
                elem_text = f"{elem.name}\n" + " | ".join(attributes)
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
    
    # 添加颜色条
    if pd.api.types.is_numeric_dtype(df[attribute]):
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')


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
