import pandas as pd
import os
from pathlib import Path

def merge_excel_files(folder_path, output_file):
    # 创建一个空列表来存储所有数据框
    all_dataframes = []
    
    # 遍历文件夹中的所有文件
    for file in Path(folder_path).glob('*.xlsx'):
        try:
            # 读取Excel文件
            df = pd.read_excel(file)
            # 添加一列来标识数据来源
            df['来源文件'] = file.name
            # 将数据框添加到列表中
            all_dataframes.append(df)
            print(f'成功读取文件: {file.name}')
        except Exception as e:
            print(f'读取文件 {file.name} 时出错: {str(e)}')
    
    # 如果没有找到任何Excel文件
    if not all_dataframes:
        print('未找到任何Excel文件！')
        return
    
    # 合并所有数据框
    merged_df = pd.concat(all_dataframes, ignore_index=True)
    
    # 保存合并后的数据到新的Excel文件
    merged_df.to_excel(output_file, index=False)
    print(f'所有文件已成功合并到: {output_file}')

# 使用示例
if __name__ == '__main__':
    # 指定要处理的文件夹路径
    folder_path = '/Users/Emins/desktop/Deep Learning/DIY-Pytorch-main/stock price prediction'
    # 指定输出文件的路径
    output_file = 'stock_data_merge.xlsx'
    
    merge_excel_files(folder_path, output_file) 