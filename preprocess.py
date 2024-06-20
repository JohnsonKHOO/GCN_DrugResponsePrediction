import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def convert_ic50(value):
    try:
        return float(value)
    except ValueError:
        if '>' in value:
            return float(value.replace('>', '')) + 1  # 转换 '>50000' 为 50001
        elif '<' in value:
            return float(value.replace('<', '')) - 1  # 转换 '<50000' 为 49999
        else:
            return np.nan  # 其他无法转换的值设为 NaN


def showInfo(df):
    # 查看数据框的基本信息
    print("Basic Information of the DataFrame:")
    print(df.info())

    # 查看数据框的前几行
    print("\nFirst 5 Rows of the DataFrame:")
    print(df.head())

    # 查看描述性统计信息
    print("\nDescriptive Statistics of the DataFrame:")
    print(df.describe())

    # 检查某个列的唯一值和频率，例如 'mol_name'
    print("\nUnique Values and Frequencies in 'mol_name' Column:")
    print(df['mol_name'].value_counts())


def handling_outliers(df):
    print("\nIC50 (nM) column quantiles:")
    print(df['IC50 (nM)'].quantile([0.25, 0.5, 0.75, 0.95, 0.99]))

    # 处理异常值：例如，去掉高于99百分位数的值
    upper_limit = df['IC50 (nM)'].quantile(0.99)
    return df[df['IC50 (nM)'] <= upper_limit]


def main():
    df = pd.read_csv('./dataset/molecule_data_ori.csv')
    showInfo(df)

    df = df.dropna()
    # 查看 IC50 (nM) 列的分布
    df['IC50 (nM)'] = df['IC50 (nM)'].apply(convert_ic50)
    df = handling_outliers(df)

    df.to_csv('./dataset/preprocess/molecule_data_cleaned.csv', index=False)

    scaler = StandardScaler()
    df[['mol_weight', 'IC50 (nM)']] = scaler.fit_transform(df[['mol_weight', 'IC50 (nM)']])

    showInfo(df)

    # 保存标准化后的数据
    df.to_csv('./dataset/preprocess/molecule_data_scaled.csv', index=False)


if __name__ == '__main__':
    main()
