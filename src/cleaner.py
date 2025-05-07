# -*- coding: utf-8 -*-
"""
链家成都二手房数据清洗模块
负责对爬取的原始数据进行清洗、转换和特征工程
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime


class ChengduHousingCleaner:
    def __init__(self):
        """
        初始化数据清洗类
        """
        # 创建数据目录
        os.makedirs('data/processed', exist_ok=True)

    def load_data(self, csv_file):
        """
        从CSV文件加载原始数据

        参数:
            csv_file: CSV文件路径

        返回:
            df: 加载的数据框
        """
        try:
            # 使用pandas读取CSV文件
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"从 {csv_file} 加载了 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

    def clean_data(self, df):
        """
        清洗和预处理房源数据，专注于空间和结构特征

        参数:
            df: 原始数据框

        返回:
            df_clean: 清洗后的数据框
        """
        if df is None or df.empty:
            print("没有数据需要清洗")
            return None

        # 复制数据框，避免修改原始数据
        df_clean = df.copy()

        # 1. 清理列名
        df_clean.columns = [col.lower().strip() for col in df_clean.columns]

        # 2. 移除重复行
        initial_count = len(df_clean)
        df_clean.drop_duplicates(subset=['link'], keep='first', inplace=True)
        print(f"移除了 {initial_count - len(df_clean)} 条重复记录")

        # 3. 转换数值列
        # 3.1 每平方米价格转换为数值
        if 'price_per_sqm' in df_clean.columns:
            df_clean['price_per_sqm'] = df_clean['price_per_sqm'].astype(str).str.replace('元/平', '').str.replace(',',
                                                                                                                   '').astype(
                float)

        # 3.2 面积转换为数值
        if 'area' in df_clean.columns:
            df_clean['area'] = pd.to_numeric(df_clean['area'], errors='coerce')

        # 3.3 总价转换为数值
        if 'total_price' in df_clean.columns:
            df_clean['total_price'] = pd.to_numeric(df_clean['total_price'], errors='coerce')

        # 4. 从户型提取房间数量
        if 'layout' in df_clean.columns:
            # 提取卧室数量
            df_clean['bedrooms'] = df_clean['layout'].str.extract(r'(\d+)室').astype(float)
            # 提取客厅数量
            df_clean['living_rooms'] = df_clean['layout'].str.extract(r'(\d+)厅').astype(float)
            # 计算总房间数
            df_clean['total_rooms'] = df_clean['bedrooms'].fillna(0) + df_clean['living_rooms'].fillna(0)

        # 5. 处理缺失值
        # 对数值列，按区域分组，用中位数填充缺失值
        for col in ['price_per_sqm', 'area', 'total_price']:
            if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
                # 按区域分组填充缺失值
                df_clean[col] = df_clean.groupby('district')[col].transform(
                    lambda x: x.fillna(x.median())
                )

        # 6. 移除仍然缺少关键数据的行
        critical_columns = ['price_per_sqm', 'area']
        critical_columns = [col for col in critical_columns if col in df_clean.columns]
        if critical_columns:
            initial_count = len(df_clean)
            df_clean.dropna(subset=critical_columns, inplace=True)
            print(f"移除了 {initial_count - len(df_clean)} 行缺少关键数据的记录")

        # 7. 创建衍生特征
        # 7.1 价格分类(分位数)
        if 'price_per_sqm' in df_clean.columns:
            df_clean['price_category'] = pd.qcut(
                df_clean['price_per_sqm'],
                q=5,
                labels=['很低', '低', '中等', '高', '很高']
            )

        # 7.2 计算单位面积总房间数(空间利用效率指标)
        if 'total_rooms' in df_clean.columns and 'area' in df_clean.columns:
            df_clean['room_density'] = df_clean['total_rooms'] / df_clean['area']

        # 7.3 计算每房间价格(性价比指标)
        if 'total_price' in df_clean.columns and 'bedrooms' in df_clean.columns:
            # 确保不会除以零
            nonzero_bedrooms = df_clean['bedrooms'].replace(0, np.nan)
            df_clean['price_per_bedroom'] = df_clean['total_price'] / nonzero_bedrooms

        return df_clean

    def handle_outliers(self, df):
        """
        处理数据中的异常值，专注于价格和面积

        参数:
            df: 数据框

        返回:
            df_no_outliers: 处理后的数据框
        """
        if df is None or df.empty:
            return None

        df_no_outliers = df.copy()

        # 使用IQR方法处理价格异常值
        if 'price_per_sqm' in df_no_outliers.columns:
            Q1 = df_no_outliers['price_per_sqm'].quantile(0.25)
            Q3 = df_no_outliers['price_per_sqm'].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 标识异常值
            outliers_mask = (df_no_outliers['price_per_sqm'] < lower_bound) | (
                        df_no_outliers['price_per_sqm'] > upper_bound)
            outlier_count = outliers_mask.sum()

            print(f"检测到 {outlier_count} 个价格异常值 ({outlier_count / len(df_no_outliers) * 100:.1f}% 的数据)")

            # 过滤掉极端异常值
            df_no_outliers = df_no_outliers[~outliers_mask]

        # 使用IQR方法处理面积异常值
        if 'area' in df_no_outliers.columns:
            Q1 = df_no_outliers['area'].quantile(0.25)
            Q3 = df_no_outliers['area'].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 标识异常值
            outliers_mask = (df_no_outliers['area'] < lower_bound) | (df_no_outliers['area'] > upper_bound)
            outlier_count = outliers_mask.sum()

            print(f"检测到 {outlier_count} 个面积异常值 ({outlier_count / len(df_no_outliers) * 100:.1f}% 的数据)")

            # 过滤掉极端异常值
            df_no_outliers = df_no_outliers[~outliers_mask]

        return df_no_outliers

    def engineer_spatial_features(self, df):
        """
        创建空间分析所需的特征

        参数:
            df: 数据框

        返回:
            df_features: 添加了空间特征的数据框
        """
        if df is None or df.empty:
            return None

        df_features = df.copy()

        # 1. 计算区域价格统计特征
        if 'district' in df_features.columns and 'price_per_sqm' in df_features.columns:
            # 计算区域平均价格
            district_avg_price = df_features.groupby('district')['price_per_sqm'].mean()
            df_features['district_avg_price'] = df_features['district'].map(district_avg_price)

            # 计算相对于区域平均价格的溢价/折扣率
            df_features['price_premium_rate'] = (df_features['price_per_sqm'] - df_features['district_avg_price']) / \
                                                df_features['district_avg_price']

        # 2. 计算小区价格统计特征
        if 'community' in df_features.columns and 'price_per_sqm' in df_features.columns:
            # 计算小区平均价格(仅对样本量足够的小区)
            community_stats = df_features.groupby('community').agg(
                community_count=('price_per_sqm', 'count'),
                community_avg_price=('price_per_sqm', 'mean')
            )
            # 仅保留样本量大于2的小区
            valid_communities = community_stats[community_stats['community_count'] > 2]

            # 映射小区平均价格
            community_avg_dict = dict(zip(valid_communities.index, valid_communities['community_avg_price']))
            df_features['community_avg_price'] = df_features['community'].map(community_avg_dict)

            # 计算相对于小区平均价格的溢价/折扣率 - 处理可能的NaN值
            mask = df_features['community_avg_price'].notna()
            df_features.loc[mask, 'community_price_premium_rate'] = (
                    (df_features.loc[mask, 'price_per_sqm'] - df_features.loc[mask, 'community_avg_price']) /
                    df_features.loc[mask, 'community_avg_price']
            )

        # 3. 标准化区域内价格(Z-score)
        if 'district' in df_features.columns and 'price_per_sqm' in df_features.columns:
            # 安全计算Z-score
            def safe_zscore(x):
                if len(x) > 1 and x.std() > 0:
                    return (x - x.mean()) / x.std()
                else:
                    return pd.Series(0, index=x.index)

            df_features['price_z_score'] = df_features.groupby('district')['price_per_sqm'].transform(safe_zscore)

        # 4. 计算结构-空间组合特征
        # 4.1 区域内户型价格比较
        if 'district' in df_features.columns and 'layout' in df_features.columns and 'price_per_sqm' in df_features.columns:
            # 计算区域内特定户型的平均价格
            district_layout_avg = df_features.groupby(['district', 'layout'])['price_per_sqm'].mean().reset_index()
            district_layout_avg.columns = ['district', 'layout', 'district_layout_avg_price']

            # 合并回原数据框 - 使用pandas合并函数而非join
            df_features = pd.merge(
                df_features,
                district_layout_avg,
                on=['district', 'layout'],
                how='left'
            )

            # 计算相对于区域内同户型的溢价/折扣率
            df_features['layout_price_premium_rate'] = (
                    (df_features['price_per_sqm'] - df_features['district_layout_avg_price']) /
                    df_features['district_layout_avg_price']
            )

        return df_features

    def save_processed_data(self, df, output_file=None):
        """
        保存处理后的数据到CSV文件

        参数:
            df: 数据框
            output_file: 输出文件路径，如果为None则自动生成

        返回:
            output_file: 保存的文件路径
        """
        if df is None or df.empty:
            print("没有数据需要保存")
            return None

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"data/processed/chengdu_properties_spatial_{timestamp}.csv"

        # 确保目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存到CSV
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"已将 {len(df)} 条记录保存到 {output_file}")

        return output_file

    def process_data(self, input_file):
        """
        完整的数据处理流程

        参数:
            input_file: 输入CSV文件路径

        返回:
            (df_features, output_file): 处理后的数据框和保存的文件路径
        """
        print("=" * 50)
        print("开始数据清洗和预处理")
        print("=" * 50)

        # 1. 加载数据
        df = self.load_data(input_file)
        if df is None:
            return None, None

        # 2. 清洗数据
        print("\n执行基础数据清洗...")
        df_clean = self.clean_data(df)
        if df_clean is None:
            return None, None

        # 3. 处理异常值
        print("\n检测并移除异常值...")
        df_no_outliers = self.handle_outliers(df_clean)
        if df_no_outliers is None:
            return None, None

        # 4. 空间特征工程
        print("\n执行空间特征工程...")
        df_features = self.engineer_spatial_features(df_no_outliers)
        if df_features is None:
            return None, None

        # 5. 保存处理后的数据
        print("\n保存处理后的数据...")
        output_file = self.save_processed_data(df_features)

        print("\n数据清洗和预处理完成!")
        print(f"原始数据: {len(df)} 条记录")
        print(f"处理后数据: {len(df_features)} 条记录")
        print(f"清洗率: {len(df_features) / len(df) * 100:.1f}%")
        print(f"处理后数据已保存至: {output_file}")

        return df_features, output_file


# 如果直接运行此脚本，则处理指定文件
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 默认使用最新的数据文件
        data_files = [f for f in os.listdir("data/raw") if f.endswith('.csv')]
        if not data_files:
            print("未找到任何数据文件，请先运行爬虫收集数据")
            sys.exit(1)

        # 按修改时间排序，选择最新的文件
        input_file = os.path.join("data/raw", sorted(data_files)[-1])

    cleaner = ChengduHousingCleaner()
    cleaner.process_data(input_file)