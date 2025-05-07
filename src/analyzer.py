# -*- coding: utf-8 -*-
"""
链家成都二手房数据分析模块
负责对清洗后的数据进行空间结构分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score


class ChengduHousingAnalyzer:
    def __init__(self):
        """
        初始化数据分析类
        """
        # 创建输出目录
        os.makedirs("data/visualizations", exist_ok=True)

    def load_processed_data(self, csv_file):
        """
        从CSV文件加载处理后的数据

        参数:
            csv_file: CSV文件路径

        返回:
            df: 加载的数据框
        """
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"从 {csv_file} 加载了 {len(df)} 条处理后的记录")
            return df
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return None

    def basic_statistics(self, df):
        """
        计算关键指标的基本统计量

        参数:
            df: 数据框

        返回:
            stats_result: 统计结果字典
        """
        if df is None or df.empty:
            return None

        stats_result = {}

        # 每平方米价格统计
        if 'price_per_sqm' in df.columns:
            price_stats = df['price_per_sqm'].describe()
            stats_result['price_per_sqm'] = price_stats.to_dict()

            # 添加额外统计指标
            stats_result['price_per_sqm']['median'] = df['price_per_sqm'].median()
            stats_result['price_per_sqm']['skewness'] = float(df['price_per_sqm'].skew())  # 确保序列化兼容性
            stats_result['price_per_sqm']['kurtosis'] = float(df['price_per_sqm'].kurtosis())

        # 面积统计
        if 'area' in df.columns:
            area_stats = df['area'].describe()
            stats_result['area'] = area_stats.to_dict()

            # 添加额外统计指标
            stats_result['area']['median'] = df['area'].median()
            stats_result['area']['skewness'] = float(df['area'].skew())
            stats_result['area']['kurtosis'] = float(df['area'].kurtosis())

        # 区域级别价格统计
        if 'district' in df.columns and 'price_per_sqm' in df.columns:
            district_stats = df.groupby('district')['price_per_sqm'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).reset_index()

            # 转换为可序列化的字典
            stats_result['district_price_stats'] = district_stats.to_dict(orient='records')

        # 户型价格统计
        if 'layout' in df.columns and 'price_per_sqm' in df.columns:
            layout_stats = df.groupby('layout')['price_per_sqm'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).reset_index().sort_values('count', ascending=False).head(10)  # 仅取前10种最常见户型

            # 转换为可序列化的字典
            stats_result['layout_price_stats'] = layout_stats.to_dict(orient='records')

        return stats_result

    def analyze_spatial_distribution(self, df):
        """
        分析房价的空间分布特征

        参数:
            df: 数据框

        返回:
            spatial_results: 空间分析结果字典
        """
        if df is None or df.empty:
            return None

        if 'district' not in df.columns or 'price_per_sqm' not in df.columns:
            print("缺少必要的列(district, price_per_sqm)")
            return None

        spatial_results = {}

        # 1. 计算区域价格统计指标
        district_analysis = df.groupby('district').agg(
            sample_size=('price_per_sqm', 'count'),
            mean_price=('price_per_sqm', 'mean'),
            median_price=('price_per_sqm', 'median'),
            std_price=('price_per_sqm', 'std'),
            min_price=('price_per_sqm', 'min'),
            max_price=('price_per_sqm', 'max'),
            price_range=('price_per_sqm', lambda x: x.max() - x.min()),
            coefficient_of_variation=('price_per_sqm', lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
        ).reset_index()

        # 计算全市平均价
        city_avg_price = df['price_per_sqm'].mean()

        # 计算各区域相对于全市平均价的溢价率
        district_analysis['premium_rate'] = (district_analysis['mean_price'] - city_avg_price) / city_avg_price

        # 对区域进行分类（高价区、中价区、低价区）
        try:
            district_analysis['price_level'] = pd.qcut(
                district_analysis['mean_price'],
                q=3,
                labels=['低价区', '中价区', '高价区']
            )
        except ValueError:
            # 处理数据点不足导致的分位数错误
            district_analysis['price_level'] = "数据不足以分类"

        # 添加到结果字典
        spatial_results['district_analysis'] = district_analysis.to_dict(orient='records')
        spatial_results['city_avg_price'] = city_avg_price

        # 2. 分析小区价格空间特征
        if 'community' in df.columns:
            # 仅分析样本量足够的小区
            community_counts = df.groupby('community').size()
            valid_communities = community_counts[community_counts >= 3].index

            if len(valid_communities) > 0:
                community_df = df[df['community'].isin(valid_communities)]

                # 计算小区价格统计指标
                community_analysis = community_df.groupby(['district', 'community']).agg(
                    sample_size=('price_per_sqm', 'count'),
                    mean_price=('price_per_sqm', 'mean'),
                    median_price=('price_per_sqm', 'median'),
                    std_price=('price_per_sqm', 'std'),
                    price_range=('price_per_sqm', lambda x: x.max() - x.min())
                ).reset_index()

                # 找出每个区域中的最贵和最便宜小区
                # 按district分组后按mean_price排序
                community_analysis_sorted = community_analysis.sort_values(['district', 'mean_price'],
                                                                           ascending=[True, False])

                # 获取每个区域中最贵的3个小区
                top_communities = community_analysis_sorted.groupby('district').head(3).reset_index(drop=True)

                # 获取每个区域中最便宜的3个小区
                bottom_communities = community_analysis.sort_values(['district', 'mean_price'], ascending=[True, True])
                bottom_communities = bottom_communities.groupby('district').head(3).reset_index(drop=True)

                # 添加到结果字典
                spatial_results['top_communities'] = top_communities.to_dict(orient='records')
                spatial_results['bottom_communities'] = bottom_communities.to_dict(orient='records')

        # 3. 计算空间价格离散度
        if len(df['district'].unique()) > 1:
            # 组内离散度(各区域内价格的平均标准差)
            within_district_dispersion = df.groupby('district')['price_per_sqm'].std().mean()

            # 组间离散度(各区域均价的标准差)
            between_district_dispersion = df.groupby('district')['price_per_sqm'].mean().std()

            # 空间价格分化指数(组间/组内离散度比)
            spatial_disparity_index = between_district_dispersion / within_district_dispersion if within_district_dispersion > 0 else 0

            # 添加到结果字典
            spatial_results['within_district_dispersion'] = float(within_district_dispersion)
            spatial_results['between_district_dispersion'] = float(between_district_dispersion)
            spatial_results['spatial_disparity_index'] = float(spatial_disparity_index)

        return spatial_results

    def analyze_structural_economics(self, df):
        """
        分析结构特征与经济指标之间的关系

        参数:
            df: 数据框

        返回:
            structural_results: 结构分析结果字典
        """
        if df is None or df.empty:
            return None

        structural_results = {}

        # 1. 面积与价格关系分析
        if 'area' in df.columns and 'price_per_sqm' in df.columns:
            # 删除缺失值
            area_price_df = df.dropna(subset=['area', 'price_per_sqm'])

            # 确保数据足够进行分析
            if len(area_price_df) > 5:
                # 计算相关性
                correlation, p_value = stats.pearsonr(area_price_df['area'], area_price_df['price_per_sqm'])

                # 线性回归分析
                X = area_price_df['area'].values.reshape(-1, 1)
                y = area_price_df['price_per_sqm'].values

                model = LinearRegression()
                model.fit(X, y)

                # 记录结果
                structural_results['area_price_correlation'] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'regression_coefficient': float(model.coef_[0]),
                    'regression_intercept': float(model.intercept_),
                    'r_squared': float(model.score(X, y))
                }

                # 不同面积区间的价格分析
                area_bins = [0, 50, 70, 90, 120, 150, 200, float('inf')]
                area_labels = ['≤50㎡', '50-70㎡', '70-90㎡', '90-120㎡', '120-150㎡', '150-200㎡', '>200㎡']

                # 使用pd.cut进行分组
                df['area_category'] = pd.cut(df['area'], bins=area_bins, labels=area_labels)

                area_price_stats = df.groupby('area_category').agg(
                    count=('price_per_sqm', 'count'),
                    mean_price=('price_per_sqm', 'mean'),
                    median_price=('price_per_sqm', 'median'),
                    mean_total_price=('total_price', 'mean') if 'total_price' in df.columns else (
                    'price_per_sqm', 'count')
                ).reset_index()

                structural_results['area_category_stats'] = area_price_stats.to_dict(orient='records')

        # 2. 户型特征与价格关系分析
        if 'layout' in df.columns and 'price_per_sqm' in df.columns:
            # 提取最常见的户型
            top_layouts = df['layout'].value_counts().head(10).index.tolist()

            if top_layouts:
                layout_price_stats = df[df['layout'].isin(top_layouts)].groupby('layout').agg(
                    count=('price_per_sqm', 'count'),
                    mean_price=('price_per_sqm', 'mean'),
                    median_price=('price_per_sqm', 'median'),
                    mean_area=('area', 'mean') if 'area' in df.columns else ('price_per_sqm', 'count'),
                    mean_total_price=('total_price', 'mean') if 'total_price' in df.columns else (
                    'price_per_sqm', 'count')
                ).reset_index().sort_values('count', ascending=False)

                structural_results['layout_price_stats'] = layout_price_stats.to_dict(orient='records')

        # 3. 卧室数量与价格关系分析
        if 'bedrooms' in df.columns and 'price_per_sqm' in df.columns:
            # 删除缺失值
            bedroom_price_df = df.dropna(subset=['bedrooms', 'price_per_sqm'])

            # 获取出现频率较高的卧室数量
            bedroom_counts = bedroom_price_df['bedrooms'].value_counts()
            common_bedrooms = bedroom_counts[bedroom_counts >= 5].index.tolist()

            if common_bedrooms:
                bedroom_price_stats = bedroom_price_df[bedroom_price_df['bedrooms'].isin(common_bedrooms)].groupby(
                    'bedrooms').agg(
                    count=('price_per_sqm', 'count'),
                    mean_price=('price_per_sqm', 'mean'),
                    median_price=('price_per_sqm', 'median'),
                    mean_area=('area', 'mean') if 'area' in df.columns else ('price_per_sqm', 'count'),
                    mean_total_price=('total_price', 'mean') if 'total_price' in df.columns else (
                    'price_per_sqm', 'count')
                ).reset_index()

                # 计算卧室数量与单价的相关性
                if len(common_bedrooms) > 1:
                    corr_df = bedroom_price_df[bedroom_price_df['bedrooms'].isin(common_bedrooms)]
                    correlation, p_value = stats.pearsonr(corr_df['bedrooms'], corr_df['price_per_sqm'])

                    bedroom_price_analysis = {
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'statistics': bedroom_price_stats.to_dict(orient='records')
                    }

                    structural_results['bedroom_price_analysis'] = bedroom_price_analysis

        # 4. 空间利用效率分析
        if 'room_density' in df.columns and 'price_per_sqm' in df.columns:
            # 删除缺失值
            density_price_df = df.dropna(subset=['room_density', 'price_per_sqm'])

            if len(density_price_df) > 5:
                # 计算空间利用效率与价格的相关性
                correlation, p_value = stats.pearsonr(density_price_df['room_density'],
                                                      density_price_df['price_per_sqm'])

                # 分箱分析
                try:
                    df['density_category'] = pd.qcut(
                        df['room_density'].dropna(),
                        q=4,
                        labels=['低密度', '中低密度', '中高密度', '高密度']
                    )

                    density_price_stats = df.groupby('density_category').agg(
                        count=('price_per_sqm', 'count'),
                        mean_price=('price_per_sqm', 'mean'),
                        median_price=('price_per_sqm', 'median')
                    ).reset_index()

                    structural_results['density_price_analysis'] = {
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'statistics': density_price_stats.to_dict(orient='records')
                    }
                except ValueError:
                    # 处理数据点不足导致的分位数错误
                    print("数据不足以进行空间利用效率分箱分析")

        return structural_results

    def identify_market_segments(self, df):
        """
        识别市场细分，使用聚类算法

        参数:
            df: 数据框

        返回:
            segment_results: 市场细分结果字典
        """
        if df is None or df.empty:
            return None

        segment_results = {}

        # 确保必要的特征存在
        required_features = ['price_per_sqm', 'area']
        missing_features = [f for f in required_features if f not in df.columns]

        if missing_features:
            print(f"缺少聚类所需的特征: {missing_features}")
            return None

        # 准备聚类数据
        cluster_features = ['price_per_sqm', 'area']
        if 'bedrooms' in df.columns:
            cluster_features.append('bedrooms')

        # 删除缺失值
        cluster_df = df[cluster_features].dropna()

        # 判断数据量是否足够
        if len(cluster_df) < 10:
            print("数据量不足，无法进行有效聚类")
            return None

        # 标准化数据
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)

        # 确定最佳聚类数
        max_clusters = min(10, len(cluster_df) // 5)  # 最多10个簇，且每个簇至少有5个样本
        if max_clusters < 2:
            max_clusters = 2

        best_k = 2
        best_score = -1

        for k in range(2, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_data)

                # 计算轮廓系数
                score = silhouette_score(scaled_data, cluster_labels)

                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"计算k={k}的聚类时出错: {e}")
                continue

        # 使用最佳聚类数进行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        df_clustered = cluster_df.copy()
        df_clustered['cluster'] = kmeans.predict(scaled_data)

        # 分析每个簇的特征
        cluster_analysis = df_clustered.groupby('cluster').agg(
            size=('price_per_sqm', 'count'),
            avg_price=('price_per_sqm', 'mean'),
            avg_area=('area', 'mean')
        )

        if 'bedrooms' in cluster_features:
            cluster_analysis['avg_bedrooms'] = df_clustered.groupby('cluster')['bedrooms'].mean()

        # 计算每个簇在标准化尺度下的中心点
        centers = kmeans.cluster_centers_

        # 将聚类中心转换回原始尺度
        original_centers = scaler.inverse_transform(centers)

        # 创建聚类中心信息
        cluster_centers = []
        for i, center in enumerate(original_centers):
            center_info = {'cluster': i}
            for j, feature in enumerate(cluster_features):
                center_info[feature] = float(center[j])  # 确保数值可序列化
            cluster_centers.append(center_info)

        # 市场细分描述
        market_segments = []

        for i in range(best_k):
            segment = {}
            segment['segment_id'] = i
            segment['sample_size'] = int(cluster_analysis.loc[i, 'size'])
            segment['percentage'] = float(segment['sample_size'] / len(df_clustered) * 100)
            segment['avg_price'] = float(cluster_analysis.loc[i, 'avg_price'])
            segment['avg_area'] = float(cluster_analysis.loc[i, 'avg_area'])

            if 'bedrooms' in cluster_features:
                segment['avg_bedrooms'] = float(cluster_analysis.loc[i, 'avg_bedrooms'])

            # 确定细分市场定位
            price_level = ''
            if segment['avg_price'] > df['price_per_sqm'].quantile(0.75):
                price_level = '高价'
            elif segment['avg_price'] < df['price_per_sqm'].quantile(0.25):
                price_level = '低价'
            else:
                price_level = '中价'

            area_level = ''
            if segment['avg_area'] > df['area'].quantile(0.75):
                area_level = '大户型'
            elif segment['avg_area'] < df['area'].quantile(0.25):
                area_level = '小户型'
            else:
                area_level = '中等户型'

            segment['market_position'] = f"{price_level}{area_level}"

            # 检查该细分市场在哪些区域更集中
            if 'district' in df.columns:
                # 将聚类结果与原始数据合并
                # 首先重置索引以保留原始索引
                cluster_df_with_index = df_clustered.reset_index()
                # 创建索引到簇的映射
                index_to_cluster = dict(zip(cluster_df_with_index['index'], cluster_df_with_index['cluster']))

                # 对原始数据框添加索引列
                original_df = df.reset_index()
                # 使用映射将簇分配添加到原始数据
                original_df['cluster'] = original_df['index'].map(index_to_cluster)

                # 计算该簇在各区域的分布
                segment_district = original_df[original_df['cluster'] == i]['district'].value_counts()

                if len(segment_district) > 0:
                    segment_district_prop = segment_district / segment_district.sum()

                    # 全市区域分布
                    city_district = df['district'].value_counts()
                    city_district_prop = city_district / city_district.sum()

                    # 计算该簇在各区域的过度表示程度(LQ指数)
                    district_lq = {}
                    for district in segment_district_prop.index:
                        if district in city_district_prop.index:
                            lq = segment_district_prop[district] / city_district_prop[district]
                            district_lq[district] = float(lq)

                    # 找出最集中的区域(LQ>1.2)
                    concentrated_districts = [d for d, lq in district_lq.items() if lq > 1.2]
                    segment['concentrated_districts'] = concentrated_districts

            market_segments.append(segment)

        # 将结果添加到字典
        segment_results['best_k'] = best_k
        segment_results['silhouette_score'] = float(best_score)
        segment_results['cluster_centers'] = cluster_centers
        segment_results['market_segments'] = market_segments

        return segment_results

    def run_complete_analysis(self, csv_file):
        """
        运行完整的分析流程

        参数:
            csv_file: CSV文件路径

        返回:
            (results, df): 分析结果字典和数据框
        """
        print("=" * 50)
        print("开始数据分析")
        print("=" * 50)

        # 加载数据
        df = self.load_processed_data(csv_file)
        if df is None:
            return None, None

        # 运行基础统计分析
        print("\n执行基础统计分析...")
        basic_stats = self.basic_statistics(df)

        # 运行空间分布分析
        print("\n执行空间分布分析...")
        spatial_analysis = self.analyze_spatial_distribution(df)

        # 运行结构经济分析
        print("\n执行结构经济分析...")
        structural_analysis = self.analyze_structural_economics(df)

        # 运行市场细分分析
        print("\n执行市场细分分析...")
        market_segments = self.identify_market_segments(df)

        # 组合结果
        results = {
            'basic_statistics': basic_stats,
            'spatial_analysis': spatial_analysis,
            'structural_analysis': structural_analysis,
            'market_segments': market_segments,
            'data_summary': {
                'total_properties': len(df),
                'districts_covered': len(df['district'].unique()) if 'district' in df.columns else 0,
                'communities_covered': len(df['community'].unique()) if 'community' in df.columns else 0,
                'avg_price_per_sqm': float(df['price_per_sqm'].mean()) if 'price_per_sqm' in df.columns else None,
                'avg_area': float(df['area'].mean()) if 'area' in df.columns else None,
                'most_common_layout': df['layout'].value_counts().index[0] if 'layout' in df.columns and len(
                    df) > 0 else None
            }
        }

        # 保存结果
        timestamp = csv_file.split("_")[-1].split(".")[0]  # 从文件名提取时间戳
        output_dir = os.path.dirname(csv_file)
        output_file = f"{output_dir}/spatial_analysis_results_{timestamp}.json"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n分析完成，结果已保存到 {output_file}")

        # 打印主要分析结果摘要
        self._print_analysis_summary(results)

        return results, df

    def _print_analysis_summary(self, results):
        """
        打印主要分析结果摘要

        参数:
            results: 分析结果字典
        """
        print("\n" + "=" * 50)
        print("分析结果摘要")
        print("=" * 50)

        # 基本数据摘要
        if 'data_summary' in results:
            summary = results['data_summary']
            print(f"\n总房源数: {summary['total_properties']}")
            print(f"覆盖区域数: {summary['districts_covered']}")
            print(f"覆盖小区数: {summary['communities_covered']}")
            print(f"平均每平方米价格: {summary['avg_price_per_sqm']:.2f} 元")
            print(f"平均面积: {summary['avg_area']:.2f} 平方米")
            print(f"最常见户型: {summary['most_common_layout']}")

        # 空间分析摘要
        if 'spatial_analysis' in results and results['spatial_analysis']:
            print("\n空间分析结果:")
            if 'district_analysis' in results['spatial_analysis']:
                # 提取价格最高的三个区域
                districts = results['spatial_analysis']['district_analysis']
                sorted_districts = sorted(districts, key=lambda x: x['mean_price'], reverse=True)
                print("\n价格最高的三个区域:")
                for i, district in enumerate(sorted_districts[:3]):
                    print(f"  {i + 1}. {district['district']}: {district['mean_price']:.2f} 元/平方米")

            if 'spatial_disparity_index' in results['spatial_analysis']:
                sdi = results['spatial_analysis']['spatial_disparity_index']
                print(f"\n空间价格分化指数: {sdi:.2f}")
                if sdi > 1.5:
                    print("说明区域间价格差异明显大于区域内价格差异，空间分化程度高")
                elif sdi < 0.5:
                    print("说明区域内价格差异大于区域间价格差异，空间分化程度低")
                else:
                    print("说明区域间价格差异与区域内价格差异相当，空间分化程度中等")

        # 结构分析摘要
        if 'structural_analysis' in results and results['structural_analysis']:
            print("\n结构特征分析结果:")
            if 'area_price_correlation' in results['structural_analysis']:
                corr = results['structural_analysis']['area_price_correlation']
                print(f"面积与单价相关性: {corr['correlation']:.3f} (p值: {corr['p_value']:.3f})")
                if corr['correlation'] > 0:
                    print("面积越大，每平方米价格越高")
                else:
                    print("面积越大，每平方米价格越低")

            if 'bedroom_price_analysis' in results['structural_analysis']:
                bedroom_corr = results['structural_analysis']['bedroom_price_analysis']
                print(f"卧室数量与单价相关性: {bedroom_corr['correlation']:.3f} (p值: {bedroom_corr['p_value']:.3f})")

        # 市场细分摘要
        if 'market_segments' in results and results['market_segments']:
            print("\n市场细分分析结果:")
            print(f"最佳分类数量: {results['market_segments']['best_k']}")
            print(f"轮廓系数: {results['market_segments']['silhouette_score']:.3f}")

            print("\n细分市场特征:")
            for segment in results['market_segments']['market_segments']:
                segment_info = f"细分市场 {segment['segment_id']}: {segment['market_position']} "
                segment_info += f"(平均价格: {segment['avg_price']:.2f}元/㎡, 平均面积: {segment['avg_area']:.2f}㎡, "
                segment_info += f"占比: {segment['percentage']:.1f}%)"
                print(segment_info)

                if 'concentrated_districts' in segment and segment['concentrated_districts']:
                    print(f"  主要集中区域: {', '.join(segment['concentrated_districts'])}")


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        # 默认使用最新的处理后数据文件
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            print(f"目录 {processed_dir} 不存在，请先运行数据清洗")
            sys.exit(1)

        data_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not data_files:
            print("未找到任何处理后的数据文件，请先运行数据清洗")
            sys.exit(1)

        # 按修改时间排序，选择最新的文件
        input_file = os.path.join(processed_dir, sorted(data_files)[-1])

    analyzer = ChengduHousingAnalyzer()
    analyzer.run_complete_analysis(input_file)