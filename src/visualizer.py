# -*- coding: utf-8 -*-
"""
链家成都二手房数据可视化模块
负责对清洗后的数据进行可视化分析，同时支持中文显示
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
from adjustText import adjust_text
import matplotlib.font_manager as fm
import seaborn as sns
import textwrap

# 从plot_utils导入工具函数
from src.utils.plot_utils import get_optimized_colormap


class ChengduHousingVisualizer:
    def __init__(self, theme="light"):
        """
        初始化数据可视化类

        参数:
            theme: 主题，可选"light"或"dark"
        """
        # 初始化样式设置
        self.theme = theme

        # 配置中文字体支持
        self._setup_chinese_fonts()

        # 设置 DPI 和图表整体美观参数
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.autolayout'] = True

        # 设置Matplotlib全局样式
        if theme == "dark":
            plt.style.use('dark_background')
        else:
            plt.style.use('seaborn-v0_8-whitegrid')

        # 创建输出目录
        os.makedirs("data/visualizations", exist_ok=True)

        # 配置地理信息可视化参数
        self.geo_enabled = False
        self.geo_file = None
        self.geo_data = None

        # 设置最大展示区域数量
        self.max_districts_to_show = 15

        # 定义全局颜色方案
        self.color_schemes = {
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "accent": "#e74c3c",
            "neutral": "#95a5a6",
            "sequential": "viridis",
            "diverging": "coolwarm"
        }

    def _setup_chinese_fonts(self):
        """设置中文字体支持"""
        # 常见的中文字体列表
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi',
                         'FangSong', 'STXihei', 'STKaiti', 'STSong', 'STFangsong',
                         'STZhongsong', 'Source Han Sans CN', 'Source Han Serif CN',
                         'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
                         'Noto Serif CJK SC', 'Droid Sans Fallback']

        # 查找系统中可用的中文字体
        font_found = False
        for font_name in chinese_fonts:
            try:
                # 尝试找到字体
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if os.path.exists(font_path) and 'ttf' in font_path.lower():
                    print(f"找到可用的中文字体: {font_name} 路径: {font_path}")
                    # 设置为matplotlib默认字体
                    plt.rcParams['font.family'] = ['sans-serif']
                    plt.rcParams['font.sans-serif'] = [font_name, 'Arial', 'DejaVu Sans']
                    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
                    font_found = True
                    break
            except Exception as e:
                continue

        if not font_found:
            print("警告：未找到可用的中文字体，尝试使用内置方法...")
            # 如果找不到系统字体，尝试其他方法
            try:
                # 尝试使用matplotlib默认的中文支持
                plt.rcParams['font.family'] = ['sans-serif']
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Bitstream Vera Sans', 'Arial Unicode MS']
                plt.rcParams['axes.unicode_minus'] = False

                # 测试中文显示
                fig = plt.figure(figsize=(1, 1))
                plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
                plt.close(fig)
                print("内置中文字体设置成功")
            except Exception as e:
                print(f"中文字体设置失败: {e}")
                print("建议手动安装中文字体后重试")

    def wrap_labels(self, axis, width, break_long_words=False):
        """
        对坐标轴标签进行自动换行处理

        参数:
            axis: matplotlib 坐标轴对象
            width: 每行最大字符数
            break_long_words: 是否强制拆分长词
        """
        labels = []
        for item in axis.get_ticklabels():
            text = item.get_text()
            labels.append(textwrap.fill(text, width=width, break_long_words=break_long_words))
        axis.set_ticklabels(labels)
        return axis

    def enable_geo_visualization(self, geojson_file):
        """
        启用地理可视化功能

        参数:
            geojson_file: GeoJSON文件路径
        """
        self.load_geojson(geojson_file)

    def load_geojson(self, geojson_file):
        """
        加载GeoJSON文件用于地理可视化

        参数:
            geojson_file: GeoJSON文件路径
        """
        if os.path.exists(geojson_file):
            try:
                import json
                with open(geojson_file, 'r', encoding='utf-8') as f:
                    self.geo_data = json.load(f)
                self.geo_enabled = True
                self.geo_file = geojson_file
                print(f"加载了 {len(self.geo_data['features'])} 个地理坐标点")
            except Exception as e:
                print(f"加载GeoJSON文件失败: {e}")
                self.geo_enabled = False
        else:
            print(f"GeoJSON文件不存在: {geojson_file}")
            self.geo_enabled = False

    def optimize_district_display(self, district_data, max_districts=None):
        """
        优化区域数据的显示，对于过多的区域进行筛选或分组

        参数:
            district_data: 区域数据的DataFrame
            max_districts: 最大显示区域数，默认为None则使用实例默认值

        返回:
            处理后的区域数据
        """
        if max_districts is None:
            max_districts = self.max_districts_to_show

        # 如果区域数量过多，只显示最重要的一些区域
        if len(district_data) > max_districts:
            # 保留价格最高和最低的一些区域
            top_n = max(3, max_districts // 3)
            bottom_n = max(2, max_districts // 3)
            middle_n = max_districts - top_n - bottom_n

            # 高价区
            top_districts = district_data.head(top_n)

            # 低价区
            bottom_districts = district_data.tail(bottom_n)

            # 选择中间区域（等间隔采样）
            if middle_n > 0 and len(district_data) > (top_n + bottom_n):
                middle_indices = np.linspace(top_n, len(district_data) - bottom_n - 1, middle_n, dtype=int)
                middle_districts = district_data.iloc[middle_indices]

                # 合并数据
                selected_districts = pd.concat([top_districts, middle_districts, bottom_districts])
            else:
                selected_districts = pd.concat([top_districts, bottom_districts])

            return selected_districts

        return district_data

    def create_price_distribution_plots(self, df):
        """
        创建价格分布可视化

        参数:
            df: 数据框
        """
        if df is None or df.empty or 'price_per_sqm' not in df.columns:
            return

        print("创建价格分布可视化...")

        # 1. 价格直方图 - 改进版
        plt.figure(figsize=(10, 6))

        # 使用更少的bin以减少视觉混乱
        bins = min(30, int(np.sqrt(len(df['price_per_sqm'].dropna()))))

        plt.hist(df['price_per_sqm'].dropna(), bins=bins, alpha=0.7,
                 color=self.color_schemes["primary"],
                 edgecolor='white', linewidth=0.8)

        # 添加核密度估计
        if len(df) > 10:  # 确保数据点足够
            density = gaussian_kde(df['price_per_sqm'].dropna())
            x_vals = np.linspace(df['price_per_sqm'].min(), df['price_per_sqm'].max(), 100)
            plt.plot(x_vals, density(x_vals) * len(df) * (df['price_per_sqm'].max() - df['price_per_sqm'].min()) / bins,
                     'r-', linewidth=2)

        plt.title('成都二手房每平方米价格分布', fontsize=14, pad=20)
        plt.xlabel('每平方米价格 (元)', fontsize=12, labelpad=10)
        plt.ylabel('数量', fontsize=12, labelpad=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 使用科学计数法来优化显示大数值
        plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # 限制Y轴刻度数量
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        # 增加留白以改善可读性
        plt.tight_layout(pad=2.0)
        self.save_figure(plt.gcf(), "price_distribution_histogram", formats=["png"])
        plt.close()

        # 2. 按区域的价格箱线图 - 改进版
        if 'district' in df.columns:
            # 计算每个区域的样本量和中位数价格
            district_stats = df.groupby('district').agg(
                count=('price_per_sqm', 'count'),
                median=('price_per_sqm', 'median')
            ).reset_index()

            # 按中位数价格排序
            district_order = district_stats.sort_values('median', ascending=False)

            # 过滤掉样本量太少的区域
            valid_districts = district_order[district_order['count'] >= 3]

            # 限制显示的区域数量，避免图表过于拥挤
            valid_districts = self.optimize_district_display(valid_districts)
            district_order = valid_districts['district'].tolist()

            if len(district_order) > 0:
                # 根据区域数量调整图表尺寸
                fig_width = min(14, max(10, len(district_order) * 0.6))
                plt.figure(figsize=(fig_width, 8))

                # 准备每个区域的数据
                positions = range(len(district_order))
                boxplot_data = [df[df['district'] == d]['price_per_sqm'].dropna() for d in district_order]

                # 为各区域生成一组美观的颜色
                district_colors = get_optimized_colormap(len(district_order), self.color_schemes["sequential"])

                # 绘制箱线图，设置较窄的宽度并减小异常值点的大小
                boxplot = plt.boxplot(
                    boxplot_data,
                    patch_artist=True,
                    positions=positions,
                    widths=0.5,
                    flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=3, linestyle='none'),
                    medianprops=dict(linewidth=1.5, color='#c0392b')
                )

                # 美化箱线图
                for box, color in zip(boxplot['boxes'], district_colors):
                    box.set(facecolor=color, alpha=0.8, linewidth=0.8)

                # 设置为白色轮廓以增强可视性
                for element in ['whiskers', 'caps']:
                    for item in boxplot[element]:
                        item.set(color='#7f8c8d', linewidth=0.8)

                plt.title('成都各区域二手房价格分布', fontsize=14, pad=20)
                plt.xlabel('区域', fontsize=12, labelpad=15)
                plt.ylabel('每平方米价格 (元)', fontsize=12, labelpad=10)

                # 使用倾斜的标签以避免重叠
                plt.xticks(range(len(district_order)), district_order, rotation=45, ha='right')

                # 如果区域太多，强制只显示部分标签
                if len(district_order) > 10:
                    # 每隔几个显示一个标签
                    show_every_nth = max(1, len(district_order) // 10)
                    labels = plt.gca().xaxis.get_ticklabels()
                    for i, label in enumerate(labels):
                        if i % show_every_nth != 0:
                            label.set_visible(False)

                plt.grid(True, alpha=0.3, axis='y', linestyle='--')

                # 使用科学计数法格式化Y轴的大数值
                plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

                # 限制Y轴刻度数量
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                # 增加顶部和底部的空间以容纳标签
                plt.tight_layout(pad=2.0)

                # 保存图表
                self.save_figure(plt.gcf(), "price_by_district_boxplot", formats=["png"])
                plt.close()

        # 3. 使用Plotly创建交互式价格分布图 - 改进版
        fig = px.histogram(
            df,
            x='price_per_sqm',
            title='成都二手房价格交互式分布图',
            labels={'price_per_sqm': '每平方米价格 (元)'},
            opacity=0.7,
            nbins=30
        )

        # 更新布局，设置中文字体
        fig.update_layout(
            font_family="SimHei, Arial, sans-serif",  # 确保中文显示正常
            xaxis=dict(
                tickformat=',',
                title_standoff=15
            ),
            yaxis=dict(
                title='数量',
                title_standoff=15
            ),
            hoverlabel=dict(
                font_size=12,
                font_family="SimHei, Arial, sans-serif"
            )
        )

        # 添加平均值和中位数线
        mean_price = df['price_per_sqm'].mean()
        median_price = df['price_per_sqm'].median()

        fig.add_vline(
            x=mean_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"平均值: {mean_price:,.0f}",
            annotation_position="top right"
        )

        fig.add_vline(
            x=median_price,
            line_dash="dash",
            line_color="green",
            annotation_text=f"中位数: {median_price:,.0f}",
            annotation_position="top left"
        )

        # 保存交互式图表
        if not os.path.exists("data/visualizations"):
            os.makedirs("data/visualizations")
        fig.write_html("data/visualizations/price_distribution_interactive.html")

    def create_spatial_analysis_plots(self, df):
        """
        创建空间分析可视化

        参数:
            df: 数据框
        """
        if df is None or df.empty or 'district' not in df.columns or 'price_per_sqm' not in df.columns:
            return

        print("创建空间分析可视化...")

        # 1. 各区域平均房价条形图 - 改进版
        # 计算各区域的统计数据
        district_stats = df.groupby('district').agg(
            mean=('price_per_sqm', 'mean'),
            count=('price_per_sqm', 'count')
        ).reset_index()

        # 过滤掉样本量太少的区域
        district_stats = district_stats[district_stats['count'] >= 3]

        # 按均价排序
        district_stats = district_stats.sort_values('mean', ascending=False)

        # 限制显示数量，避免拥挤
        district_stats = self.optimize_district_display(district_stats)

        if len(district_stats) > 0:
            # 根据区域数量调整图表宽度
            fig_width = min(15, max(10, len(district_stats) * 0.5))
            plt.figure(figsize=(fig_width, 7))

            # 为各区域生成渐变色
            district_colors = get_optimized_colormap(len(district_stats), self.color_schemes["sequential"])

            # 创建条形图
            bars = plt.bar(
                range(len(district_stats)),
                district_stats['mean'],
                color=district_colors,
                edgecolor='white',
                linewidth=0.8,
                width=0.7  # 减小条形宽度以增加间隔
            )

            plt.title('成都各区域二手房平均价格', fontsize=14, pad=20)
            plt.xlabel('区域', fontsize=12, labelpad=15)
            plt.ylabel('平均每平方米价格 (元)', fontsize=12, labelpad=10)

            # 倾斜标签以避免重叠
            plt.xticks(range(len(district_stats)), district_stats['district'], rotation=45, ha='right')

            # 如果区域太多，强制只显示部分标签
            if len(district_stats) > 10:
                # 每隔几个显示一个标签
                show_every_nth = max(1, len(district_stats) // 10)
                labels = plt.gca().xaxis.get_ticklabels()
                for i, label in enumerate(labels):
                    if i % show_every_nth != 0:
                        label.set_visible(False)

            plt.grid(True, alpha=0.3, axis='y', linestyle='--')

            # 使用科学计数法格式化Y轴
            plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

            # 限制Y轴刻度数量
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

            # 添加价格标签，但避免拥挤
            if len(district_stats) <= 20:  # 只在条形数量合理时添加标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height):,}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=0
                    )

            plt.tight_layout(pad=2.0)
            self.save_figure(plt.gcf(), "avg_price_by_district", formats=["png"])
            plt.close()

        # 2. 创建交互式区域价格热力图
        fig = px.bar(
            district_stats,
            x='district',
            y='mean',
            color='mean',
            color_continuous_scale='Viridis',
            title='成都各区域房价热力图',
            labels={'district': '区域', 'mean': '平均每平方米价格 (元)'}
        )

        # 更新布局
        fig.update_layout(
            font_family="SimHei, Arial, sans-serif",  # 确保中文显示正常
            xaxis=dict(
                title='区域',
                tickangle=45
            ),
            yaxis=dict(
                title='平均每平方米价格 (元)',
                tickformat=','
            ),
            coloraxis_colorbar=dict(
                title='价格 (元/平方米)'
            )
        )

        # 保存交互式图表
        if not os.path.exists("data/visualizations"):
            os.makedirs("data/visualizations")
        fig.write_html("data/visualizations/district_price_heatmap_interactive.html")

    def create_structural_analysis_plots(self, df):
        """
        创建结构分析可视化

        参数:
            df: 数据框
        """
        if df is None or df.empty:
            return

        print("创建结构分析可视化...")

        # 1. 面积与价格散点图 - 改进版
        if 'area' in df.columns and 'price_per_sqm' in df.columns:
            area_price_df = df[['area', 'price_per_sqm']].dropna()

            if len(area_price_df) > 5:
                plt.figure(figsize=(10, 7))

                # 选择颜色
                if 'total_price' in df.columns:
                    # 使用总价作为颜色映射
                    scatter = plt.scatter(
                        area_price_df['area'],
                        area_price_df['price_per_sqm'],
                        c=df.loc[area_price_df.index, 'total_price'],
                        cmap='viridis',
                        alpha=0.7,
                        s=40,  # 减小点的大小以减少重叠
                        edgecolors='white',
                        linewidths=0.3
                    )
                    cbar = plt.colorbar(scatter, label='总价 (万元)')
                    # 美化颜色条
                    cbar.ax.tick_params(labelsize=9)
                    cbar.set_label('总价 (万元)', size=10, labelpad=10)
                else:
                    # 使用固定颜色
                    plt.scatter(
                        area_price_df['area'],
                        area_price_df['price_per_sqm'],
                        color=self.color_schemes["primary"],
                        alpha=0.7,
                        s=40,
                        edgecolors='white',
                        linewidths=0.3
                    )

                # 添加趋势线
                z = np.polyfit(area_price_df['area'], area_price_df['price_per_sqm'], 1)
                p = np.poly1d(z)
                x_range = np.sort(area_price_df['area'])
                plt.plot(
                    x_range,
                    p(x_range),
                    "r--",
                    linewidth=2,
                    label=f'趋势线: y = {z[0]:.1f}x + {z[1]:.1f}'
                )
                plt.legend(fontsize=10)

                plt.title('成都二手房面积与单价关系', fontsize=14, pad=20)
                plt.xlabel('面积 (平方米)', fontsize=12, labelpad=10)
                plt.ylabel('每平方米价格 (元)', fontsize=12, labelpad=10)
                plt.grid(True, alpha=0.3, linestyle='--')

                # 使用科学计数法格式化Y轴
                plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

                # 限制刻度数量
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                plt.tight_layout(pad=2.0)
                self.save_figure(plt.gcf(), "area_vs_price_scatter", formats=["png"])
                plt.close()

                # 使用Plotly创建交互式散点图 - 改进版
                fig = px.scatter(
                    df,
                    x='area',
                    y='price_per_sqm',
                    color='total_price' if 'total_price' in df.columns else None,
                    title='成都二手房面积与单价交互式关系图',
                    labels={
                        'area': '面积 (平方米)',
                        'price_per_sqm': '每平方米价格 (元)',
                        'total_price': '总价 (万元)'
                    },
                    hover_data=['district'] if 'district' in df.columns else None
                )

                # 添加趋势线
                fig.add_traces(
                    px.scatter(
                        x=x_range,
                        y=p(x_range),
                        labels={'x': '面积 (平方米)', 'y': '每平方米价格 (元)'},
                    ).update_traces(
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        showlegend=True,
                        name=f'趋势线: y = {z[0]:.1f}x + {z[1]:.1f}'
                    ).data
                )

                # 更新布局
                fig.update_layout(
                    font_family="SimHei, Arial, sans-serif",  # 确保中文显示正常
                    xaxis=dict(
                        title='面积 (平方米)',
                        tickangle=0
                    ),
                    yaxis=dict(
                        title='每平方米价格 (元)',
                        tickformat=','
                    )
                )

                # 保存交互式图表
                if not os.path.exists("data/visualizations"):
                    os.makedirs("data/visualizations")
                fig.write_html("data/visualizations/area_vs_price_interactive.html")

    def save_figure(self, fig, filename, formats=None):
        """
        保存图表到文件

        参数:
            fig: matplotlib图形对象
            filename: 文件名(不含扩展名)
            formats: 文件格式列表，如["png", "pdf"]
        """
        if formats is None:
            formats = ["png"]

        # 创建输出目录
        os.makedirs("data/visualizations", exist_ok=True)

        for fmt in formats:
            output_path = f"data/visualizations/{filename}.{fmt}"
            if isinstance(fig, plt.Figure):
                fig.savefig(output_path, bbox_inches='tight', dpi=150)
            else:  # Plotly figure
                if fmt == "html":
                    fig.write_html(output_path)
                elif fmt == "png":
                    fig.write_image(output_path)
                elif fmt == "json":
                    fig.write_json(output_path)

    def create_heatmap(self, df, value_column='price_per_sqm',
                       title="成都二手房价格热力图", colorscale="Viridis"):
        """
        创建热力图 - 地理可视化

        参数:
            df: 包含经纬度的数据框
            value_column: 用于确定热力值的列名
            title: 图表标题
            colorscale: 颜色映射
        """
        if not self.geo_enabled:
            print("警告: 未加载GeoJSON数据，无法创建热力图")
            return None

        # 检查数据框是否包含经纬度
        if 'lng' not in df.columns or 'lat' not in df.columns:
            print("错误: 数据框中没有经纬度列")
            return None

        # 过滤掉无效数据
        valid_df = df.dropna(subset=['lng', 'lat', value_column])

        if len(valid_df) == 0:
            print("错误: 没有有效的坐标和热力值数据")
            return None

        print("创建地理热力图...")

        # 创建基础地图
        fig = px.density_mapbox(
            valid_df,
            lat='lat',
            lon='lng',
            z=value_column,
            radius=10,
            center={"lat": valid_df['lat'].mean(), "lon": valid_df['lng'].mean()},
            zoom=10,
            mapbox_style="carto-positron",
            title=title,
            labels={value_column: '房价 (元/平方米)'}
        )

        # 更新布局
        fig.update_layout(
            font_family="SimHei, Arial, sans-serif",  # 确保中文显示正常
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(
                title='房价<br>(元/平方米)',
                tickfont=dict(family="SimHei, Arial, sans-serif")
            )
        )

        # 保存交互式图表
        if not os.path.exists("data/visualizations"):
            os.makedirs("data/visualizations")
        fig.write_html("data/visualizations/price_heatmap.html")

        return fig

    def create_dashboard(self, df, analysis_results=None):
        """
        创建综合数据面板可视化

        参数:
            df: 数据框
            analysis_results: 分析结果字典
        """
        if df is None or df.empty:
            return

        print("创建综合数据面板...")

        # 创建包含多个子图的数据面板
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '各区域价格分布',
                '面积与价格关系',
                '户型价格对比',
                '价格分布直方图'
            ),
            specs=[
                [{"type": "box"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.14,  # 增加垂直间距
            horizontal_spacing=0.10  # 增加水平间距
        )

        # 1. 各区域价格分布
        if 'district' in df.columns:
            # 计算每个区域的统计指标
            district_stats = df.groupby('district').agg(
                count=('price_per_sqm', 'count'),
                median=('price_per_sqm', 'median'),
                mean=('price_per_sqm', 'mean'),
                std=('price_per_sqm', 'std')
            ).reset_index()

            # 过滤掉样本量太少的区域
            district_stats = district_stats[district_stats['count'] >= 5]

            # 根据样本量和价格方差排序，找出最具代表性的区域
            district_stats['price_variation'] = district_stats['std'] / district_stats['mean']

            # 选择价格最高的2个区域
            high_price_districts = district_stats.sort_values('median', ascending=False).head(2)

            # 选择价格最低的2个区域
            low_price_districts = district_stats.sort_values('median', ascending=True).head(2)

            # 选择样本量最大的2个区域
            large_sample_districts = district_stats.sort_values('count', ascending=False).head(2)

            # 选择价格变异系数最大的2个区域(代表价格跨度最大的区域)
            high_variation_districts = district_stats.sort_values('price_variation', ascending=False).head(2)

            # 合并所有选中区域并去重
            selected_districts = pd.concat([
                high_price_districts,
                low_price_districts,
                large_sample_districts,
                high_variation_districts
            ]).drop_duplicates('district')

            # 如果选出的区域太多，限制数量
            if len(selected_districts) > 8:
                selected_districts = selected_districts.head(8)

            # 按中位数价格排序
            selected_districts = selected_districts.sort_values('median', ascending=False)

            # 对每个选中的区域添加箱线图
            colors = px.colors.qualitative.Plotly  # 使用Plotly的默认颜色方案
            for i, (_, row) in enumerate(selected_districts.iterrows()):
                district = row['district']
                district_data = df[df['district'] == district]['price_per_sqm'].dropna()

                if len(district_data) > 0:
                    color_idx = i % len(colors)
                    fig.add_trace(
                        go.Box(
                            y=district_data,
                            name=district,
                            marker_color=colors[color_idx],
                            boxmean=True  # 显示均值
                        ),
                        row=1, col=1
                    )

        # 2. 面积与价格散点图
        if 'area' in df.columns and 'price_per_sqm' in df.columns:
            # 抽样以避免点过多
            if len(df) > 500:
                scatter_df = df.sample(500, random_state=42)
            else:
                scatter_df = df

            scatter_data = scatter_df[['area', 'price_per_sqm']].dropna()

            if len(scatter_data) > 5:
                fig.add_trace(
                    go.Scatter(
                        x=scatter_data['area'],
                        y=scatter_data['price_per_sqm'],
                        mode='markers',
                        marker=dict(
                            size=7,
                            color=scatter_df['total_price'] if 'total_price' in scatter_df.columns else None,
                            colorscale='Viridis',
                            showscale=True if 'total_price' in scatter_df.columns else False,
                            colorbar=dict(title="总价 (万元)") if 'total_price' in scatter_df.columns else None
                        ),
                        name='房源'
                    ),
                    row=1, col=2
                )

                # 添加趋势线
                if len(scatter_data) > 2:
                    z = np.polyfit(scatter_data['area'], scatter_data['price_per_sqm'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(scatter_data['area'].min(), scatter_data['area'].max(), 100)

                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=p(x_range),
                            mode='lines',
                            line=dict(color='red', dash='dash'),
                            name=f'趋势线: y = {z[0]:.1f}x + {z[1]:.1f}'
                        ),
                        row=1, col=2
                    )

        # 3. 户型价格对比
        if 'layout' in df.columns and 'price_per_sqm' in df.columns:
            # 获取前6个最常见户型
            layout_counts = df['layout'].value_counts()
            top_layouts = layout_counts[layout_counts >= 3].head(6).index.tolist()

            if top_layouts:
                # 计算每种户型的平均价格
                layout_avg = df[df['layout'].isin(top_layouts)].groupby('layout')['price_per_sqm'].mean()
                layout_avg = layout_avg.sort_values(ascending=False)

                fig.add_trace(
                    go.Bar(
                        x=layout_avg.index,
                        y=layout_avg.values,
                        marker_color=px.colors.sequential.Viridis,
                        text=[f"{val:,.0f}" for val in layout_avg.values],
                        textposition='auto',
                        name='户型均价'
                    ),
                    row=2, col=1
                )

        # 4. 价格分布直方图
        price_data = df['price_per_sqm'].dropna()
        if len(price_data) > 0:
            fig.add_trace(
                go.Histogram(
                    x=price_data,
                    nbinsx=30,
                    marker_color='rgba(52, 152, 219, 0.7)',
                    name='价格分布'
                ),
                row=2, col=2
            )

            # 添加平均值和中位数线
            mean_price = price_data.mean()
            median_price = price_data.median()

            fig.add_vline(
                x=mean_price,
                line_dash="dash",
                line_color="red",
                row=2, col=2
            )

            fig.add_vline(
                x=median_price,
                line_dash="dash",
                line_color="green",
                row=2, col=2
            )

            # 添加标注
            fig.add_annotation(
                x=mean_price,
                y=0.9,
                text=f"均值: {mean_price:,.0f}",
                showarrow=False,
                font=dict(color="red"),
                xref="x4",
                yref="paper",
                row=2, col=2
            )

            fig.add_annotation(
                x=median_price,
                y=0.8,
                text=f"中位数: {median_price:,.0f}",
                showarrow=False,
                font=dict(color="green"),
                xref="x4",
                yref="paper",
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            title=dict(
                text='成都二手房市场分析面板',
                font=dict(family="SimHei, Arial, sans-serif", size=16),
                x=0.5
            ),
            height=800,
            width=1100,
            font=dict(family="SimHei, Arial, sans-serif", size=10),
            showlegend=False,
            margin=dict(l=60, r=50, t=100, b=60)
        )

        # 更新坐标轴标题
        fig.update_xaxes(title_text='区域', row=1, col=1, tickangle=45,
                         title_font=dict(family="SimHei, Arial, sans-serif"))
        fig.update_yaxes(title_text='每平方米价格 (元)', row=1, col=1, tickformat=',',
                         title_font=dict(family="SimHei, Arial, sans-serif"))

        fig.update_xaxes(title_text='面积 (平方米)', row=1, col=2, title_font=dict(family="SimHei, Arial, sans-serif"))
        fig.update_yaxes(title_text='每平方米价格 (元)', row=1, col=2, tickformat=',',
                         title_font=dict(family="SimHei, Arial, sans-serif"))

        fig.update_xaxes(title_text='户型', row=2, col=1, tickangle=45,
                         title_font=dict(family="SimHei, Arial, sans-serif"))
        fig.update_yaxes(title_text='平均每平方米价格 (元)', row=2, col=1, tickformat=',',
                         title_font=dict(family="SimHei, Arial, sans-serif"))

        fig.update_xaxes(title_text='每平方米价格 (元)', row=2, col=2, tickformat=',',
                         title_font=dict(family="SimHei, Arial, sans-serif"))
        fig.update_yaxes(title_text='数量', row=2, col=2, title_font=dict(family="SimHei, Arial, sans-serif"))

        # 保存交互式数据面板
        if not os.path.exists("data/visualizations"):
            os.makedirs("data/visualizations")
        fig.write_html("data/visualizations/chengdu_housing_dashboard.html")
        print("交互式数据面板已创建: data/visualizations/chengdu_housing_dashboard.html")

    def create_all_visualizations(self, df, analysis_results=None):
        """
        创建所有数据可视化

        参数:
            df: 数据框
            analysis_results: 分析结果字典（可选）
        """
        try:
            print("\n" + "=" * 50)
            print("开始创建可视化图表")
            print("=" * 50)

            # 创建价格分布可视化
            self.create_price_distribution_plots(df)

            # 创建空间分析可视化
            self.create_spatial_analysis_plots(df)

            # 创建结构分析可视化
            self.create_structural_analysis_plots(df)

            # 检查是否存在地理数据
            if self.geo_enabled and 'lat' in df.columns and 'lng' in df.columns:
                # 创建热力图
                self.create_heatmap(df)
            elif 'lat' in df.columns and 'lng' in df.columns:
                # 尝试启用地理可视化
                print("检测到经纬度数据，尝试创建地理可视化...")
                self.geo_enabled = True
                self.create_heatmap(df)

            # 创建综合数据面板
            self.create_dashboard(df, analysis_results)

            print("\n所有可视化已创建完成!")
            print(f"可视化文件保存在 data/visualizations/ 目录下")

        except Exception as e:
            print(f"创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
