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
import textwrap
import matplotlib.font_manager as fm

# 导入自定义绘图工具
# from src.utils.plot_utils import PlotStyler, get_optimized_colormap
# 假设这里有对应的工具类，如果没有，我们会在下方定义


class PlotStyler:
    def __init__(self, theme="light"):
        self.theme = theme
        
    def save_figure(self, fig, filename, formats=["png"]):
        os.makedirs("data/visualizations", exist_ok=True)
        for fmt in formats:
            path = f"data/visualizations/{filename}.{fmt}"
            if isinstance(fig, plt.Figure):
                fig.savefig(path, bbox_inches='tight', dpi=150)
            else:  # Plotly figure
                if fmt == "html":
                    fig.write_html(path)
                elif fmt == "png":
                    fig.write_image(path)
                elif fmt == "json":
                    fig.write_json(path)
        
    def create_plotly_figure(self, title="", x_title="", y_title=""):
        fig = go.Figure()
        # 设置基础布局
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(family="Arial, SimHei", size=16),
                x=0.5
            ),
            xaxis=dict(
                title=dict(
                    text=x_title,
                    font=dict(family="Arial, SimHei", size=14)
                ),
                tickfont=dict(family="Arial, SimHei", size=12)
            ),
            yaxis=dict(
                title=dict(
                    text=y_title,
                    font=dict(family="Arial, SimHei", size=14)
                ),
                tickfont=dict(family="Arial, SimHei", size=12)
            ),
            template='plotly_white' if self.theme == 'light' else 'plotly_dark',
            font=dict(family="Arial, SimHei", size=12)
        )
        return fig


def get_optimized_colormap(n_colors, cmap_name):
    """生成优化的颜色映射"""
    cmap = plt.cm.get_cmap(cmap_name, n_colors)
    colors = [cmap(i) for i in range(n_colors)]
    return colors


class ChengduHousingVisualizer:
    def __init__(self, theme="light"):
        """
        初始化数据可视化类

        参数:
            theme: 主题，可选"light"或"dark"
        """
        # 初始化样式处理器
        self.styler = PlotStyler(theme=theme)

        # ====== 修复中文显示问题 ======
        # 查找系统中的中文字体
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
        self.styler.save_figure(plt.gcf(), "price_distribution_histogram", formats=["png"])
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
                self.styler.save_figure(plt.gcf(), "price_by_district_boxplot", formats=["png"])
                plt.close()

        # 3. 使用Plotly创建交互式价格分布图 - 改进版
        fig = self.styler.create_plotly_figure(
            title='成都二手房价格交互式分布图',
            x_title='每平方米价格 (元)',
            y_title='数量'
        )

        # 添加直方图
        fig.add_trace(
            go.Histogram(
                x=df['price_per_sqm'],
                name='价格分布',
                marker_color=self.color_schemes["primary"],
                opacity=0.7,
                nbinsx=30,
                hovertemplate='价格范围: %{x:,.0f}元/㎡<br>数量: %{y}<extra></extra>'
            )
        )

        # 添加箱线图
        fig.add_trace(
            go.Box(
                x=df['price_per_sqm'],
                name='价格分布',
                marker_color='indianred',
                boxmean=True,  # 显示均值
                orientation='h',
                line=dict(width=1.5),
                hovertemplate='最小值: %{min:,.0f}元/㎡<br>第一四分位: %{q1:,.0f}元/㎡<br>中位数: %{median:,.0f}元/㎡<br>第三四分位: %{q3:,.0f}元/㎡<br>最大值: %{max:,.0f}元/㎡<extra></extra>'
            )
        )

        # 格式化刻度，避免过密
        fig.update_layout(
            xaxis=dict(
                tickformat=',',
                title_standoff=15
            ),
            yaxis=dict(
                title_standoff=15
            ),
            hoverlabel=dict(
                font_size=12,
                font_family="Arial, SimHei"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 保存交互式图表
        self.styler.save_figure(fig, "price_distribution_interactive", formats=["html"])

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
            self.styler.save_figure(plt.gcf(), "avg_price_by_district", formats=["png"])
            plt.close()

        # 2. 区域价格变异系数比较 - 改进版
        district_cv = df.groupby('district').agg(
            mean=('price_per_sqm', 'mean'),
            std=('price_per_sqm', 'std'),
            count=('price_per_sqm', 'count')
        )

        # 计算变异系数
        district_cv['cv'] = district_cv['std'] / district_cv['mean']

        # 过滤掉样本量过少的区域
        district_cv = district_cv[district_cv['count'] >= 5]

        # 按变异系数排序并重置索引
        district_cv = district_cv.sort_values('cv', ascending=False).reset_index()

        # 限制显示数量，避免拥挤
        district_cv = self.optimize_district_display(district_cv)

        if len(district_cv) > 0:
            # 根据区域数量调整图表宽度
            fig_width = min(15, max(10, len(district_cv) * 0.5))
            plt.figure(figsize=(fig_width, 7))

            # 为各区域生成渐变色
            cv_colors = get_optimized_colormap(len(district_cv), 'plasma')

            bars = plt.bar(
                range(len(district_cv)),
                district_cv['cv'],
                color=cv_colors,
                edgecolor='white',
                linewidth=0.8,
                width=0.7  # 减小条形宽度以增加间隔
            )

            plt.title('成都各区域房价内部变异系数对比', fontsize=14, pad=20)
            plt.xlabel('区域', fontsize=12, labelpad=15)
            plt.ylabel('变异系数 (标准差/均值)', fontsize=12, labelpad=10)

            # 倾斜标签以避免重叠
            plt.xticks(range(len(district_cv)), district_cv['district'], rotation=45, ha='right')

            # 如果区域太多，强制只显示部分标签
            if len(district_cv) > 10:
                # 每隔几个显示一个标签
                show_every_nth = max(1, len(district_cv) // 10)
                labels = plt.gca().xaxis.get_ticklabels()
                for i, label in enumerate(labels):
                    if i % show_every_nth != 0:
                        label.set_visible(False)

            plt.grid(True, alpha=0.3, axis='y', linestyle='--')

            # 限制Y轴刻度数量
            plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

            # 添加数值标签，但避免拥挤
            if len(district_cv) <= 20:  # 只在条形数量合理时添加标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{height:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=8,
                        rotation=0
                    )

            plt.tight_layout(pad=2.0)
            self.styler.save_figure(plt.gcf(), "district_price_variation", formats=["png"])
            plt.close()

        # 3. 小区均价前10名条形图 - 改进版
        if 'community' in df.columns:
            # 计算每个小区的样本量和均价
            community_stats = df.groupby('community').agg(
                mean=('price_per_sqm', 'mean'),
                count=('price_per_sqm', 'count')
            ).reset_index()

            # 过滤掉样本量太少的小区
            valid_communities = community_stats[community_stats['count'] >= 3]

            # 获取前10名最贵小区
            top_communities = valid_communities.sort_values('mean', ascending=False).head(10)

            if len(top_communities) > 0:
                # 对小区名称进行处理，以免太长
                top_communities['short_name'] = top_communities['community'].apply(
                    lambda x: x if len(x) < 12 else x[:10] + '...')

                plt.figure(figsize=(12, 7))

                # 为各小区生成渐变色
                community_colors = get_optimized_colormap(len(top_communities), 'viridis')

                bars = plt.bar(
                    range(len(top_communities)),
                    top_communities['mean'],
                    color=community_colors,
                    edgecolor='white',
                    linewidth=0.8,
                    width=0.7
                )

                plt.title('成都二手房价格最高的10个小区', fontsize=14, pad=20)
                plt.xlabel('小区', fontsize=12, labelpad=15)
                plt.ylabel('平均每平方米价格 (元)', fontsize=12, labelpad=10)
                plt.xticks(range(len(top_communities)), top_communities['short_name'], rotation=30, ha='right')
                plt.grid(True, alpha=0.3, axis='y', linestyle='--')

                # 使用科学计数法格式化Y轴
                plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

                # 限制Y轴刻度数量
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                # 添加价格标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height):,}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )

                plt.tight_layout(pad=2.0)
                self.styler.save_figure(plt.gcf(), "top_price_communities", formats=["png"])
                plt.close()

        # 4. 使用Plotly创建交互式区域价格热力图 - 改进版
        district_pivot = df.groupby('district').agg(
            mean=('price_per_sqm', 'mean'),
            median=('price_per_sqm', 'median'),
            std=('price_per_sqm', 'std'),
            count=('price_per_sqm', 'count')
        ).reset_index()

        # 计算变异系数
        district_pivot['cv'] = district_pivot['std'] / district_pivot['mean']

        # 过滤掉样本量太少的区域
        district_pivot = district_pivot[district_pivot['count'] >= 3]

        # 按平均价格排序
        district_pivot = district_pivot.sort_values('mean', ascending=False)

        # 限制显示区域数量
        if len(district_pivot) > 20:
            district_pivot = district_pivot.head(20)

        if len(district_pivot) > 0:
            # 创建热力图
            fig = self.styler.create_plotly_figure(
                title="成都各区域房价指标对比",
                x_title="区域",
                y_title="指标类型"
            )

            # 准备热力图数据
            z_data = [
                district_pivot['mean'].tolist(),
                district_pivot['median'].tolist(),
                district_pivot['std'].tolist(),
                district_pivot['cv'].tolist()
            ]

            y_labels = ['平均价格', '中位数价格', '标准差', '变异系数']

            # 创建热力图
            fig.add_trace(go.Heatmap(
                z=z_data,
                x=district_pivot['district'],
                y=y_labels,
                colorscale='Viridis',
                text=[[f"{val:,.0f}" for val in district_pivot['mean']],
                      [f"{val:,.0f}" for val in district_pivot['median']],
                      [f"{val:,.0f}" for val in district_pivot['std']],
                      [f"{val:.2f}" for val in district_pivot['cv']]],
                texttemplate="%{text}",
                textfont={"size": 11, "family": "Arial, SimHei"},
                hovertemplate="区域: %{x}<br>指标: %{y}<br>值: %{text}<extra></extra>"
            ))

            # 更新布局
            fig.update_layout(
                height=600,
                xaxis=dict(
                    tickangle=45,
                    title_standoff=30
                ),
                yaxis=dict(
                    title_standoff=20
                ),
                margin=dict(l=60, r=50, t=100, b=100)
            )

            # 保存交互式图表
            self.styler.save_figure(fig, "district_price_metrics_interactive", formats=["html"])
            
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
                self.styler.save_figure(plt.gcf(), "area_vs_price_scatter", formats=["png"])
                plt.close()

                # 使用Plotly创建交互式散点图 - 改进版
                fig = self.styler.create_plotly_figure(
                    title='成都二手房面积与单价交互式关系图',
                    x_title='面积 (平方米)',
                    y_title='每平方米价格 (元)'
                )

                # 准备悬停数据
                hover_data = {
                    'price_per_sqm': True,
                    'area': True
                }

                if 'district' in df.columns:
                    hover_data['district'] = True

                if 'layout' in df.columns:
                    hover_data['layout'] = True

                if 'total_price' in df.columns:
                    hover_data['total_price'] = True

                # 添加散点图数据
                fig.add_trace(go.Scatter(
                    x=area_price_df['area'],
                    y=area_price_df['price_per_sqm'],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=df.loc[area_price_df.index, 'total_price'] if 'total_price' in df.columns else
                        self.color_schemes["primary"],
                        colorscale='Viridis',
                        showscale=True if 'total_price' in df.columns else False,
                        colorbar=dict(title="总价 (万元)") if 'total_price' in df.columns else None,
                        opacity=0.7
                    ),
                    name='房源',
                    text=[f"区域: {d}<br>户型: {l}<br>总价: {p}万<br>单价: {s:,.0f}元/㎡<br>面积: {a:.1f}㎡"
                          for d, l, p, s, a in zip(
                            df.loc[area_price_df.index, 'district'] if 'district' in df.columns else [''] * len(
                                area_price_df),
                            df.loc[area_price_df.index, 'layout'] if 'layout' in df.columns else [''] * len(
                                area_price_df),
                            df.loc[area_price_df.index, 'total_price'] if 'total_price' in df.columns else [''] * len(
                                area_price_df),
                            area_price_df['price_per_sqm'],
                            area_price_df['area']
                        )],
                    hoverinfo='text'
                ))

                # 添加趋势线
                x_range = np.linspace(min(area_price_df['area']), max(area_price_df['area']), 100)
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=p(x_range),
                    mode='lines',
                    name=f'趋势线: y = {z[0]:.1f}x + {z[1]:.1f}',
                    line=dict(color='red', width=2, dash='dash')
                ))

                # 格式化刻度
                fig.update_layout(
                    xaxis=dict(
                        tickmode='auto',
                        nticks=10,
                        title_standoff=20
                    ),
                    yaxis=dict(
                        tickformat=',',
                        title_standoff=20
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=60, r=50, t=80, b=60)
                )

                # 保存交互式图表
                self.styler.save_figure(fig, "area_vs_price_interactive", formats=["html"])

        # 2. 户型价格对比图 - 改进版
        if 'layout' in df.columns and 'price_per_sqm' in df.columns:
            # 计算每个户型的样本量和中位数价格
            layout_stats = df.groupby('layout').agg(
                count=('price_per_sqm', 'count'),
                median=('price_per_sqm', 'median')
            ).reset_index()

            # 获取前10个最常见户型
            top_layouts = layout_stats[layout_stats['count'] >= 3].sort_values('count', ascending=False).head(10)[
                'layout'].tolist()

            if top_layouts:
                # 按照中位数价格排序
                layout_order = df[df['layout'].isin(top_layouts)].groupby('layout')[
                    'price_per_sqm'].median().sort_values(ascending=False).index.tolist()

                plt.figure(figsize=(12, 7))

                # 为各户型生成渐变色
                layout_colors = get_optimized_colormap(len(layout_order), 'plasma')

                # 准备每个户型的数据
                positions = range(len(layout_order))
                boxplot_data = [df[df['layout'] == d]['price_per_sqm'].dropna() for d in layout_order]

                # 绘制箱线图，减小异常值点的大小
                boxplot = plt.boxplot(
                    boxplot_data,
                    patch_artist=True,
                    positions=positions,
                    widths=0.6,
                    flierprops=dict(marker='o', markerfacecolor='#95a5a6', markersize=3, linestyle='none'),
                    medianprops=dict(linewidth=1.5, color='#c0392b')
                )

                # 美化箱线图
                for box, color in zip(boxplot['boxes'], layout_colors):
                    box.set(facecolor=color, alpha=0.8, linewidth=0.8)

                # 设置为白色轮廓以增强可视性
                for element in ['whiskers', 'caps']:
                    for item in boxplot[element]:
                        item.set(color='#7f8c8d', linewidth=0.8)

                plt.title('成都二手房常见户型价格对比', fontsize=14, pad=20)
                plt.xlabel('户型', fontsize=12, labelpad=15)
                plt.ylabel('每平方米价格 (元)', fontsize=12, labelpad=10)
                plt.xticks(range(len(layout_order)), layout_order, rotation=30, ha='right')
                plt.grid(True, alpha=0.3, axis='y', linestyle='--')

                # 使用科学计数法格式化Y轴
                plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

                # 限制Y轴刻度数量
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                plt.tight_layout(pad=2.0)
                self.styler.save_figure(plt.gcf(), "layout_price_comparison", formats=["png"])
                plt.close()

                # 创建交互式箱线图
                fig = self.styler.create_plotly_figure(
                    title='成都二手房常见户型价格对比(交互式)',
                    x_title='户型',
                    y_title='每平方米价格 (元)'
                )

                # 添加箱线图数据
                for i, layout in enumerate(layout_order):
                    # 转换颜色格式为 rgba
                    rgba_color = layout_colors[i]
                    plotly_color = f'rgba({int(rgba_color[0] * 255)},{int(rgba_color[1] * 255)},' \
                                   f'{int(rgba_color[2] * 255)},{rgba_color[3]})'

                    fig.add_trace(go.Box(
                        y=df[df['layout'] == layout]['price_per_sqm'].dropna(),
                        name=layout,
                        marker_color=plotly_color,
                        boxmean=True,  # 显示均值
                        line=dict(width=1),
                        hovertemplate='户型: %{x}<br>价格: %{y:,.0f}元/㎡<extra></extra>'
                    ))

                # 格式化刻度
                fig.update_layout(
                    xaxis=dict(
                        title_standoff=20
                    ),
                    yaxis=dict(
                        tickformat=',',
                        title_standoff=20
                    ),
                    boxmode='group',
                    margin=dict(l=60, r=50, t=80, b=80)
                )

                # 保存交互式图表
                self.styler.save_figure(fig, "layout_price_comparison_interactive", formats=["html"])
                
        # 3. 卧室数量与价格关系图 - 改进版
        if 'bedrooms' in df.columns and 'price_per_sqm' in df.columns:
            # 过滤掉缺失值
            bedroom_df = df.dropna(subset=['bedrooms', 'price_per_sqm'])

            # 计算每个卧室数的样本量
            bedroom_counts = bedroom_df['bedrooms'].value_counts()

            # 过滤掉极端值，只保留样本量>=5的卧室数
            valid_bedrooms = bedroom_counts[bedroom_counts >= 5].index.tolist()

            if valid_bedrooms:
                # 计算每个卧室数的平均价格
                bedroom_avg = bedroom_df[bedroom_df['bedrooms'].isin(valid_bedrooms)].groupby('bedrooms')[
                    'price_per_sqm'].mean()

                # 确保卧室数是升序的
                bedroom_avg = bedroom_avg.sort_index()

                plt.figure(figsize=(10, 7))

                # 为各卧室数生成渐变色
                bedroom_colors = get_optimized_colormap(len(bedroom_avg), 'viridis')

                bars = plt.bar(
                    bedroom_avg.index,
                    bedroom_avg.values,
                    color=bedroom_colors,
                    edgecolor='white',
                    linewidth=0.8,
                    width=0.7
                )

                plt.title('成都二手房卧室数量与平均单价关系', fontsize=14, pad=20)
                plt.xlabel('卧室数量', fontsize=12, labelpad=10)
                plt.ylabel('平均每平方米价格 (元)', fontsize=12, labelpad=10)
                plt.grid(True, alpha=0.3, axis='y', linestyle='--')

                # 使用科学计数法格式化Y轴
                plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

                # 确保X轴是整数
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

                # 限制Y轴刻度数量
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

                # 添加价格标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height,
                        f'{int(height):,}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        rotation=0
                    )

                plt.tight_layout(pad=2.0)
                self.styler.save_figure(plt.gcf(), "bedrooms_vs_price", formats=["png"])
                plt.close()

                # 创建交互式图表 - 改进版
                bedroom_stats = bedroom_df[bedroom_df['bedrooms'].isin(valid_bedrooms)].groupby('bedrooms').agg(
                    mean_price=('price_per_sqm', 'mean'),
                    median_price=('price_per_sqm', 'median'),
                    mean_area=('area', 'mean') if 'area' in df.columns else ('price_per_sqm', 'count'),
                    mean_total=('total_price', 'mean') if 'total_price' in df.columns else ('price_per_sqm', 'count'),
                    count=('price_per_sqm', 'count')
                ).reset_index().sort_values('bedrooms')

                fig = self.styler.create_plotly_figure(
                    title="成都二手房卧室数量与价格和面积的关系",
                    x_title="卧室数量",
                    y_title="每平方米价格 (元)"
                )

                # 添加条形图数据
                fig.add_trace(
                    go.Bar(
                        x=bedroom_stats['bedrooms'],
                        y=bedroom_stats['mean_price'],
                        name="平均单价 (元/㎡)",
                        marker_color=self.color_schemes["primary"],
                        text=[f"{price:,.0f}" for price in bedroom_stats['mean_price']],
                        textposition='auto',
                        hovertemplate='卧室数: %{x}<br>平均单价: %{text}元/㎡<br>样本数: %{customdata}套<extra></extra>',
                        customdata=bedroom_stats['count']
                    )
                )

                if 'area' in df.columns:
                    # 添加第二个Y轴
                    fig.update_layout(
                        yaxis2=dict(
                            title="平均面积 (平方米)",
                            titlefont=dict(color="#ff7f0e"),
                            tickfont=dict(color="#ff7f0e"),
                            anchor="x",
                            overlaying="y",
                            side="right"
                        )
                    )

                    # 添加面积线图
                    fig.add_trace(
                        go.Scatter(
                            x=bedroom_stats['bedrooms'],
                            y=bedroom_stats['mean_area'],
                            name="平均面积 (㎡)",
                            mode='lines+markers',
                            line=dict(color='#ff7f0e', width=3),
                            marker=dict(size=10),
                            yaxis="y2",
                            hovertemplate='卧室数: %{x}<br>平均面积: %{y:.1f}㎡<extra></extra>'
                        )
                    )

                # 格式化刻度
                fig.update_layout(
                    xaxis=dict(
                        tickmode='array',
                        tickvals=bedroom_stats['bedrooms'],
                        title_standoff=20
                    ),
                    yaxis=dict(
                        tickformat=',',
                        title_standoff=20
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    margin=dict(l=60, r=60, t=80, b=60)
                )

                # 保存交互式图表
                self.styler.save_figure(fig, "bedrooms_price_area_interactive", formats=["html"])
                
        # 4. 面积分布直方图 - 改进版
        if 'area' in df.columns:
            area_data = df['area'].dropna()

            if len(area_data) > 5:
                plt.figure(figsize=(10, 7))

                # 使用更少的bin以减少视觉混乱
                bins = min(30, int(np.sqrt(len(area_data))))

                plt.hist(area_data, bins=bins, alpha=0.7,
                         color=self.color_schemes["secondary"],
                         edgecolor='white', linewidth=0.8)

                # 添加核密度估计
                density = gaussian_kde(area_data)
                x_vals = np.linspace(area_data.min(), area_data.max(), 100)
                plt.plot(
                    x_vals,
                    density(x_vals) * len(area_data) * (area_data.max() - area_data.min()) / bins,
                    'r-',
                    linewidth=2
                )

                plt.title('成都二手房面积分布', fontsize=14, pad=20)
                plt.xlabel('面积 (平方米)', fontsize=12, labelpad=10)
                plt.ylabel('数量', fontsize=12, labelpad=10)
                plt.grid(True, alpha=0.3, linestyle='--')

                # 限制X轴和Y轴刻度数量
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
                plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

                plt.tight_layout(pad=2.0)
                self.styler.save_figure(plt.gcf(), "area_distribution", formats=["png"])
                plt.close()

                # 创建交互式面积分布图
                fig = self.styler.create_plotly_figure(
                    title='成都二手房面积分布',
                    x_title='面积 (平方米)',
                    y_title='数量'
                )

                # 添加直方图
                fig.add_trace(go.Histogram(
                    x=area_data,
                    nbinsx=30,
                    marker_color=self.color_schemes["secondary"],
                    opacity=0.7,
                    name='面积分布',
                    hovertemplate='面积范围: %{x:.1f}㎡<br>数量: %{y}<extra></extra>'
                ))

                # 格式化刻度
                fig.update_layout(
                    xaxis=dict(
                        title_standoff=20
                    ),
                    yaxis=dict(
                        title_standoff=20
                    ),
                    bargap=0.05,
                    margin=dict(l=60, r=50, t=80, b=60)
                )

                # 保存交互式图表
                self.styler.save_figure(fig, "area_distribution_interactive", formats=["html"])

    def create_market_segment_plots(self, df, segment_results):
        """
        创建市场细分可视化

        参数:
            df: 数据框
            segment_results: 市场细分结果字典
        """
        if df is None or df.empty or segment_results is None:
            return

        if 'market_segments' not in segment_results:
            return

        print("创建市场细分可视化...")

        # 准备聚类数据
        cluster_features = ['price_per_sqm', 'area']
        if 'bedrooms' in df.columns:
            cluster_features.append('bedrooms')

        # 删除缺失值
        cluster_df = df[cluster_features].dropna()

        if len(cluster_df) < 10:
            print("数据不足，无法创建市场细分可视化")
            return

        # 添加聚类结果
        n_clusters = segment_results['best_k']
        centers = []

        for center_info in segment_results['cluster_centers']:
            center = []
            for feature in cluster_features:
                center.append(center_info[feature])
            centers.append(center)
        centers = np.array(centers)

        # 使用KMeans聚类
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_df)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        # 设置聚类中心
        transformed_centers = scaler.transform(centers)
        kmeans.cluster_centers_ = transformed_centers
        cluster_labels = kmeans.predict(scaled_data)

        # 将聚类结果合并回原始数据
        df_clustered = cluster_df.copy()
        df_clustered['cluster'] = cluster_labels

        # 1. 散点图展示市场细分 - 改进版
        plt.figure(figsize=(10, 8))

        # 为每个簇选择不同的颜色
        cluster_colors = get_optimized_colormap(n_clusters, 'tab10')

        # 绘制散点图
        for i in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == i]
            plt.scatter(
                cluster_data['area'],
                cluster_data['price_per_sqm'],
                color=cluster_colors[i],
                alpha=0.6,
                s=40,  # 减小点的大小
                edgecolors='white',
                linewidths=0.3,
                label=f'细分市场 {i}'
            )

        # 添加聚类中心
        centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
        plt.scatter(
            centers_original[:, 1],  # 面积
            centers_original[:, 0],  # 价格
            c='red',
            s=150,
            alpha=0.9,
            marker='X',
            edgecolors='white',
            linewidths=1.0,
            label='细分市场中心'
        )

        plt.title('成都二手房市场细分分析', fontsize=14, pad=20)
        plt.xlabel('面积 (平方米)', fontsize=12, labelpad=10)
        plt.ylabel('每平方米价格 (元)', fontsize=12, labelpad=10)
        plt.grid(True, alpha=0.3, linestyle='--')

        # 创建一个更好的图例
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9, edgecolor='gray')

        # 使用科学计数法格式化Y轴
        plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

        # 限制刻度数量
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

        # 2. 使用Plotly创建交互式市场细分图 - 改进版
        if 'bedrooms' in cluster_features:
            # 3D散点图
            fig = self.styler.create_plotly_figure(
                title='成都二手房3D市场细分可视化',
                x_title='面积 (平方米)',
                y_title='卧室数量'
            )

            # 为每个簇添加散点
            for i in range(n_clusters):
                cluster_data = df_clustered[df_clustered['cluster'] == i]

                # 转换颜色格式
                rgba_color = cluster_colors[i]
                plotly_color = f'rgba({int(rgba_color[0] * 255)},{int(rgba_color[1] * 255)},{int(rgba_color[2] * 255)},{rgba_color[3]})'

                fig.add_trace(go.Scatter3d(
                    x=cluster_data['area'],
                    y=cluster_data['bedrooms'],
                    z=cluster_data['price_per_sqm'],
                    mode='markers',
                    marker=dict(
                        size=4,  # 减小点的大小以减少视觉混乱
                        color=plotly_color,
                        opacity=0.7
                    ),
                    name=f'细分市场 {i}',
                    hovertemplate='面积: %{x:.1f}㎡<br>卧室: %{y}<br>单价: %{z:,.0f}元/㎡<extra>细分市场 %{customdata}</extra>',
                    customdata=[i] * len(cluster_data)
                ))

            # 添加聚类中心
            fig.add_trace(go.Scatter3d(
                x=centers_original[:, 1],  # 面积
                y=centers_original[:, 2] if centers_original.shape[1] > 2 else [0] * len(centers_original),  # 卧室
                z=centers_original[:, 0],  # 价格
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='cross',
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                name='细分市场中心',
                hovertemplate='面积: %{x:.1f}㎡<br>卧室: %{y}<br>单价: %{z:,.0f}元/㎡<extra>市场中心 %{customdata}</extra>',
                customdata=np.arange(n_clusters)
            ))

            # 调整3D图布局
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title='面积 (平方米)',
                        titlefont=dict(size=12),
                        backgroundcolor='rgba(255, 255, 255, 0.1)',
                        gridcolor='rgba(200, 200, 200, 0.5)'
                    ),
                    yaxis=dict(
                        title='卧室数量',
                        titlefont=dict(size=12),
                        backgroundcolor='rgba(255, 255, 255, 0.1)',
                        gridcolor='rgba(200, 200, 200, 0.5)',
                        tickmode='array',
                        tickvals=sorted(df_clustered['bedrooms'].unique()),
                    ),
                    zaxis=dict(
                        title='每平方米价格 (元)',
                        titlefont=dict(size=12),
                        tickformat=',.0f',
                        backgroundcolor='rgba(255, 255, 255, 0.1)',
                        gridcolor='rgba(200, 200, 200, 0.5)'
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=0.8, z=1)
                ),
                height=700,
                margin=dict(l=0, r=0, t=50, b=0)
            )

        else:
            # 2D散点图
            fig = self.styler.create_plotly_figure(
                title='成都二手房市场细分可视化',
                x_title='面积 (平方米)',
                y_title='每平方米价格 (元)'
            )

            # 为每个簇添加散点
            for i in range(n_clusters):
                cluster_data = df_clustered[df_clustered['cluster'] == i]

                # 转换颜色格式
                rgba_color = cluster_colors[i]
                plotly_color = f'rgba({int(rgba_color[0] * 255)},{int(rgba_color[1] * 255)},{int(rgba_color[2] * 255)},{rgba_color[3]})'

                fig.add_trace(go.Scatter(
                    x=cluster_data['area'],
                    y=cluster_data['price_per_sqm'],
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=plotly_color,
                        opacity=0.7
                    ),
                    name=f'细分市场 {i}',
                    hovertemplate='面积: %{x:.1f}㎡<br>单价: %{y:,.0f}元/㎡<extra>细分市场 %{customdata}</extra>',
                    customdata=[i] * len(cluster_data)
                ))

            # 添加聚类中心
            for i, center in enumerate(centers_original):
                fig.add_trace(go.Scatter(
                    x=[center[1]],  # 面积
                    y=[center[0]],  # 价格
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=12,
                        symbol='x',
                        line=dict(
                            color='white',
                            width=1
                        )
                    ),
                    name=f'细分{i}中心',
                    hoverinfo='name',
                    hovertemplate='面积: %{x:.1f}㎡<br>单价: %{y:,.0f}元/㎡<extra>市场中心 %{customdata}</extra>',
                    customdata=[i]
                ))

            # 格式化刻度
            fig.update_layout(
                xaxis=dict(
                    title_standoff=20
                ),
                yaxis=dict(
                    tickformat=',',
                    title_standoff=20
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=60, r=50, t=80, b=60)
            )

        # 保存交互式图表
        self.styler.save_figure(fig, "market_segmentation_interactive", formats=["html"])

        # 3. 各细分市场特征雷达图 - 改进版
        segments = segment_results['market_segments']

        # 准备雷达图数据
        categories = ['均价', '面积']
        if 'bedrooms' in cluster_features:
            categories.append('卧室数')

        # 归一化特征
        price_max = df['price_per_sqm'].max()
        area_max = df['area'].max()
        bedrooms_max = df['bedrooms'].max() if 'bedrooms' in df.columns else 1

        # 创建雷达图
        fig = self.styler.create_plotly_figure(
            title="成都二手房市场细分特征对比"
        )

        # 为每个细分市场添加雷达图数据
        for segment in segments:
            segment_id = segment['segment_id']

            # 准备归一化的特征值
            values = [
                segment['avg_price'] / price_max,
                segment['avg_area'] / area_max
            ]

            if 'bedrooms' in cluster_features:
                values.append(segment['avg_bedrooms'] / bedrooms_max)

            # 添加第一个点的重复，以闭合雷达图
            values.append(values[0])
            categories_closed = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                fill='toself',
                name=f'细分市场 {segment_id}: {segment["market_position"]}',
                hovertemplate='%{theta}: %{r:.2f}<extra>%{fullData.name}</extra>'
            ))

        # 调整雷达图布局
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            height=600,
            margin=dict(l=60, r=60, t=80, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 保存雷达图
        self.styler.save_figure(fig, "market_segments_radar", formats=["html"])

        # 4. 细分市场柱状图 - 改进版
        fig = self.styler.create_plotly_figure(
            title="成都二手房市场细分对比",
            x_title="细分市场",
            y_title="数值"
        )

        # 准备数据
        market_ids = [s['segment_id'] for s in segments]
        positions = [s['market_position'] for s in segments]
        prices = [s['avg_price'] for s in segments]
        areas = [s['avg_area'] for s in segments]
        percentages = [s['percentage'] for s in segments]

        # 添加均价柱状图
        fig.add_trace(go.Bar(
            x=market_ids,
            y=prices,
            name='平均单价 (元/㎡)',
            text=[f"{p:,.0f}" for p in prices],
            textposition='auto',
            marker_color=self.color_schemes["primary"],
            customdata=positions,
            hovertemplate='细分市场 %{x}: %{customdata}<br>均价: %{y:,.0f}元/㎡<extra></extra>'
        ))

        # 添加面积柱状图
        fig.add_trace(go.Bar(
            x=market_ids,
            y=areas,
            name='平均面积 (㎡)',
            text=[f"{a:.1f}" for a in areas],
            textposition='auto',
            marker_color=self.color_schemes["secondary"],
            customdata=positions,
            hovertemplate='细分市场 %{x}: %{customdata}<br>面积: %{y:.1f}㎡<extra></extra>'
        ))

        # 添加占比柱状图
        fig.add_trace(go.Bar(
            x=market_ids,
            y=percentages,
            name='占比 (%)',
            text=[f"{p:.1f}%" for p in percentages],
            textposition='auto',
            marker_color=self.color_schemes["accent"],
            customdata=positions,
            hovertemplate='细分市场 %{x}: %{customdata}<br>占比: %{y:.1f}%<extra></extra>'
        ))

        # 更新布局，添加按钮
        fig.update_layout(
            barmode='group',
            updatemenus=[{
                'buttons': [
                    {'label': "全部指标", 'method': "update", 'args': [{"visible": [True, True, True]}]},
                    {'label': "均价", 'method': "update", 'args': [{"visible": [True, False, False]}]},
                    {'label': "面积", 'method': "update", 'args': [{"visible": [False, True, False]}]},
                    {'label': "占比", 'method': "update", 'args': [{"visible": [False, False, True]}]}
                ],
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'y': 1.15,
                'xanchor': 'left',
                'yanchor': 'top',
                'bgcolor': 'rgba(240, 240, 240, 0.8)',
                'bordercolor': 'rgba(0, 0, 0, 0.5)',
                'font': {'size': 12}
            }],
            annotations=[
                dict(
                    text="选择指标:",
                    showarrow=False,
                    x=0,
                    y=1.15,
                    xref="paper",
                    yref="paper",
                    xanchor="left",
                    yanchor="top"
                )
            ],
            xaxis=dict(
                title_standoff=20
            ),
            yaxis=dict(
                title_standoff=20
            ),
            margin=dict(l=60, r=50, t=100, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # 格式化x轴，显示市场定位
        fig.update_xaxes(
            ticktext=[f"细分{i}: {p}" for i, p in zip(market_ids, positions)],
            tickvals=market_ids,
            tickangle=0
        )

        # 保存图表
        self.styler.save_figure(fig, "market_segments_comparison", formats=["html"])
        
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
        fig = self.styler.create_plotly_figure(title=title)

        # 添加热力图图层
        fig.add_trace(go.Densitymapbox(
            lat=valid_df['lat'],
            lon=valid_df['lng'],
            z=valid_df[value_column],
            radius=10,
            colorscale=colorscale,
            zmin=valid_df[value_column].quantile(0.1),  # 下限截断以突出差异
            zmax=valid_df[value_column].quantile(0.9),  # 上限截断以突出差异
            colorbar=dict(
                title=f"{value_column}",
                titlefont=dict(family="SimHei, Arial", size=12),
                tickfont=dict(family="SimHei, Arial", size=10),
                tickformat=","
            ),
            hovertext=[f"价格: {v:,.0f}元/㎡<br>地址: {a}"
                       for v, a in zip(valid_df[value_column], valid_df['formatted_address'])],
            hoverinfo="text"
        ))

        # 设置地图样式
        mapbox_style = "light" if self.styler.theme == "light" else "dark"

        # 计算地图中心点
        center_lat = valid_df['lat'].mean()
        center_lng = valid_df['lng'].mean()

        # 更新地图布局
        fig.update_layout(
            mapbox=dict(
                style=mapbox_style,
                center=dict(lat=center_lat, lon=center_lng),
                zoom=10
            ),
            height=800,  # 更大的高度以适应地图
            margin=dict(l=0, r=0, t=50, b=0),  # 减小边距以最大化地图区域
            hoverlabel=dict(
                font_size=12,
                font_family="Arial, SimHei"
            )
        )

        # 保存图表
        self.styler.save_figure(fig, "price_heatmap", formats=["html"])

        return fig

    def create_scatter_map(self, df, color_column='price_per_sqm', size_column='area',
                           title="成都二手房分布图"):
        """
        创建散点地图 - 地理可视化

        参数:
            df: 包含经纬度的数据框
            color_column: 用于确定点颜色的列名
            size_column: 用于确定点大小的列名
            title: 图表标题
        """
        if not self.geo_enabled:
            print("警告: 未加载GeoJSON数据，无法创建散点地图")
            return None

        # 检查数据框是否包含经纬度
        if 'lng' not in df.columns or 'lat' not in df.columns:
            print("错误: 数据框中没有经纬度列")
            return None

        # 过滤掉无效数据
        valid_df = df.dropna(subset=['lng', 'lat'])

        if len(valid_df) == 0:
            print("错误: 没有有效的坐标数据")
            return None

        print("创建地理散点图...")

        # 使用plotly express创建散点地图
        hover_data = {
            'lng': False,
            'lat': False,
            color_column: True,
            'formatted_address': True
        }

        if 'community' in valid_df.columns:
            hover_data['community'] = True

        if 'layout' in valid_df.columns:
            hover_data['layout'] = True

        if 'total_price' in valid_df.columns:
            hover_data['total_price'] = True
            
        fig = px.scatter_mapbox(
            valid_df,
            lat="lat",
            lon="lng",
            color=color_column,
            size=size_column if size_column in valid_df.columns else None,
            color_continuous_scale=px.colors.sequential.Viridis,
            size_max=12,  # 减小最大点大小以避免视觉混乱
            zoom=10,
            hover_name="district" if "district" in valid_df.columns else None,
            hover_data=hover_data,
            mapbox_style="light" if self.styler.theme == "light" else "dark",
            title=title,
            opacity=0.7  # 增加透明度以减少密集区域的视觉混乱
        )

        # 调整布局
        fig.update_layout(
            height=800,
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(
                title=color_column,
                titlefont=dict(family="SimHei, Arial", size=12),
                tickfont=dict(family="SimHei, Arial", size=10),
                tickformat=","
            ),
            hoverlabel=dict(
                font_size=12,
                font_family="Arial, SimHei"
            )
        )

        # 保存图表
        self.styler.save_figure(fig, "property_scatter_map", formats=["html"])

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
            # 计算每个区域的中位数价格，用于排序
            district_stats = df.groupby('district').agg(
                median=('price_per_sqm', 'median'),
                count=('price_per_sqm', 'count')
            ).reset_index()

            # 过滤掉样本量太少的区域
            district_stats = district_stats[district_stats['count'] >= 3]

            # 按中位数价格排序
            district_stats = district_stats.sort_values('median', ascending=False)

            # 限制显示的区域数，避免过于拥挤
            max_districts = 8  # 最多显示8个区域
            if len(district_stats) > max_districts:
                district_stats = district_stats.head(max_districts)

            # 为各区域生成渐变色
            district_colors = get_optimized_colormap(len(district_stats), 'viridis')

            for i, (_, row) in enumerate(district_stats.iterrows()):
                district = row['district']
                district_data = df[df['district'] == district]

                # 转换颜色格式为 rgba
                rgba_color = district_colors[i]
                plotly_color = f'rgba({int(rgba_color[0] * 255)},{int(rgba_color[1] * 255)},' \
                               f'{int(rgba_color[2] * 255)},{rgba_color[3]})'

                fig.add_trace(
                    go.Box(
                        y=district_data['price_per_sqm'].dropna(),
                        name=district,
                        marker_color=plotly_color,
                        showlegend=False,
                        boxmean=True,  # 显示均值
                        hovertemplate='区域: %{x}<br>价格: %{y:,.0f}元/㎡<extra></extra>'
                    ),
                    row=1, col=1
                )

        # 2. 面积与价格散点图
        if 'area' in df.columns and 'price_per_sqm' in df.columns:
            area_price_df = df[['area', 'price_per_sqm']].dropna()

            if len(area_price_df) > 5:
                # 抽样以减少点数，提高性能
                if len(area_price_df) > 500:
                    sample_df = area_price_df.sample(500, random_state=42)
                else:
                    sample_df = area_price_df

                fig.add_trace(
                    go.Scatter(
                        x=sample_df['area'],
                        y=sample_df['price_per_sqm'],
                        mode='markers',
                        marker=dict(
                            size=5,
                            opacity=0.7,
                            color=self.color_schemes["primary"]
                        ),
                        name='房源',
                        showlegend=False,
                        hovertemplate='面积: %{x:.1f}㎡<br>单价: %{y:,.0f}元/㎡<extra></extra>'
                    ),
                    row=1, col=2
                )

        # 3. 户型价格对比
        if 'layout' in df.columns and 'price_per_sqm' in df.columns:
            # 获取前5个最常见户型
            layout_counts = df['layout'].value_counts()
            top_layouts = layout_counts[layout_counts >= 3].index[:5].tolist()

            if top_layouts:
                layout_avg = df[df['layout'].isin(top_layouts)].groupby('layout')['price_per_sqm'].mean().sort_values(
                    ascending=False)

                # 为各户型生成渐变色
                layout_colors = get_optimized_colormap(len(layout_avg), 'plasma')

                # 转换颜色格式
                plotly_colors = [f'rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},{c[3]})'
                                 for c in layout_colors]

                fig.add_trace(
                    go.Bar(
                        x=layout_avg.index,
                        y=layout_avg.values,
                        marker_color=plotly_colors,
                        showlegend=False,
                        text=[f"{val:,.0f}" for val in layout_avg.values],
                        textposition='auto',
                        textfont=dict(size=10),
                        hovertemplate='户型: %{x}<br>均价: %{y:,.0f}元/㎡<extra></extra>'
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
                    marker_color=self.color_schemes["secondary"],
                    opacity=0.7,
                    showlegend=False,
                    hovertemplate='价格区间: %{x:,.0f}元/㎡<br>数量: %{y}<extra></extra>'
                ),
                row=2, col=2
            )

            # 添加核密度估计曲线
            kde = stats.gaussian_kde(price_data)
            x_vals = np.linspace(min(price_data), max(price_data), 100)
            y_vals = kde(x_vals)

            # 缩放KDE值以匹配直方图高度
            max_bin_count = np.histogram(price_data, bins=30)[0].max()
            y_vals = y_vals * max_bin_count / y_vals.max()

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            title=dict(
                text='成都二手房市场分析面板',
                font=dict(family="SimHei, Arial", size=16),
                x=0.5,
                y=0.99
            ),
            height=800,
            width=1000,
            template='plotly_white' if self.styler.theme == 'light' else 'plotly_dark',
            font=dict(family="SimHei, Arial", size=10),
            margin=dict(l=60, r=50, t=100, b=60),
            hoverlabel=dict(
                font_size=12,
                font_family="Arial, SimHei"
            )
        )

        # 更新坐标轴标题和格式
        fig.update_xaxes(title_text='区域', row=1, col=1, tickangle=30, title_standoff=20)
        fig.update_yaxes(title_text='每平方米价格 (元)', row=1, col=1, tickformat=',', title_standoff=20)

        fig.update_xaxes(title_text='面积 (平方米)', row=1, col=2, title_standoff=20)
        fig.update_yaxes(title_text='每平方米价格 (元)', row=1, col=2, tickformat=',', title_standoff=20)

        fig.update_xaxes(title_text='户型', row=2, col=1, tickangle=30, title_standoff=20)
        fig.update_yaxes(title_text='平均每平方米价格 (元)', row=2, col=1, tickformat=',', title_standoff=20)

        fig.update_xaxes(title_text='每平方米价格 (元)', row=2, col=2, tickformat=',', title_standoff=20)
        fig.update_yaxes(title_text='数量', row=2, col=2, title_standoff=20)

        # 更新子图标题字体
        for annotation in fig.layout.annotations:
            annotation.font.size = 12
            annotation.font.family = "SimHei, Arial"

        # 保存交互式数据面板
        self.styler.save_figure(fig, "chengdu_housing_dashboard", formats=["html"])
        print("交互式数据面板已创建: data/visualizations/chengdu_housing_dashboard.html")

    def create_all_visualizations(self, df, analysis_results=None):
        """
        创建所有数据可视化

        参数:
            df: 数据框
            analysis_results: 分析结果字典
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

            # 创建市场细分可视化
            if analysis_results and 'market_segments' in analysis_results:
                self.create_market_segment_plots(df, analysis_results['market_segments'])

            # 检查是否存在地理数据
            if self.geo_enabled:
                # 创建热力图
                self.create_heatmap(df)

                # 创建散点地图
                self.create_scatter_map(df)

            # 创建综合数据面板
            self.create_dashboard(df, analysis_results)

            print("\n所有可视化已创建完成!")
            print(f"可视化文件保存在 data/visualizations/ 目录下")

        except Exception as e:
            print(f"创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    import sys

    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        geojson_file = sys.argv[2] if len(sys.argv) > 2 else None
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

        # 检查是否有地理编码数据
        geocoded_dir = "data/geocoded"
        if os.path.exists(geocoded_dir):
            geojson_files = [f for f in os.listdir(geocoded_dir) if f.endswith('.geojson')]
            geojson_file = os.path.join(geocoded_dir, sorted(geojson_files)[-1]) if geojson_files else None
        else:
            geojson_file = None

    # 加载数据
    try:
        df = pd.read_csv(input_file, encoding='utf-8-sig')
        print(f"从 {input_file} 加载了 {len(df)} 条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        sys.exit(1)

    # 创建可视化对象
    visualizer = ChengduHousingVisualizer(theme="light")

    # 如果有地理文件，加载它
    if geojson_file and os.path.exists(geojson_file):
        visualizer.enable_geo_visualization(geojson_file)

    # 创建可视化
    visualizer.create_all_visualizations(df)

                # 添加趋势线
                z = np.polyfit(area_price_df['area'], area_price_df['price_per_sqm'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(area_price_df['area'].min(), area_price_df['area'].max(), 100)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='趋势线',
                        showlegend=False,
                        hovertemplate='面积: %{x:.1f}㎡<br>预测单价: %{y:,.0f}元/㎡<extra></extra>'
                    ),
                    row=1, col=2
                )
