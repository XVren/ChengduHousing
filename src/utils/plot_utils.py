# -*- coding: utf-8 -*-
"""
链家成都二手房数据可视化美化工具模块
用于改进matplotlib和plotly图表的美观性
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)


class PlotStyler:
    """
    图表样式美化工具类
    提供对matplotlib和plotly图表的美化功能
    """

    def __init__(self, font_family="SimHei", theme="light", figure_size=(10, 6)):
        """
        初始化样式设置工具

        参数:
            font_family: 字体名称，默认为"SimHei"(黑体)
            theme: 主题风格，可选"light"或"dark"
            figure_size: 默认图表大小
        """
        self.font_family = font_family
        self.theme = theme
        self.figure_size = figure_size

        # 成都市区域颜色映射
        self.district_colors = {}  # 将在setup_matplotlib_style中初始化

        # 价格区间颜色映射
        self.price_colors = {}  # 将在setup_matplotlib_style中初始化

        # 配置matplotlib
        self.setup_matplotlib_style()

        # 创建输出目录
        os.makedirs("data/visualizations", exist_ok=True)

    def setup_matplotlib_style(self):
        """
        配置matplotlib样式
        设置字体、颜色等基本样式
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = [self.font_family, 'Microsoft YaHei', 'SimSun', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

        # 设置图表默认大小
        plt.rcParams['figure.figsize'] = self.figure_size

        # 设置DPI
        plt.rcParams['figure.dpi'] = 150

        # 设置线条宽度
        plt.rcParams['lines.linewidth'] = 2.0

        # 设置网格线样式
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.3

        # 设置刻度标签大小
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        # 设置标题和轴标签大小
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12

        # 设置图例样式
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['legend.framealpha'] = 0.8

        # 设置紧凑布局
        plt.rcParams['figure.constrained_layout.use'] = True

        # 设置深浅主题
        if self.theme == "dark":
            plt.rcParams['figure.facecolor'] = '#2E3440'
            plt.rcParams['axes.facecolor'] = '#2E3440'
            plt.rcParams['text.color'] = 'white'
            plt.rcParams['axes.labelcolor'] = 'white'
            plt.rcParams['xtick.color'] = 'white'
            plt.rcParams['ytick.color'] = 'white'
            plt.rcParams['grid.color'] = '#4C566A'
            plt.rcParams['legend.facecolor'] = '#3B4252'

            # 暗色主题下的颜色映射
            district_cmap = plt.cm.plasma
            price_cmap = plt.cm.viridis
        else:
            # 浅色主题下的颜色映射
            district_cmap = plt.cm.viridis
            price_cmap = plt.cm.plasma

        # 为成都各区域生成颜色映射
        district_names = [
            "锦江区", "青羊区", "金牛区", "武侯区", "成华区", "龙泉驿区",
            "青白江区", "新都区", "温江区", "双流区", "郫都区", "新津区",
            "金堂县", "大邑县", "蒲江县", "都江堰市", "彭州市", "崇州市",
            "邛崃市", "简阳市", "高新区", "天府新区"
        ]

        # 生成颜色列表
        district_colors = district_cmap(np.linspace(0, 1, len(district_names)))
        self.district_colors = {name: color for name, color in zip(district_names, district_colors)}

        # 价格区间颜色映射
        price_ranges = ["低价", "中低价", "中价", "中高价", "高价", "超高价"]
        price_colors = price_cmap(np.linspace(0, 1, len(price_ranges)))
        self.price_colors = {name: color for name, color in zip(price_ranges, price_colors)}

    def configure_axis_ticks(self, ax, axis='both', max_ticks=10, integer=False, rotation=None):
        """
        配置坐标轴刻度，避免过于密集的刻度标签

        参数:
            ax: matplotlib轴对象
            axis: 要配置的轴，可选'x', 'y', 'both'
            max_ticks: 最大刻度数量
            integer: 是否强制使用整数刻度
            rotation: 刻度标签旋转角度
        """
        if axis in ['x', 'both']:
            # 设置x轴最大刻度数
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks, integer=integer))

            # 处理标签重叠
            if rotation is not None:
                plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')

        if axis in ['y', 'both']:
            # 设置y轴最大刻度数
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=max_ticks, integer=integer))

    def add_value_labels(self, ax, spacing=5, fmt="{:.0f}"):
        """
        为条形图添加数值标签

        参数:
            ax: matplotlib轴对象
            spacing: 值标签与条形顶部的间距
            fmt: 数值格式化字符串
        """
        # 遍历所有条形
        for rect in ax.patches:
            # 获取条形高度
            height = rect.get_height()

            # 在条形顶部添加文本标签
            ax.annotate(
                fmt.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, spacing),  # 垂直偏移
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9,
                fontweight='bold'
            )

    def create_plotly_figure(self, title="", x_title="", y_title="", template="plotly_white"):
        """
        创建基本的plotly图表对象

        参数:
            title: 图表标题
            x_title: x轴标题
            y_title: y轴标题
            template: plotly模板名称

        返回:
            go.Figure: plotly图表对象
        """
        # 创建图表
        fig = go.Figure()

        # 设置布局
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'family': 'SimHei, Arial'},
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis={
                'title': {
                    'text': x_title,
                    'font': {'size': 14, 'family': 'SimHei, Arial'}
                },
                'tickfont': {'size': 12, 'family': 'SimHei, Arial'}
            },
            yaxis={
                'title': {
                    'text': y_title,
                    'font': {'size': 14, 'family': 'SimHei, Arial'}
                },
                'tickfont': {'size': 12, 'family': 'SimHei, Arial'}
            },
            template=template,
            width=900,  # 默认宽度
            height=600,  # 默认高度
            margin=dict(l=80, r=80, t=100, b=80),  # 边距
            legend={
                'font': {'size': 12, 'family': 'SimHei, Arial'},
                'orientation': 'h',
                'y': -0.15  # 将图例放在图表下方
            },
            # 悬停提示框样式
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="SimHei, Arial"
            )
        )

        # 如果是暗色主题
        if self.theme == "dark":
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='#2E3440',
                plot_bgcolor='#2E3440',
                font=dict(color='white')
            )

        return fig

    def format_plotly_ticks(self, fig, max_xticks=10, max_yticks=10, xrotation=0, show_x_every_nth=1):
        """
        格式化plotly图表刻度

        参数:
            fig: plotly图表对象
            max_xticks: x轴最大刻度数
            max_yticks: y轴最大刻度数
            xrotation: x轴标签旋转角度
            show_x_every_nth: 显示每n个x刻度，用于稀疏刻度
        """
        # 更新x轴显示
        fig.update_xaxes(
            tickangle=xrotation,
            nticks=max_xticks
        )

        # 更新y轴显示
        fig.update_yaxes(
            nticks=max_yticks
        )

        # 如果需要跳过刻度
        if show_x_every_nth > 1:
            # 获取当前刻度
            try:
                current_ticks = fig.layout.xaxis.tickvals
                current_labels = fig.layout.xaxis.ticktext

                if current_ticks is not None:
                    # 筛选要显示的刻度
                    new_ticks = current_ticks[::show_x_every_nth]
                    new_labels = current_labels[::show_x_every_nth]

                    # 更新刻度
                    fig.update_xaxes(tickvals=new_ticks, ticktext=new_labels)
            except:
                pass  # 如果获取刻度失败，则跳过

        return fig

    def save_figure(self, fig, filename, formats=["png", "html"]):
        """
        保存图表到文件

        参数:
            fig: matplotlib图形对象或plotly图表对象
            filename: 文件名(不含扩展名)
            formats: 要保存的格式列表
        """
        # 创建输出目录
        os.makedirs("data/visualizations", exist_ok=True)

        # 构建完整文件路径(不含扩展名)
        filepath_base = f"data/visualizations/{filename}"

        # 判断图表类型并保存
        if isinstance(fig, plt.Figure):  # matplotlib图表
            for fmt in formats:
                if fmt.lower() == "png":
                    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches='tight')
                elif fmt.lower() == "pdf":
                    fig.savefig(f"{filepath_base}.pdf", bbox_inches='tight')
                elif fmt.lower() == "svg":
                    fig.savefig(f"{filepath_base}.svg", bbox_inches='tight')
            plt.close(fig)  # 关闭图表释放内存

        else:  # 假定为plotly图表
            for fmt in formats:
                if fmt.lower() == "html":
                    fig.write_html(f"{filepath_base}.html")
                elif fmt.lower() == "png":
                    fig.write_image(f"{filepath_base}.png", scale=2)
                elif fmt.lower() == "pdf":
                    fig.write_image(f"{filepath_base}.pdf")
                elif fmt.lower() == "svg":
                    fig.write_image(f"{filepath_base}.svg")


# 中文字体检测和推荐函数
def check_chinese_fonts():
    """
    检测系统中可用的中文字体

    返回:
        list: 可用中文字体列表
    """
    chinese_fonts = []

    # 常见中文字体关键字
    chinese_keywords = ['SimHei', 'SimSun', 'Microsoft YaHei', 'FangSong', 'KaiTi',
                        '黑体', '宋体', '微软雅黑', '仿宋', '楷体',
                        'Noto Sans CJK', 'Noto Serif CJK', 'WenQuanYi',
                        'Source Han', '思源', 'Heiti']

    # 遍历系统字体
    for font in fm.fontManager.ttflist:
        for keyword in chinese_keywords:
            if keyword.lower() in font.name.lower():
                chinese_fonts.append(font.name)
                break

    # 去重
    chinese_fonts = list(set(chinese_fonts))

    return chinese_fonts


# 优化过的色彩方案
def get_optimized_colormap(n_colors, cmap_name='viridis'):
    """
    获取优化的颜色方案

    参数:
        n_colors: 需要的颜色数量
        cmap_name: 色彩映射名称

    返回:
        list: 颜色列表
    """
    # 色彩映射对象
    cmap = plt.cm.get_cmap(cmap_name, n_colors)

    # 提取颜色
    colors = [cmap(i) for i in range(n_colors)]

    # 如果颜色过少，直接返回
    if n_colors <= 2:
        return colors

    # 打乱颜色顺序，使相邻颜色更易区分
    # 采用间隔取色的方式
    reordered_colors = []

    # 将奇数和偶数位置的颜色分开处理
    even_indices = list(range(0, n_colors, 2))
    odd_indices = list(range(1, n_colors, 2))

    # 交错合并两组颜色
    for i in range(max(len(even_indices), len(odd_indices))):
        if i < len(even_indices):
            reordered_colors.append(colors[even_indices[i]])
        if i < len(odd_indices):
            reordered_colors.append(colors[odd_indices[i]])

    return reordered_colors


# 初始化包
if __name__ == "__main__":
    # 检测可用中文字体
    available_fonts = check_chinese_fonts()
    print("可用中文字体:")
    for font in available_fonts:
        print(f"  - {font}")