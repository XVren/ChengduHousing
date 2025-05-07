# 链家成都二手房数据分析系统

## 项目简介
本项目是一个成都二手房数据分析系统，用于爬取、清洗、分析和可视化链家网上的成都二手房数据，特别关注房价的空间结构分析。

## 功能特点
- 数据爬取：从链家网抓取成都二手房数据
- 数据清洗：处理异常值、格式化数据、特征工程
- 地理编码：将房源地址转换为经纬度坐标
- 空间分析：分析房价空间分布特征
- 结构分析：分析户型、面积等结构因素对价格的影响
- 市场细分：使用聚类算法识别不同的市场细分
- 数据可视化：生成各类分析图表和交互式面板

## 系统要求
- Python 3.8+
- 依赖库列表见 requirements.txt

## 安装方法
```bash
# 克隆仓库
git clone https://github.com/yourusername/lianjia-chengdu-housing.git
cd lianjia-chengdu-housing

# 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate  # Linux/MacOS
# 或
.\.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 使用方法
```bash
# 运行完整工作流程
python main.py

# 指定输入文件
python main.py --input path/to/your/data.csv

# 使用地理编码功能（需要高德地图API密钥）
python main.py --api_key YOUR_AMAP_API_KEY
```

## 项目结构
```
lianjia_chengdu_housing/
├── data/                   # 数据目录
├── src/                    # 源代码目录
│   ├── scraper.py          # 数据爬取模块
│   ├── cleaner.py          # 数据清洗模块
│   ├── geocoder.py         # 地理编码模块
│   ├── analyzer.py         # 数据分析模块
│   ├── visualizer.py       # 数据可视化模块
│   └── utils/              # 工具函数目录
├── main.py                 # 主程序入口
└── requirements.txt        # 项目依赖
```

## 输出结果
- 处理后的数据文件: `data/processed/`
- 地理编码结果: `data/geocoded/`
- 可视化结果: `data/visualizations/`
- 分析报告: 生成的JSON文件包含详细统计指标

## 更新记录
- 2025/5/8：更新配置文件
- 2025/5/7: 解决可视化中文显示问题
- 2025/5/6: 重构项目结构，优化可视化模块
- 2025/5/4: 添加空间结构分析功能
- 2025/5/2：添加地理编码功能
- 2025/4/30: 初始版本创建
