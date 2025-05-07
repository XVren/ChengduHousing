#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
链家成都二手房数据分析主程序
协调整个数据分析流程，从爬取到可视化
"""
import os
import time
import argparse
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('lianjia_chengdu.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# 导入项目模块
from src.scraper import LianjiaChengduScraper
from src.cleaner import ChengduHousingCleaner
from src.geocoder import ChengduGeocoder
from src.analyzer import ChengduHousingAnalyzer
from src.visualizer import ChengduHousingVisualizer


def run_analysis_workflow(input_file=None, api_key=None, num_pages=5, enable_scraping=False):
    """
    运行完整的分析工作流程

    参数:
        input_file: 输入CSV文件路径，若为None则使用爬虫采集
        api_key: 高德地图API密钥，用于地理编码(可选)
        num_pages: 爬虫采集的页面数量
        enable_scraping: 是否启用爬虫采集新数据
    """
    logger.info("=" * 50)
    logger.info("成都二手房价格空间结构分析工作流程启动")
    logger.info("=" * 50)

    # 创建必要的目录
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/visualizations", exist_ok=True)
    os.makedirs("data/geocoded", exist_ok=True)

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 步骤1: 数据采集
    if enable_scraping or input_file is None:
        logger.info("\n步骤 1: 数据爬取")
        logger.info("-" * 30)
        try:
            scraper = LianjiaChengduScraper()
            input_file = scraper.run(num_pages=num_pages)
            if not input_file:
                logger.error("数据爬取失败")
                return
        except Exception as e:
            logger.error(f"数据爬取步骤出错: {e}", exc_info=True)
            if input_file is None:
                return
    else:
        logger.info("跳过数据爬取步骤，使用已有数据")

    # 确认输入文件存在
    if not os.path.exists(input_file):
        logger.error(f"错误: 输入文件 {input_file} 不存在")
        return

    logger.info(f"使用数据集: {input_file}")

    # 步骤2: 数据清洗和预处理
    logger.info("\n步骤 2: 数据清洗和预处理")
    logger.info("-" * 30)
    try:
        cleaner = ChengduHousingCleaner()
        processed_data, output_file = cleaner.process_data(input_file)

        if processed_data is None or output_file is None:
            logger.error(f"数据清洗失败")
            return

        logger.info(f"成功清洗和处理了 {len(processed_data)} 条房源数据")
        logger.info(f"处理后的数据已保存至: {output_file}")
    except Exception as e:
        logger.error(f"数据清洗步骤出错: {e}", exc_info=True)
        return

    # 步骤3: 地理编码 (可选)
    geocoded_file = None
    geojson_file = None
    enable_geo = False

    if api_key:
        logger.info("\n步骤 3: 地理编码房源数据")
        logger.info("-" * 30)
        try:
            geocoder = ChengduGeocoder(api_key=api_key)

            # 执行地理编码流程
            geocoded_data, geocoded_file = geocoder.process_data(output_file, api_key)

            # 检查是否创建了GeoJSON文件
            geojson_files = [f for f in os.listdir("data/geocoded") if f.endswith('.geojson')]
            if geojson_files:
                geojson_file = os.path.join("data/geocoded", sorted(geojson_files)[-1])
                enable_geo = True
                logger.info(f"地理编码和GeoJSON生成成功")

            # 更新工作流中使用的数据文件为地理编码后的文件
            if geocoded_file and os.path.exists(geocoded_file):
                output_file = geocoded_file
                logger.info(f"将使用地理编码后的数据继续分析")
        except Exception as e:
            logger.error(f"地理编码步骤出错: {e}", exc_info=True)
            logger.info("将继续使用未编码的数据进行分析")

    # 步骤4: 数据分析
    logger.info("\n步骤 4: 空间结构数据分析")
    logger.info("-" * 30)
    try:
        analyzer = ChengduHousingAnalyzer()
        results, df = analyzer.run_complete_analysis(output_file)

        if results is None or df is None:
            logger.error("分析失败")
            return

        logger.info("空间结构分析成功完成")
    except Exception as e:
        logger.error(f"数据分析步骤出错: {e}", exc_info=True)
        return

    # 步骤5: 可视化
    logger.info("\n步骤 5: 空间结构数据可视化")
    logger.info("-" * 30)
    try:
        visualizer = ChengduHousingVisualizer()

        # 如果地理编码成功，加载地理数据
        if enable_geo and geojson_file and os.path.exists(geojson_file):
            visualizer.enable_geo_visualization(geojson_file)
            logger.info(f"已启用地理空间可视化")

        visualizer.create_all_visualizations(df, results)
        logger.info("可视化创建成功")
    except Exception as e:
        logger.error(f"可视化步骤出错: {e}", exc_info=True)
        return

    # 完成所有步骤，打印汇总信息
    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\n" + "=" * 50)
    logger.info(f"空间结构分析工作流程成功完成，总耗时: {total_time:.2f}秒")
    logger.info("=" * 50)
    logger.info("\n结果保存在:")
    logger.info(f"- 输入数据: {input_file}")
    logger.info(f"- 处理后数据: {output_file}")
    if enable_geo:
        logger.info(f"- 地理编码数据: data/geocoded/")
    logger.info(f"- 可视化结果: data/visualizations/")

    # 打开数据面板
    dashboard_path = os.path.abspath("data/visualizations/chengdu_housing_dashboard.html")
    logger.info(f"\n要查看分析结果，请在浏览器中打开数据面板:\n{dashboard_path}")


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='成都二手房价格空间结构分析工作流程')
    parser.add_argument('--input', '-i', type=str,
                        default="null",
                        help='输入CSV文件路径，若不提供则使用默认路径')
    parser.add_argument('--api_key', '-k', type=str,
                        default="null",
                        help='高德地图API密钥，用于地理编码')
    parser.add_argument('--pages', '-p', type=int, default=5,
                        help='爬虫采集的页面数量，默认为5')
    parser.add_argument('--scrape', '-s', action='store_true',
                        help='是否启用爬虫采集新数据')
    args = parser.parse_args()

    # 运行分析工作流程
    try:
        run_analysis_workflow(
            input_file=args.input,
            api_key=args.api_key,
            num_pages=args.pages,
            enable_scraping=args.scrape
        )
    except Exception as e:
        logger.error(f"程序执行出错: {e}", exc_info=True)