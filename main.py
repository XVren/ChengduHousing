#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
链家成都二手房数据分析主程序
"""

import os
import sys
import time
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scraper import LianjiaChengduScraper
from src.cleaner import ChengduHousingCleaner
from src.analyzer import ChengduHousingAnalyzer
from src.geocoder import ChengduGeocoder
from src.visualizer import ChengduHousingVisualizer


def run_scraping_workflow(num_pages=5):
    """
    运行爬虫工作流
    """
    scraper = LianjiaChengduScraper()
    raw_data_file = scraper.run(num_pages=num_pages)
    return raw_data_file


def run_cleaning_workflow(input_file=None):
    """
    运行数据清洗工作流
    """
    cleaner = ChengduHousingCleaner()

    # 如果未指定输入文件，使用最新的爬取数据
    if not input_file:
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir):
            print(f"目录 {raw_dir} 不存在，请先运行爬虫")
            return None

        data_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not data_files:
            print("未找到任何原始数据文件，请先运行爬虫")
            return None

        # 按修改时间排序，选择最新的文件
        input_file = os.path.join(raw_dir, sorted(data_files)[-1])

    # 运行数据清洗
    _, processed_file = cleaner.process_data(input_file)
    return processed_file


def run_geocoding_workflow(input_file=None, api_key=None):
    """
    运行地理编码工作流
    """
    # 检查API密钥
    if not api_key:
        api_key_file = "api_key.txt"
        if os.path.exists(api_key_file):
            with open(api_key_file, "r") as f:
                api_key = f.read().strip()
        else:
            print("警告：未提供高德地图API密钥，地理编码将仅使用缓存数据")

    # 如果未指定输入文件，使用最新的清洗后数据
    if not input_file:
        processed_dir = "data/processed"
        if not os.path.exists(processed_dir):
            print(f"目录 {processed_dir} 不存在，请先运行数据清洗")
            return None

        data_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
        if not data_files:
            print("未找到任何处理后的数据文件，请先运行数据清洗")
            return None

        # 按修改时间排序，选择最新的文件
        input_file = os.path.join(processed_dir, sorted(data_files)[-1])

    # 运行地理编码
    geocoder = ChengduGeocoder(api_key=api_key)
    _, geocoded_file = geocoder.process_data(input_file)
    return geocoded_file


def run_analysis_workflow(input_file=None, visualize=True):
    """
    运行数据分析工作流
    """
    # 如果未指定输入文件，使用最新的地理编码后或清洗后数据
    if not input_file:
        # 首先检查地理编码后的数据
        geocoded_dir = "data/geocoded"
        if os.path.exists(geocoded_dir):
            geo_files = [f for f in os.listdir(geocoded_dir) if f.endswith('.csv')]
            if geo_files:
                # 按修改时间排序，选择最新的文件
                input_file = os.path.join(geocoded_dir, sorted(geo_files)[-1])
                print(f"使用最新的地理编码数据: {input_file}")

        # 如果没有地理编码数据，使用清洗后的数据
        if not input_file:
            processed_dir = "data/processed"
            if not os.path.exists(processed_dir):
                print(f"目录 {processed_dir} 不存在，请先运行数据清洗")
                return None

            data_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv')]
            if not data_files:
                print("未找到任何处理后的数据文件，请先运行数据清洗")
                return None

            # 按修改时间排序，选择最新的文件
            input_file = os.path.join(processed_dir, sorted(data_files)[-1])
            print(f"使用最新的清洗数据: {input_file}")

    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在 - {input_file}")
        return None

    # 运行数据分析
    analyzer = ChengduHousingAnalyzer()
    results, df = analyzer.run_complete_analysis(input_file)

    # 如果需要可视化
    if visualize and df is not None:
        print("\n步骤 5: 空间结构可视化")
        print("=" * 50)

        # 初始化可视化器
        visualizer = ChengduHousingVisualizer(theme="light")

        # 检查是否有地理数据文件
        geojson_files = [f for f in os.listdir("data/geocoded") if f.endswith('.geojson')] if os.path.exists(
            "data/geocoded") else []
        if geojson_files:
            # 使用最新的GeoJSON文件
            geojson_file = os.path.join("data/geocoded", sorted(geojson_files)[-1])
            print(f"已启用地理可视化: {geojson_file}")
            visualizer.enable_geo_visualization(geojson_file)

        # 创建所有可视化 - 注意这里传递了分析结果参数
        visualizer.create_all_visualizations(df, results)

    return results


def run_complete_workflow(num_pages=5, enable_geocoding=False, api_key=None):
    """
    运行完整的工作流程，从爬取数据到分析可视化

    参数:
        num_pages: 爬取的页面数
        enable_geocoding: 是否启用地理编码
        api_key: 高德地图API密钥(地理编码需要)
    """
    print("\n" + "=" * 50)
    print("开始运行成都二手房数据分析工作流")
    print("=" * 50)

    start_time = time.time()

    # 步骤1: 数据爬取
    print("\n步骤 1: 数据爬取")
    raw_data_file = run_scraping_workflow(num_pages)

    # 步骤2: 数据清洗
    print("\n步骤 2: 数据清洗")
    processed_file = run_cleaning_workflow(raw_data_file)

    # 步骤3: 地理编码(可选)
    geocoded_file = None
    if enable_geocoding:
        print("\n步骤 3: 地理编码")
        geocoded_file = run_geocoding_workflow(processed_file, api_key)
        input_file_for_analysis = geocoded_file
    else:
        input_file_for_analysis = processed_file

    # 步骤4: 数据分析
    print("\n步骤 4: 空间结构分析")
    results = run_analysis_workflow(input_file_for_analysis)

    # 输出总结
    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 50)
    print("工作流程执行完成")
    print(f"总运行时间: {elapsed_time:.2f} 秒")
    print("=" * 50)


def main():
    """
    主函数，处理命令行参数并执行相应的工作流
    """
    import argparse

    parser = argparse.ArgumentParser(description='成都链家二手房数据分析工具')

    # 子命令设置
    subparsers = parser.add_subparsers(dest='command', help='运行模式')

    # 完整工作流
    workflow_parser = subparsers.add_parser('workflow', help='运行完整工作流')
    workflow_parser.add_argument('--pages', type=int, default=5, help='爬取页数')
    workflow_parser.add_argument('--geo', action='store_true', help='启用地理编码')
    workflow_parser.add_argument('--key', type=str, help='高德地图API密钥')

    # 单独运行爬虫
    scraper_parser = subparsers.add_parser('scrape', help='只运行爬虫')
    scraper_parser.add_argument('--pages', type=int, default=5, help='爬取页数')

    # 单独运行数据清洗
    cleaner_parser = subparsers.add_parser('clean', help='只运行数据清洗')
    cleaner_parser.add_argument('--input', type=str, help='输入CSV文件路径')

    # 单独运行地理编码
    geocode_parser = subparsers.add_parser('geocode', help='只运行地理编码')
    geocode_parser.add_argument('--input', type=str, help='输入CSV文件路径')
    geocode_parser.add_argument('--key', type=str, help='高德地图API密钥')

    # 单独运行分析
    analyze_parser = subparsers.add_parser('analyze', help='只运行数据分析')
    analyze_parser.add_argument('--input', type=str, help='输入CSV文件路径')
    analyze_parser.add_argument('--no-viz', action='store_true', help='不生成可视化')

    # 解析参数
    args = parser.parse_args()

    # 简单调试输出
    print(f"命令行参数: {args}")

    # 处理命令
    if args.command == 'workflow':
        run_complete_workflow(args.pages, args.geo, args.key)
    elif args.command == 'scrape':
        run_scraping_workflow(args.pages)
    elif args.command == 'clean':
        run_cleaning_workflow(args.input)
    elif args.command == 'geocode':
        run_geocoding_workflow(args.input, args.key)
    elif args.command == 'analyze':
        run_analysis_workflow(args.input, not args.no_viz)
    else:
        # 如果没有指定命令或命令不匹配，显示帮助
        print("未指定有效命令，请使用以下命令之一: workflow, scrape, clean, geocode, analyze")
        parser.print_help()
        # 默认执行分析命令(如果这是您想要的行为)
        if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
            print("\n检测到'analyze'命令，执行数据分析...")
            run_analysis_workflow(None, True)


if __name__ == "__main__":
    main()
