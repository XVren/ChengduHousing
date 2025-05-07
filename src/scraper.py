# -*- coding: utf-8 -*-
"""
链家成都二手房数据爬虫模块
负责从链家网站爬取成都二手房数据
"""
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import csv
from datetime import datetime


class LianjiaChengduScraper:
    def __init__(self):
        """
        初始化爬虫类
        """
        # 设置链家成都二手房的基础URL
        self.base_url = "https://cd.lianjia.com/ershoufang/"

        # 设置请求头信息，模拟浏览器访问，避免被反爬
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Referer': 'https://cd.lianjia.com/',
            'Connection': 'keep-alive',
        }

        # 创建数据保存目录
        os.makedirs('data/raw', exist_ok=True)

    def scrape_listings_page(self, page_url):
        """
        爬取单个页面的房源列表数据

        参数:
            page_url: 页面URL

        返回:
            properties_data: 包含该页面所有房源数据的列表
        """
        try:
            # 添加随机延迟，避免请求过于频繁导致被封IP
            time.sleep(random.uniform(1, 3))

            # 发送HTTP请求获取页面内容
            print(f"正在请求页面: {page_url}")
            response = requests.get(page_url, headers=self.headers)

            # 检查请求是否成功
            if response.status_code != 200:
                print(f"请求失败，状态码: {response.status_code}")
                return []

            # 使用BeautifulSoup解析HTML内容
            soup = BeautifulSoup(response.text, 'lxml')

            # 找到所有房源列表项
            listings = soup.select('ul.sellListContent li.clear')
            print(f"找到 {len(listings)} 个房源")

            # 初始化存储房源数据的列表
            properties_data = []

            # 处理每个房源
            for listing in listings:
                try:
                    # 提取房源标题
                    title_elem = listing.select_one('.title a')
                    title = title_elem.text.strip() if title_elem else "未知"

                    # 获取房源详情页链接
                    link = title_elem['href'] if title_elem else ""

                    # 提取区域和小区信息
                    position_info = listing.select_one('.positionInfo')
                    if position_info:
                        district_elem = position_info.select_one('a')
                        district = district_elem.text.strip() if district_elem else "未知"

                        community_elem = position_info.select('a')
                        community = community_elem[1].text.strip() if len(community_elem) > 1 else "未知"
                    else:
                        district = "未知"
                        community = "未知"

                    # 提取房源详细信息
                    house_info = listing.select_one('.houseInfo')
                    if house_info:
                        house_info_text = house_info.text.strip()
                        house_info_parts = house_info_text.split('|')

                        # 提取户型(如: 3室2厅)
                        layout = house_info_parts[0].strip() if len(house_info_parts) > 0 else "未知"

                        # 提取面积(如: 89平米)
                        area_text = house_info_parts[1].strip() if len(house_info_parts) > 1 else "未知"
                        area = area_text.replace('平米', '').strip() if '平米' in area_text else "未知"
                    else:
                        layout = "未知"
                        area = "未知"

                    # 提取建筑年代(房龄)信息
                    # 通常在positionInfo中包含类似"2008年建"的信息
                    position_text = position_info.text.strip() if position_info else ""
                    year = None
                    if '年建' in position_text:
                        year_text = position_text.split('年建')[0].strip()
                        # 从文本中提取数字
                        year_digits = ''.join(filter(str.isdigit, year_text))
                        if year_digits:
                            year = int(year_digits)

                    # 计算房龄
                    current_year = datetime.now().year
                    property_age = current_year - year if year else None

                    # 提取总价信息
                    total_price_elem = listing.select_one('.totalPrice span')
                    total_price = total_price_elem.text.strip() if total_price_elem else "未知"

                    # 提取单价信息(每平方米价格)
                    unit_price_elem = listing.select_one('.unitPrice span')
                    unit_price_text = unit_price_elem.text.strip() if unit_price_elem else "未知"
                    # 清洗单价文本(移除"单价"和"元/平米"字样)
                    unit_price = unit_price_text.replace('单价', '').replace('元/平米', '').strip()

                    # 存储房源数据为字典
                    property_data = {
                        'title': title,  # 标题
                        'link': link,  # 链接
                        'district': district,  # 区域
                        'community': community,  # 小区名
                        'layout': layout,  # 户型
                        'area': area,  # 面积
                        'construction_year': year,  # 建筑年代
                        'property_age': property_age,  # 房龄
                        'total_price': total_price,  # 总价(万元)
                        'price_per_sqm': unit_price  # 单价(元/平方米)
                    }

                    properties_data.append(property_data)
                except Exception as e:
                    print(f"处理房源时出错: {e}")
                    continue

            return properties_data

        except Exception as e:
            print(f"爬取页面 {page_url} 时出错: {e}")
            return []

    def scrape_multiple_pages(self, num_pages=5):
        """
        爬取多个页面的房源数据并保存为CSV文件

        参数:
            num_pages: 要爬取的页面数量

        返回:
            csv_file: 保存的CSV文件路径
            all_properties: 所有爬取到的房源数据列表
        """
        all_properties = []

        for page in range(1, num_pages + 1):
            # 构建页面URL(链家分页格式为pg1, pg2...)
            page_url = f"{self.base_url}pg{page}/"
            print(f"开始爬取第 {page} 页，共 {num_pages} 页: {page_url}")

            # 爬取页面数据
            page_properties = self.scrape_listings_page(page_url)
            all_properties.extend(page_properties)

            print(f"第 {page} 页爬取完成，获取到 {len(page_properties)} 条房源数据")

        # 保存为CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"data/raw/chengdu_properties_{timestamp}.csv"

        with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            if all_properties:
                writer = csv.DictWriter(f, fieldnames=all_properties[0].keys())
                writer.writeheader()
                writer.writerows(all_properties)

        print(f"共爬取 {len(all_properties)} 条房源数据，已保存到 {csv_file}")
        return csv_file, all_properties

    def run(self, num_pages=5):
        """
        运行爬虫并返回结果

        参数:
            num_pages: 要爬取的页面数量

        返回:
            csv_file: 保存的CSV文件路径
        """
        print("=" * 50)
        print("开始爬取链家成都二手房数据")
        print("=" * 50)

        csv_file, _ = self.scrape_multiple_pages(num_pages)

        print("\n爬取完成!")
        print(f"数据已保存至: {csv_file}")

        return csv_file


# 如果直接运行此脚本，则执行爬虫
if __name__ == "__main__":
    scraper = LianjiaChengduScraper()
    scraper.run(num_pages=5)