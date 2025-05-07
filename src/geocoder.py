# -*- coding: utf-8 -*-
"""
链家成都二手房地理编码模块
用于给房源数据添加地理坐标，支持多种地理编码方式和坐标系转换
"""
import requests
import pandas as pd
import numpy as np
import json
import os
import time
import math
import hashlib
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import random


class CoordinateConverter:
    """
    中国坐标系统转换工具
    支持WGS84(GPS)、GCJ02(火星坐标系)、BD09(百度坐标系)相互转换
    """

    def __init__(self):
        """
        初始化坐标转换器
        """
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = 3.1415926535897932384626  # π
        self.a = 6378245.0  # 长半轴
        self.ee = 0.00669342162296594323  # 偏心率平方

    def _out_of_china(self, lng, lat):
        """
        判断坐标是否在中国境外

        参数:
            lng: 经度
            lat: 纬度

        返回:
            bool: 是否在中国境外
        """
        return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)

    def _transform_lat(self, lng, lat):
        """
        WGS84转GCJ02的纬度转换算法

        参数:
            lng: WGS84经度
            lat: WGS84纬度

        返回:
            float: 偏移量
        """
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
                math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
                math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lng(self, lng, lat):
        """
        WGS84转GCJ02的经度转换算法

        参数:
            lng: WGS84经度
            lat: WGS84纬度

        返回:
            float: 偏移量
        """
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
                math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
                math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def wgs84_to_gcj02(self, lng, lat):
        """
        WGS84(GPS)坐标系 转 GCJ02(火星坐标系)

        参数:
            lng: WGS84经度
            lat: WGS84纬度

        返回:
            tuple: (GCJ02经度, GCJ02纬度)
        """
        if self._out_of_china(lng, lat):
            return lng, lat

        d_lat = self._transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self._transform_lng(lng - 105.0, lat - 35.0)

        rad_lat = lat / 180.0 * self.pi
        magic = math.sin(rad_lat)
        magic = 1 - self.ee * magic * magic

        sqrt_magic = math.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrt_magic) * self.pi)
        d_lng = (d_lng * 180.0) / (self.a / sqrt_magic * math.cos(rad_lat) * self.pi)

        return lng + d_lng, lat + d_lat

    def gcj02_to_wgs84(self, lng, lat):
        """
        GCJ02(火星坐标系) 转 WGS84(GPS)坐标系

        参数:
            lng: GCJ02经度
            lat: GCJ02纬度

        返回:
            tuple: (WGS84经度, WGS84纬度)
        """
        if self._out_of_china(lng, lat):
            return lng, lat

        d_lat = self._transform_lat(lng - 105.0, lat - 35.0)
        d_lng = self._transform_lng(lng - 105.0, lat - 35.0)

        rad_lat = lat / 180.0 * self.pi
        magic = math.sin(rad_lat)
        magic = 1 - self.ee * magic * magic

        sqrt_magic = math.sqrt(magic)
        d_lat = (d_lat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrt_magic) * self.pi)
        d_lng = (d_lng * 180.0) / (self.a / sqrt_magic * math.cos(rad_lat) * self.pi)

        return lng - d_lng, lat - d_lat

    def gcj02_to_bd09(self, lng, lat):
        """
        GCJ02(火星坐标系) 转 BD09(百度坐标系)

        参数:
            lng: GCJ02经度
            lat: GCJ02纬度

        返回:
            tuple: (BD09经度, BD09纬度)
        """
        z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * self.x_pi)
        theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * self.x_pi)
        bd_lng = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return bd_lng, bd_lat

    def bd09_to_gcj02(self, lng, lat):
        """
        BD09(百度坐标系) 转 GCJ02(火星坐标系)

        参数:
            lng: BD09经度
            lat: BD09纬度

        返回:
            tuple: (GCJ02经度, GCJ02纬度)
        """
        x = lng - 0.0065
        y = lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gcj_lng = z * math.cos(theta)
        gcj_lat = z * math.sin(theta)
        return gcj_lng, gcj_lat

    def wgs84_to_bd09(self, lng, lat):
        """
        WGS84(GPS)坐标系 转 BD09(百度坐标系)

        参数:
            lng: WGS84经度
            lat: WGS84纬度

        返回:
            tuple: (BD09经度, BD09纬度)
        """
        lng, lat = self.wgs84_to_gcj02(lng, lat)
        return self.gcj02_to_bd09(lng, lat)

    def bd09_to_wgs84(self, lng, lat):
        """
        BD09(百度坐标系) 转 WGS84(GPS)坐标系

        参数:
            lng: BD09经度
            lat: BD09纬度

        返回:
            tuple: (WGS84经度, WGS84纬度)
        """
        lng, lat = self.bd09_to_gcj02(lng, lat)
        return self.gcj02_to_wgs84(lng, lat)

    def convert_coordinates(self, lng, lat, from_type, to_type):
        """
        坐标系统转换通用接口

        参数:
            lng: 输入坐标经度
            lat: 输入坐标纬度
            from_type: 输入坐标系类型，可选 'wgs84', 'gcj02', 'bd09'
            to_type: 输出坐标系类型，可选 'wgs84', 'gcj02', 'bd09'

        返回:
            tuple: 转换后的(经度, 纬度)坐标
        """
        if from_type == to_type:
            return lng, lat

        # 先转换为WGS84作为中间坐标系
        if from_type == 'gcj02':
            lng, lat = self.gcj02_to_wgs84(lng, lat)
        elif from_type == 'bd09':
            lng, lat = self.bd09_to_wgs84(lng, lat)

        # 再从WGS84转换为目标坐标系
        if to_type == 'gcj02':
            return self.wgs84_to_gcj02(lng, lat)
        elif to_type == 'bd09':
            return self.wgs84_to_bd09(lng, lat)

        return lng, lat  # WGS84


class ChengduGeocoder:
    """
    成都地区地理编码器
    结合高德地图API和本地缓存，实现高效地理编码
    """

    def __init__(self, api_key=None, cache_file="chengdu_geocode_cache.pkl"):
        """
        初始化地理编码器

        参数:
            api_key: 高德地图API密钥，若不提供则尝试仅使用离线缓存
            cache_file: 缓存文件路径
        """
        self.api_key = api_key
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.coord_converter = CoordinateConverter()

        # 创建数据保存目录
        os.makedirs('data/geocoded', exist_ok=True)

        # 检查缓存状态
        if len(self.cache) > 0:
            print(f"已加载 {len(self.cache)} 条地址缓存")

        # 添加请求节流控制参数
        self.last_request_time = 0
        self.min_request_interval = 0.4  # 每秒最多2.5次请求，低于API的3次/秒限制

    def _load_cache(self):
        """
        加载缓存数据

        返回:
            dict: 缓存数据字典
        """
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"加载缓存文件失败: {e}")
                return {}
        return {}

    def _save_cache(self):
        """
        保存缓存数据到文件
        """
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"保存缓存文件失败: {e}")

    def _cache_key(self, address):
        """
        生成缓存键

        参数:
            address: 地址字符串

        返回:
            str: 缓存键
        """
        return hashlib.md5(address.encode('utf-8')).hexdigest()

    def _throttle_request(self):
        """
        控制请求频率，确保不超过API限制
        """
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.min_request_interval:
            # 计算需要等待的时间
            sleep_time = self.min_request_interval - elapsed
            time.sleep(sleep_time)

        # 更新上次请求时间
        self.last_request_time = time.time()

    def preprocess_address(self, address):
        """
        预处理成都地址，提高地理编码准确率

        参数:
            address: 原始地址

        返回:
            str: 处理后的地址
        """
        if not address:
            return ""

        # 移除多余空格
        address = ' '.join(address.split())

        # 如果地址不包含省市信息，添加成都市前缀
        if "成都" not in address and "四川" not in address:
            address = "成都市" + address
        elif "四川" in address and "成都" not in address:
            address = address.replace("四川", "四川省成都市")

        # 标准化成都区域名称
        district_mapping = {
            "武侯": "武侯区",
            "锦江": "锦江区",
            "青羊": "青羊区",
            "金牛": "金牛区",
            "成华": "成华区",
            "高新": "高新区",
            "天府": "天府新区",
            "双流": "双流区",
            "郫都": "郫都区",
            "温江": "温江区",
            "新都": "新都区",
            "龙泉驿": "龙泉驿区",
            "青白江": "青白江区",
            "简阳": "简阳市",
            "彭州": "彭州市",
            "邛崃": "邛崃市",
            "崇州": "崇州市",
            "金堂": "金堂县",
            "新津": "新津区",
            "都江堰": "都江堰市",
            "蒲江": "蒲江县",
            "大邑": "大邑县"
        }

        for key, value in district_mapping.items():
            if key in address and value not in address:
                # 确保只替换独立的区名，而不是替换地址中包含这些字的部分
                parts = address.split()
                for i, part in enumerate(parts):
                    if key in part and value not in part:
                        parts[i] = part.replace(key, value)
                address = ' '.join(parts)

        return address

    def geocode_with_retry(self, address, max_retries=3):
        """
        带重试逻辑的地理编码

        参数:
            address: 地址字符串
            max_retries: 最大重试次数

        返回:
            dict: 地理编码结果或None
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                result = self.geocode_address(address)
                return result
            except Exception as e:
                error_msg = str(e)
                if "CUQPS_HAS_EXCEEDED_THE_LIMIT" in error_msg:
                    retry_count += 1
                    # 指数退避，每次重试等待时间增加
                    wait_time = 2 ** retry_count
                    print(f"请求频率限制，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    # 其他错误直接抛出
                    print(f"地理编码失败: {error_msg}")
                    return None

        print(f"达到最大重试次数({max_retries})，地理编码失败")
        return None

    def geocode_address(self, address):
        """
        对单个地址进行地理编码

        参数:
            address: 地址字符串

        返回:
            dict: 包含经纬度的结果字典
        """
        if not address:
            return None

        # 预处理地址
        processed_address = self.preprocess_address(address)

        # 检查缓存
        cache_key = self._cache_key(processed_address)
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 如果没有API密钥，无法进行在线地理编码
        if not self.api_key:
            print(f"无法地理编码: {processed_address} (未提供API密钥)")
            return None

        # 使用高德地图API进行地理编码
        try:
            # 应用请求节流控制
            self._throttle_request()

            url = 'https://restapi.amap.com/v3/geocode/geo'
            params = {
                'address': processed_address,
                'key': self.api_key,
                'city': '成都'  # 限定城市范围提高准确度
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                result = response.json()
                if result['status'] == '1' and result['geocodes']:
                    geocode = result['geocodes'][0]

                    # 解析经纬度（高德返回格式为"经度,纬度"的字符串）
                    location = geocode['location']
                    lng, lat = map(float, location.split(','))

                    # 构建结果字典
                    geocode_result = {
                        'address': address,
                        'formatted_address': geocode['formatted_address'],
                        'lng': lng,
                        'lat': lat,
                        'district': geocode.get('district', ''),
                        'township': geocode.get('township', ''),
                        'neighborhood': geocode.get('neighborhood', ''),
                        'coordinate_system': 'gcj02'  # 高德坐标系统
                    }

                    # 更新缓存
                    self.cache[cache_key] = geocode_result
                    self._save_cache()

                    return geocode_result
                else:
                    error_info = result.get('info', '')
                    if "CUQPS_HAS_EXCEEDED_THE_LIMIT" in error_info:
                        raise Exception(f"API请求频率超限: {error_info}")
                    else:
                        print(f"地理编码API错误: {processed_address}, {error_info}")
            else:
                print(f"HTTP请求错误, 状态码: {response.status_code}")

        except Exception as e:
            print(f"地理编码异常 {processed_address}: {e}")
            raise  # 重新抛出异常，让重试机制捕获

        return None

    def geocode_batch(self, addresses, max_workers=2, sleep_time=0.5):
        """
        批量地理编码

        参数:
            addresses: 地址列表
            max_workers: 最大并行线程数 (减少到2，API限制是3)
            sleep_time: 请求间隔时间(秒) (增加到0.5秒)

        返回:
            list: 地理编码结果列表
        """
        results = []
        uncached_addresses = []

        print("开始批量地理编码...")

        # 先检查哪些地址在缓存中
        for address in addresses:
            processed_address = self.preprocess_address(address)
            cache_key = self._cache_key(processed_address)

            if cache_key in self.cache:
                results.append((address, self.cache[cache_key]))
            else:
                uncached_addresses.append((address, processed_address))

        print(f"缓存命中: {len(results)}/{len(addresses)}")
        print(f"需要地理编码: {len(uncached_addresses)}")

        # 如果没有API密钥，只能使用缓存数据
        if not self.api_key and uncached_addresses:
            print(f"警告: 无法获取 {len(uncached_addresses)} 个地址的坐标，未提供API密钥")
            for address, _ in uncached_addresses:
                results.append((address, None))
        # 处理未缓存的地址
        elif uncached_addresses:
            if max_workers > 1 and len(uncached_addresses) > 10:
                # 多线程并行处理
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    for i, (orig_address, proc_address) in enumerate(uncached_addresses):
                        # 添加延迟以避免API限流
                        time.sleep(sleep_time)
                        futures.append(executor.submit(self.geocode_with_retry, proc_address))

                    # 获取结果
                    for i, future in enumerate(tqdm(futures, desc="地理编码进度")):
                        try:
                            orig_address = uncached_addresses[i][0]
                            geocode_result = future.result()
                            results.append((orig_address, geocode_result))

                            # 每处理10个请求就保存一次缓存
                            if i % 10 == 0:
                                self._save_cache()
                        except Exception as e:
                            print(f"处理地址时出错: {e}")
                            results.append((uncached_addresses[i][0], None))
            else:
                # 单线程顺序处理
                for i, (orig_address, proc_address) in enumerate(tqdm(uncached_addresses, desc="地理编码进度")):
                    try:
                        # 应用延迟前增加随机偏移，避免严格的定时模式
                        jitter = 0.2 * (0.5 - random.random())  # ±0.1随机偏移
                        time.sleep(sleep_time + jitter)

                        geocode_result = self.geocode_with_retry(proc_address)
                        results.append((orig_address, geocode_result))

                        # 每处理5个请求就保存一次缓存
                        if i % 5 == 0:
                            self._save_cache()
                    except Exception as e:
                        print(f"处理地址时出错: {e}")
                        results.append((orig_address, None))

                        # 遇到错误时额外等待，避免连续请求失败
                        time.sleep(2)

        # 最终保存一次缓存
        self._save_cache()

        # 转换结果格式
        address_to_result = {addr: res for addr, res in results}
        final_results = [address_to_result.get(addr) for addr in addresses]

        return final_results

    def convert_coordinate_system(self, results, to_system='wgs84'):
        """
        转换地理编码结果的坐标系统

        参数:
            results: 地理编码结果列表
            to_system: 目标坐标系统，可选 'wgs84', 'gcj02', 'bd09'

        返回:
            list: 转换后的地理编码结果列表
        """
        converted_results = []

        for result in results:
            if result is None or 'lng' not in result or 'lat' not in result:
                converted_results.append(result)
                continue

            from_system = result.get('coordinate_system', 'gcj02')  # 默认假设是高德坐标(GCJ02)

            if from_system == to_system:
                converted_results.append(result)
                continue

            lng, lat = self.coord_converter.convert_coordinates(
                result['lng'], result['lat'], from_system, to_system
            )

            # 创建新的结果对象，避免修改原始对象
            new_result = result.copy()
            new_result['lng'] = lng
            new_result['lat'] = lat
            new_result['coordinate_system'] = to_system

            converted_results.append(new_result)

        return converted_results

    def process_addresses_in_batches(self, addresses, batch_size=20):
        """
        分批处理地址，避免触发API限制

        参数:
            addresses: 地址列表
            batch_size: 每批处理的地址数量

        返回:
            list: 所有地址的地理编码结果
        """
        all_results = []
        total_batches = (len(addresses) + batch_size - 1) // batch_size

        for i in range(0, len(addresses), batch_size):
            # 分批处理
            batch = addresses[i:i + batch_size]
            batch_num = i // batch_size + 1

            print(f"\n处理地址批次 {batch_num}/{total_batches} (共{len(batch)}个地址)")
            batch_results = self.geocode_batch(batch, max_workers=2, sleep_time=0.5)
            all_results.extend(batch_results)

            # 每个批次后保存缓存
            self._save_cache()

            # 批次间等待，避免触发频率限制
            if i + batch_size < len(addresses):
                wait_time = min(5, 1 + batch_num // 5)  # 随着批次增加，适当增加等待时间
                print(f"批次处理完成，等待{wait_time}秒后继续...")
                time.sleep(wait_time)

        return all_results

    def geocode_dataframe(self, df, address_columns=None, create_address=True, coordinate_system='wgs84'):
        """
        为数据框添加地理编码信息

        参数:
            df: 待处理的数据框
            address_columns: 用于构建地址的列名列表，如 ['district', 'community']
            create_address: 是否从多列创建完整地址
            coordinate_system: 输出坐标系统

        返回:
            pd.DataFrame: 添加经纬度后的数据框
        """
        # 如果没有指定地址列且数据框中存在 "address" 列
        if address_columns is None and "address" in df.columns:
            address_columns = ["address"]

        # 如果没有指定地址列且数据框中存在 "district" 和 "community" 列
        if address_columns is None and "district" in df.columns and "community" in df.columns:
            address_columns = ["district", "community"]

        # 如果仍然无法确定地址列
        if address_columns is None:
            print("错误: 无法确定地址列，请指定 address_columns 参数")
            return df

        # 创建完整地址
        if create_address and len(address_columns) > 1:
            # 当地址由多列组成时，将它们连接成一个完整地址
            df['geocode_address'] = df[address_columns].apply(
                lambda row: "成都市" + ''.join(str(x) for x in row if pd.notna(x) and str(x) != 'nan'),
                axis=1
            )
        else:
            # 使用单列作为地址
            df['geocode_address'] = df[address_columns[0]]

        # 获取所有地址的列表
        addresses = df['geocode_address'].tolist()

        # 分批处理地址
        print(f"开始地理编码 {len(addresses)} 个地址...")
        geocode_results = self.process_addresses_in_batches(addresses, batch_size=20)

        # 转换坐标系统
        if coordinate_system != 'gcj02':
            print(f"转换坐标系统到 {coordinate_system}...")
            geocode_results = self.convert_coordinate_system(geocode_results, coordinate_system)

        # 添加经纬度到数据框
        df['lng'] = [r['lng'] if r else None for r in geocode_results]
        df['lat'] = [r['lat'] if r else None for r in geocode_results]

        # 添加其他地理信息
        df['formatted_address'] = [r['formatted_address'] if r else None for r in geocode_results]
        df['coordinate_system'] = coordinate_system

        # 删除临时地址列
        if create_address:
            df.drop('geocode_address', axis=1, inplace=True)

        # 计算成功率
        success_count = sum(1 for r in geocode_results if r is not None)
        print(f"地理编码完成: {success_count}/{len(addresses)} 成功 "
              f"({success_count / len(addresses) * 100:.1f}%)")

        return df

    def save_to_geojson(self, df, output_file="data/geocoded/chengdu_properties.geojson"):
        """
        将包含经纬度的数据框保存为GeoJSON文件

        参数:
            df: 包含经纬度的数据框
            output_file: 输出文件路径

        返回:
            bool: 是否成功保存
        """
        # 检查数据框是否包含经纬度
        if 'lng' not in df.columns or 'lat' not in df.columns:
            print("错误: 数据框中没有经纬度列")
            return False

        # 过滤掉没有坐标的记录
        valid_df = df.dropna(subset=['lng', 'lat'])

        if len(valid_df) == 0:
            print("错误: 没有有效的坐标数据")
            return False

        # 构建GeoJSON特征集合
        features = []

        for _, row in valid_df.iterrows():
            # 构建属性字典
            properties = {col: row[col] for col in row.index if col not in ['lng', 'lat']}

            # 将数值转换为Python原生类型，避免JSON序列化错误
            for key, value in properties.items():
                if pd.isna(value):
                    properties[key] = None
                elif isinstance(value, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
                    properties[key] = value.isoformat()
                elif isinstance(value, (float, np.float64, np.float32)):
                    properties[key] = float(value)
                elif isinstance(value, (int, np.int64, np.int32)):
                    properties[key] = int(value)

            # 构建GeoJSON特征
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row['lng']), float(row['lat'])]
                },
                "properties": properties
            }

            features.append(feature)

        # 构建GeoJSON对象
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        # 保存到文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, ensure_ascii=False, indent=2)
            print(f"已保存 {len(features)} 条记录到 {output_file}")
            return True
        except Exception as e:
            print(f"保存GeoJSON文件失败: {e}")
            return False

    def process_data(self, input_file, api_key=None):
        """
        运行地理编码流程

        参数:
            input_file: 输入CSV文件
            api_key: 高德地图API密钥(可选)

        返回:
            (geocoded_df, output_file): 地理编码后的数据框和保存的文件路径
        """
        print("=" * 50)
        print("开始地理编码")
        print("=" * 50)

        # 设置API密钥
        if api_key:
            self.api_key = api_key

        # 检查API密钥
        if not self.api_key:
            print("警告: 未提供API密钥，将仅使用缓存数据")

        # 加载数据
        try:
            df = pd.read_csv(input_file, encoding='utf-8-sig')
            print(f"从 {input_file} 加载了 {len(df)} 条记录")
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None, None

        # 进行地理编码
        geocoded_df = self.geocode_dataframe(
            df,
            address_columns=['district', 'community'],
            coordinate_system='wgs84'
        )

        # 保存地理编码结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"data/geocoded/chengdu_properties_geocoded_{timestamp}.csv"
        geocoded_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"地理编码结果已保存至: {output_file}")

        # 保存为GeoJSON格式
        geojson_file = f"data/geocoded/chengdu_properties_{timestamp}.geojson"
        if 'lng' in geocoded_df.columns and 'lat' in geocoded_df.columns:
            self.save_to_geojson(geocoded_df, geojson_file)
            print(f"GeoJSON数据已保存至: {geojson_file}")

        print("\n地理编码完成!")

        return geocoded_df, output_file


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    import sys
    import random

    # 默认使用测试地址
    test_addresses = [
        "成都市武侯区天府大道北段1700号环球中心",
        "锦江区红星路三段1号成都IFS国际金融中心",
        "成华区建设北路三段28号",
        "高新区天府大道中段688号"
    ]

    # 从命令行获取API密钥和输入文件
    api_key = None
    input_file = None

    if len(sys.argv) > 1:
        if os.path.exists(sys.argv[1]):
            input_file = sys.argv[1]
        else:
            api_key = sys.argv[1]

    if len(sys.argv) > 2:
        if api_key is None:
            api_key = sys.argv[2]
        elif input_file is None and os.path.exists(sys.argv[2]):
            input_file = sys.argv[2]

    # 初始化地理编码器
    geocoder = ChengduGeocoder(api_key=api_key)

    # 根据输入参数决定运行模式
    if input_file:
        print(f"使用输入文件: {input_file}")
        geocoder.process_data(input_file, api_key)
    else:
        # 测试单个地址
        print("测试单个地址地理编码:")
        result = geocoder.geocode_with_retry(test_addresses[0])
        print(result)

        # 测试批量地理编码
        print("\n测试批量地理编码:")
        results = geocoder.process_addresses_in_batches(test_addresses, batch_size=2)

        for addr, result in zip(test_addresses, results):
            if result:
                print(f"{addr} -> ({result['lng']}, {result['lat']})")
            else:
                print(f"{addr} -> 地理编码失败")