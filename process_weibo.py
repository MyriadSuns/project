# -*- coding: utf-8 -*-
"""Weibo数据集处理脚本"""

import os
import json
import csv
import logging
import requests
import re
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Weibo')
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_weibo')
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'weibo_images')
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'weibo_videos')
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

CSV_HEADER = ['id', 'text', 'image', 'video', 'label', 'proof']

# 配置参数
MIN_TEXT_LENGTH = 5

# 创建会话
SESSION = requests.Session()


# ===================== 目录创建函数 =====================
def create_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(VIDEOS_DIR, exist_ok=True)


# ===================== 文本有效性检查 =====================
def has_valid_text(text):
    if not text or len(text.strip()) < MIN_TEXT_LENGTH:
        return False
    return True


# ===================== 图片有效性检查 =====================
def has_valid_images(data):
    pic_infos = data.get('pic_infos', {})
    return bool(pic_infos and len(pic_infos) > 0)


# ===================== 视频有效性检查 =====================
def has_valid_video(data):
    url_struct = data.get('url_struct', [])
    for url_item in url_struct:
        for url_key in ['short_url', 'long_url']:
            url = url_item.get(url_key, '')
            if url and 'http' in url and any(keyword in url for keyword in ['weibo.com/tv/show', 'video.weibo.com']):
                return True

    text_raw = data.get('text_raw', '')
    if text_raw:
        video_patterns = [
            r'https?://weibo\.com/tv/show/[^\s]+',
            r'https?://video\.weibo\.com/[^\s]+',
        ]
        for pattern in video_patterns:
            if re.search(pattern, text_raw):
                return True

    page_info = data.get('page_info', {})
    media_info = page_info.get('media_info', {})

    video_url_keys = [
        'stream_url_hd',
        'mp4_sd_url',
        'mp4_hd_url',
        'h265_mp4_hd',
        'h265_mp4_ld',
        'inch_4_mp4_hd',
        'inch_5_mp4_hd',
        'inch_5_5_mp4_hd',
        'mp4_720p_mp4'
    ]

    for key in video_url_keys:
        url = media_info.get(key, '')
        if url and 'http' in url and '.mp4' in url:
            return True

    return False


# ===================== 手动文件查找函数 =====================
def find_manual_image(post_id):
    image_path = os.path.join(IMAGES_DIR, f"{post_id}.jpg")
    if os.path.exists(image_path):
        return os.path.join('images', f"{post_id}.jpg")
    return None


def find_manual_video(post_id):
    video_path = os.path.join(VIDEOS_DIR, f"{post_id}.mp4")
    if os.path.exists(video_path):
        return os.path.join('videos', f"{post_id}.mp4")
    return None


# ===================== 类别处理函数 =====================
def process_category(category_path, label):
    category_name = 'fake_news' if label == 1 else 'real_news'
    samples = []
    success_count = 0
    error_count = 0

    if not os.path.exists(category_path):
        return samples

    for root, dirs, files in os.walk(category_path):
        for dir_name in tqdm(dirs, desc=f"处理 {category_name}"):
            try:
                json_path = os.path.join(root, dir_name, 'new.json')
                if not os.path.exists(json_path):
                    error_count += 1
                    continue

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                has_text = has_valid_text(data.get('text_raw', ''))

                if not has_text:
                    continue

                post_id = data.get('idstr', dir_name)

                image_path = find_manual_image(post_id)
                video_path = find_manual_video(post_id)

                if not image_path and not video_path:
                    continue

                proof = data.get('proof', '')
                if isinstance(proof, list):
                    proof = ';'.join(proof)
                elif not isinstance(proof, str):
                    proof = str(proof)
                proof = proof.strip()

                sample = {
                    'id': post_id,
                    'text': data.get('text_raw', '').strip(),
                    'image': image_path if image_path else '',
                    'video': video_path if video_path else '',
                    'label': label,
                    'proof': proof
                }
                samples.append(sample)
                success_count += 1

            except Exception:
                error_count += 1

    return samples


# ===================== CSV创建函数 =====================
def create_empty_csv(file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()


def split_train_val_test(samples, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    import random
    random.seed(42)  # 固定随机种子
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


# ===================== 主函数 =====================
def main():
    logger.info("开始处理Weibo数据集...")

    create_directories()
    fake_news_path = os.path.join(DATA_DIR, 'fake_news')
    real_news_path = os.path.join(DATA_DIR, 'real_news')
    all_samples = []

    if os.path.exists(fake_news_path):
        fake_samples = process_category(fake_news_path, 1)
        all_samples.extend(fake_samples)

    if os.path.exists(real_news_path):
        real_samples = process_category(real_news_path, 0)
        all_samples.extend(real_samples)

    if not all_samples:
        logger.error("未处理到任何有效样本！")
        return

    train_samples, val_samples, test_samples = split_train_val_test(all_samples)

    train_csv_path = os.path.join(OUTPUT_DIR, 'weibo_train.csv')
    val_csv_path = os.path.join(OUTPUT_DIR, 'weibo_val.csv')
    test_csv_path = os.path.join(OUTPUT_DIR, 'weibo_test.csv')

    create_empty_csv(train_csv_path)
    create_empty_csv(val_csv_path)
    create_empty_csv(test_csv_path)

    with open(train_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(train_samples)

    with open(val_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(val_samples)

    with open(test_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writerows(test_samples)

    total = len(all_samples)
    logger.info(f"\n===== 处理结果 =====")
    logger.info(f"总有效样本数: {total}")
    logger.info(f"训练集: {len(train_samples)} 条")
    logger.info(f"验证集: {len(val_samples)} 条")
    logger.info(f"测试集: {len(test_samples)} 条")
    logger.info(f"训练集CSV: {train_csv_path}")
    logger.info(f"验证集CSV: {val_csv_path}")
    logger.info(f"测试集CSV: {test_csv_path}")
    logger.info(f"图片目录: {IMAGES_DIR}")
    logger.info(f"视频目录: {VIDEOS_DIR}")


if __name__ == "__main__":
    main()