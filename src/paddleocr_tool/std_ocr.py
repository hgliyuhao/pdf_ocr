#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/25 16:04
# @Author  : jiaoxiaoyu
# @File    : std_ocr.py
# @Description :
import requests
import numpy as np
import cv2
import re
import os
from bs4 import BeautifulSoup

from .paddleocr_pac import PaddleOCR

# BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(BASE_PATH)

# MODEL_BASE = os.path.join(BASE_PATH,"model/whl")

# print(MODEL_BASE)

MODEL_BASE = "model/whl"

ocr = PaddleOCR(use_angle_cls=True, lang="ch", cls_image_shape="3, 1024, 2048",
                label_list=['0', '90', '180', '270'],
                show_log=True, use_onnx=True,
                cls_model_dir=os.path.join(MODEL_BASE, 'cls/text_image_orientation_infer/model.onnx'),
                det_model_dir=os.path.join(MODEL_BASE, 'det/ch/ch_PP-OCRv4_det_infer/model.onnx'),
                rec_model_dir=os.path.join(MODEL_BASE, 'rec/ch/ch_PP-OCRv4_rec_infer/model.onnx'),
                )

def runV2(img):
    results = []

    result = ocr.ocr(img, det=True, rec=True, cls=False)
    for idx in range(len(result)):
        res = result[idx]
        if res is None:
            continue

        for line in res:
            # 保留一些关键符号
            txt = re.sub(r'[^\w\u4e00-\u9fa5.-]', '', line[1][0])
            txt_conf = line[1][1]
            if len(txt) == 0 or txt_conf <= 0.4:
                continue
            point = line[0][0] + line[0][2]
            results.append(txt)

    return results

def download_and_convert_image(image_url):
    # 使用requests下载图片
    response = requests.get(image_url)
    # 确认请求成功
    if response.status_code == 200:
        image_data = response.content
        # 将原始数据转换为numpy数组，适用于图像解码
        image_array = np.frombuffer(image_data, np.uint8)
        # 使用cv2.imdecode读取图像数据
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return img
    else:
        print("Failed to download image.")
        return None


if __name__ == "__main__":
    pass
