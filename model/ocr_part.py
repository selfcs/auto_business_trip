#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/15 18:20
# @Author  : senlanxu@gmail.com
# @File    : ocr_part.py

"""
通过paddlepaddle框架进行发票、行程单的ocr识别
"""

import os
import cv2
import paddlehub as hub
from paddleocr import PaddleOCR, draw_ocr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ORC(object):
    def __init__(self, img_path):
        self.img_path = img_path
        # self.ocr_handle = PaddleOCR(use_angle_cls=True, lang="ch")
        self.ocr_handle = hub.Module(name="chinese_ocr_db_crnn_server")

    def clean_img(self):
        np_images = [cv2.imread(self.img_path)]
        return np_images

    def vat_ocr(self):
        # result = self.ocr_handle.ocr(self.img_path, cls=True)
        # get_info = {"发票代码", "发票号码", "开票日期", ""}
        # for line in result:
        #     # 1. 发票代码
        #     print(line[5][0])
        np_images = self.clean_img()

        results = self.ocr_handle.recognize_text(
            images=np_images,  # 图片数据，ndarray.shape 为 [H, W, C]，BGR格式；
            use_gpu=False,  # 是否使用 GPU；若使用GPU，请先设置CUDA_VISIBLE_DEVICES环境变量
            output_dir='../data/output_data/',  # 图片的保存路径，默认设为 ocr_result；
            visualization=False,  # 是否将识别结果保存为图片文件；
            box_thresh=0.3,  # 检测文本框置信度的阈值；
            text_thresh=0.5)  # 识别中文文本置信度的阈值；
        return results


if __name__ == '__main__':
    ocr_cls = ORC("../data/input_data/20210622111500165.jpg")
    print(ocr_cls.vat_ocr()[0])
