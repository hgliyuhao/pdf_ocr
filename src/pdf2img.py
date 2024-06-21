import fitz  # PyMuPDF 的别名
import os
import cv2
import numpy as np
import json
from paddleocr_tool.std_ocr import runV2
from tqdm import tqdm

def pdf_to_ocr(pdf_path, output_folder, dpi=300):
    """
    将 PDF 文件转换为图像，并使用OCR提取文本，将结果保存到文件，不在磁盘上保存图像。
    
    :param pdf_path: PDF 文件路径
    :param output_folder: 输出结果文件夹路径
    :param dpi: 图像的分辨率（默认 300）
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)
    
    # 打开 PDF 文件
    with fitz.open(pdf_path) as pdf_document:
        zoom = dpi / 72  # 将 DPI 转换为 zoom 因子
        mat = fitz.Matrix(zoom, zoom)
        results = {}
        
        # 遍历每一页
        for page_num in tqdm(range(len(pdf_document))):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat)
            
            # 将图像数据从PyMuPDF的Pixmap转换为OpenCV格式
            img = cv2.imdecode(np.frombuffer(pix.tobytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            
            # 运行OCR
            try:
                ocr_result = runV2(img)
                results[f"page_{page_num + 1}"] = ocr_result
                print(f"OCR results for Page {page_num + 1} processed and stored.")
            except Exception as e:
                print(f"Failed to process OCR for Page {page_num + 1}: {e}")
            

    # 将OCR结果保存到文件
    results_path = os.path.join(output_folder, "ocr_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"All OCR results have been saved to {results_path}.")

# 调用函数示例
pdf_to_ocr(r"D:\pdf2text\1.pdf", "output_folder_path")
