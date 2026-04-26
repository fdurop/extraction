"""
PDF文档解析器
使用PyMuPDF (fitz) 解析PDF文件
"""

import os
import fitz  # PyMuPDF
from typing import Dict, Any, List
from .base_parser import BaseParser


class PDFParser(BaseParser):
    """PDF文档解析器"""
    
    def __init__(self, logger=None):
        """
        初始化PDF解析器
        
        Args:
            logger: 日志记录器
        """
        super().__init__(logger)
        self.current_doc = None
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析PDF文档
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            解析结果
        """
        if not self.validate_file(file_path):
            self.log_error(f"文件不存在或无效: {file_path}")
            return {"success": False, "error": "文件无效"}
        
        try:
            self.log_info(f"开始解析PDF: {file_path}")
            
            doc = fitz.open(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # 提取所有页面
            pages = []
            for page_num in range(len(doc)):
                self.log_info(f"解析第 {page_num + 1}/{len(doc)} 页")
                page_content = self.extract_page_content(file_path, page_num)
                pages.append(page_content)
            
            # 获取元数据
            metadata = self.get_metadata(file_path)
            metadata.update({
                "total_pages": len(doc),
                "pdf_metadata": doc.metadata
            })
            
            doc.close()
            
            result = {
                "success": True,
                "filename": filename,
                "pages": pages,
                "metadata": metadata
            }
            
            self.log_info(f"PDF解析完成: {filename}")
            return result
            
        except Exception as e:
            self.log_error(f"PDF解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def get_page_count(self, file_path: str) -> int:
        """获取PDF页数"""
        try:
            doc = fitz.open(file_path)
            count = len(doc)
            doc.close()
            return count
        except Exception as e:
            self.log_error(f"获取页数失败: {e}")
            return 0
    
    def extract_page_content(self, file_path: str, page_num: int) -> Dict[str, Any]:
        """
        提取PDF页面内容
        
        Args:
            file_path: PDF文件路径
            page_num: 页码
            
        Returns:
            页面内容
        """
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            
            # 提取文本
            text = page.get_text()
            
            # 提取图像
            images = []
            image_list = page.get_images(full=True)
            
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # 保存图像
                img_path = f"output/images/{filename}_p{page_num+1}_img{img_index+1}.{base_image['ext']}"
                os.makedirs(os.path.dirname(img_path), exist_ok=True)
                
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images.append(img_path)
            
            result = {
                "page_num": page_num,
                "text": text,
                "images": images,
                "raw_page": page,
                "page_size": {"width": page.rect.width, "height": page.rect.height}
            }
            
            doc.close()
            return result
            
        except Exception as e:
            self.log_error(f"提取页面内容失败 (页{page_num}): {e}")
            return {
                "page_num": page_num,
                "text": "",
                "images": [],
                "raw_page": None,
                "error": str(e)
            }
    
    def extract_page_images_only(self, file_path: str, page_num: int, output_dir: str = "output/images") -> List[str]:
        """
        仅提取页面图像（不提取文本）
        
        Args:
            file_path: PDF文件路径
            page_num: 页码
            output_dir: 输出目录
            
        Returns:
            图像路径列表
        """
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            image_list = page.get_images(full=True)
            
            images = []
            filename = os.path.splitext(os.path.basename(file_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                img_path = f"{output_dir}/{filename}_p{page_num+1}_img{img_index+1}.{base_image['ext']}"
                
                with open(img_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                images.append(img_path)
            
            doc.close()
            return images
            
        except Exception as e:
            self.log_error(f"提取图像失败 (页{page_num}): {e}")
            return []
    
    def extract_page_text_only(self, file_path: str, page_num: int) -> str:
        """
        仅提取页面文本
        
        Args:
            file_path: PDF文件路径
            page_num: 页码
            
        Returns:
            文本内容
        """
        try:
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            text = page.get_text()
            doc.close()
            return text
        except Exception as e:
            self.log_error(f"提取文本失败 (页{page_num}): {e}")
            return ""
