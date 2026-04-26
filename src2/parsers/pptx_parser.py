"""
PPTX文档解析器
使用python-pptx和ZIP解析方法
"""

import os
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
import re
from typing import Dict, Any, List
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from .base_parser import BaseParser


class PPTXParser(BaseParser):
    """PPTX文档解析器"""
    
    def __init__(self, logger=None):
        """
        初始化PPTX解析器
        
        Args:
            logger: 日志记录器
        """
        super().__init__(logger)
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析PPTX文档
        
        Args:
            file_path: PPTX文件路径
            
        Returns:
            解析结果
        """
        if not self.validate_file(file_path):
            self.log_error(f"文件不存在或无效: {file_path}")
            return {"success": False, "error": "文件无效"}
        
        try:
            self.log_info(f"开始解析PPTX: {file_path}")
            
            prs = Presentation(file_path)
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            # 提取所有幻灯片
            pages = []
            for slide_num in range(len(prs.slides)):
                self.log_info(f"解析第 {slide_num + 1}/{len(prs.slides)} 张幻灯片")
                page_content = self.extract_page_content(file_path, slide_num)
                pages.append(page_content)
            
            # 提取图像（使用ZIP方法）
            image_mapping = self.extract_all_images_via_zip(file_path)
            
            # 获取元数据
            metadata = self.get_metadata(file_path)
            metadata.update({
                "total_slides": len(prs.slides),
                "image_mapping": image_mapping
            })
            
            result = {
                "success": True,
                "filename": filename,
                "pages": pages,
                "metadata": metadata
            }
            
            self.log_info(f"PPTX解析完成: {filename}")
            return result
            
        except Exception as e:
            self.log_error(f"PPTX解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def get_page_count(self, file_path: str) -> int:
        """获取PPTX幻灯片数"""
        try:
            prs = Presentation(file_path)
            return len(prs.slides)
        except Exception as e:
            self.log_error(f"获取幻灯片数失败: {e}")
            return 0
    
    def extract_page_content(self, file_path: str, page_num: int) -> Dict[str, Any]:
        """
        提取PPTX幻灯片内容
        
        Args:
            file_path: PPTX文件路径
            page_num: 幻灯片编号
            
        Returns:
            幻灯片内容
        """
        try:
            prs = Presentation(file_path)
            slide = prs.slides[page_num]
            
            # 提取文本
            text_parts = []
            tables = []
            
            for shape in slide.shapes:
                # 提取文本
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            text_parts.append(run.text)
                
                # 标记表格（具体提取由TableExtractor处理）
                if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                    tables.append({
                        "shape": shape,
                        "position": (shape.left, shape.top, shape.width, shape.height)
                    })
            
            text = '\n'.join(text_parts)
            
            result = {
                "page_num": page_num,
                "text": text,
                "images": [],  # 图像通过ZIP方法提取
                "tables": tables,
                "raw_slide": slide
            }
            
            return result
            
        except Exception as e:
            self.log_error(f"提取幻灯片内容失败 (第{page_num}张): {e}")
            return {
                "page_num": page_num,
                "text": "",
                "images": [],
                "tables": [],
                "raw_slide": None,
                "error": str(e)
            }
    
    def extract_all_images_via_zip(self, file_path: str) -> Dict[int, List[str]]:
        """
        通过ZIP方法提取所有图像
        
        Args:
            file_path: PPTX文件路径
            
        Returns:
            {幻灯片编号: [图像路径列表]}
        """
        self.log_info("使用ZIP方法提取图像")
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        slide_image_mapping = {}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 解压PPTX
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                media_dir = os.path.join(temp_dir, "ppt", "media")
                slides_dir = os.path.join(temp_dir, "ppt", "slides")
                rels_dir = os.path.join(temp_dir, "ppt", "slides", "_rels")
                
                if not os.path.exists(media_dir):
                    self.log_info("未找到media目录")
                    return slide_image_mapping
                
                # 遍历所有幻灯片
                if os.path.exists(slides_dir):
                    for slide_file in os.listdir(slides_dir):
                        if slide_file.startswith("slide") and slide_file.endswith(".xml"):
                            slide_num = self._extract_slide_number(slide_file)
                            if slide_num is None:
                                continue
                            
                            # 解析关系文件
                            rels_file = os.path.join(rels_dir, f"{slide_file}.rels")
                            if not os.path.exists(rels_file):
                                continue
                            
                            # 获取图像文件
                            image_files = self._get_slide_images(rels_file, media_dir)
                            
                            # 复制图像到输出目录
                            output_images = []
                            for idx, img_file in enumerate(image_files):
                                file_ext = os.path.splitext(img_file)[1].lower()
                                output_path = f"output/images/{filename}_slide_{slide_num}_img_{idx+1}{file_ext}"
                                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                                
                                shutil.copy2(img_file, output_path)
                                
                                # 检查是否为矢量图，尝试转换为PNG
                                if file_ext in ['.emf', '.wmf', '.svg']:
                                    converted_path = self._convert_vector_to_png(output_path)
                                    if converted_path:
                                        # 转换成功，使用PNG文件
                                        output_images.append(converted_path)
                                    else:
                                        # 转换失败，跳过该图像（不添加到列表）
                                        self.log_warning(f"跳过无法转换的矢量图: {output_path}")
                                else:
                                    # 普通图像格式，直接添加
                                    output_images.append(output_path)
                            
                            if output_images:
                                slide_image_mapping[slide_num] = output_images
                
                self.log_info(f"成功提取 {sum(len(imgs) for imgs in slide_image_mapping.values())} 张图像")
                
            except Exception as e:
                self.log_error(f"ZIP方法提取图像失败: {e}")
        
        return slide_image_mapping
    
    def _extract_slide_number(self, slide_filename: str) -> int:
        """从文件名提取幻灯片编号"""
        match = re.search(r'slide(\d+)', slide_filename)
        if match:
            return int(match.group(1))
        return None
    
    def _get_slide_images(self, rels_file: str, media_dir: str) -> List[str]:
        """获取幻灯片的图像文件"""
        images = []
        
        try:
            tree = ET.parse(rels_file)
            root = tree.getroot()
            
            # 查找图像关系
            for rel in root.findall(".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"):
                rel_type = rel.get('Type', '')
                if 'image' in rel_type.lower():
                    target = rel.get('Target', '')
                    # 解析相对路径
                    img_filename = os.path.basename(target)
                    img_path = os.path.join(media_dir, img_filename)
                    
                    if os.path.exists(img_path):
                        images.append(img_path)
        
        except Exception as e:
            self.log_error(f"解析关系文件失败: {e}")
        
        return images
    
    def _convert_vector_to_png(self, image_path: str) -> str:
        """
        尝试将矢量图（EMF/WMF/SVG）转换为PNG
        
        Args:
            image_path: 原始图片路径
            
        Returns:
            转换后的PNG路径，如果转换失败返回None
        """
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in ['.emf', '.wmf', '.svg']:
            return image_path  # 不需要转换
        
        # 生成PNG路径
        png_path = image_path.rsplit('.', 1)[0] + '_converted.png'
        
        self.log_info(f"检测到矢量图格式: {file_ext}，尝试转换为PNG")
        
        # 方法1: 尝试使用ImageMagick
        try:
            import subprocess
            # 检查是否安装了ImageMagick
            try:
                subprocess.run(['magick', '-version'], capture_output=True, check=True)
                has_imagemagick = True
            except:
                has_imagemagick = False
            
            if has_imagemagick:
                self.log_info(f"  使用ImageMagick转换: {file_ext} -> PNG")
                cmd = ['magick', 'convert', image_path, '-background', 'white', '-alpha', 'remove', png_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(png_path):
                    self.log_info(f"  转换成功: {png_path}")
                    return png_path
        except Exception as e:
            self.log_warning(f"ImageMagick转换失败: {e}")
        
        # 方法2: 尝试使用wand库
        try:
            from wand.image import Image as WandImage
            self.log_info(f"  使用Wand库转换: {file_ext} -> PNG")
            
            with WandImage(filename=image_path, resolution=300) as img:
                img.background_color = 'white'
                img.alpha_channel = 'remove'
                img.format = 'png'
                img.save(filename=png_path)
            
            if os.path.exists(png_path):
                self.log_info(f"  Wand转换成功: {png_path}")
                return png_path
                
        except ImportError:
            self.log_info("  提示: 安装wand库可以支持矢量图转换 (pip install wand)")
        except Exception as e:
            self.log_warning(f"  Wand转换失败: {e}")
        
        # 转换失败
        self.log_warning(f"矢量图转换失败，跳过该图像: {image_path}")
        return None
    
    def extract_slide_text_only(self, file_path: str, slide_num: int) -> str:
        """
        仅提取幻灯片文本
        
        Args:
            file_path: PPTX文件路径
            slide_num: 幻灯片编号
            
        Returns:
            文本内容
        """
        try:
            prs = Presentation(file_path)
            slide = prs.slides[slide_num]
            
            text_parts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        for run in paragraph.runs:
                            text_parts.append(run.text)
            
            return '\n'.join(text_parts)
        except Exception as e:
            self.log_error(f"提取文本失败 (第{slide_num}张): {e}")
            return ""
