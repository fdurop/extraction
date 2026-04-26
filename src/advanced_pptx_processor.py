import os
import json
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import pandas as pd
import re
import subprocess
from logger_config import get_logger



class AdvancedPPTProcessor:
    def __init__(self, preprocessor, fast_mode=False):
        """
        初始化高级PPTX处理器
        
        Args:
            preprocessor: MultimodalPreprocessor实例，用于重用输出目录与结果记录
            fast_mode: 快速模式，跳过耗时的CLIP描述生成
        """
        self.logger = get_logger("AdvancedPPTProcessor")
        self.logger.info("初始化高级PPTX处理器")
        
        self.preprocessor = preprocessor
        self.fast_mode = fast_mode
        self.output_text_dir = "output/text"
        self.output_table_dir = "output/tables"
        self.output_img_dir = "output/images"
        
        # 跟踪识别失败的项目（用于后续人工补充）
        self.failed_recognitions = []
        
        # 跟踪缺少专业内容的图片（用于人工补充）
        self.images_without_professional_content = []
        
        # 专业术语库（从所有描述中提取）
        self.professional_terms = set()
        
        # 确保输出目录存在
        os.makedirs(self.output_text_dir, exist_ok=True)
        os.makedirs(self.output_table_dir, exist_ok=True)
        os.makedirs(self.output_img_dir, exist_ok=True)
        
        self.logger.info(f"快速模式: {fast_mode}")

    def extract_all_images_via_zip(self, file_path):
        """
        通过ZIP解压和XML解析提取PPTX中的所有图片
        
        Args:
            file_path: PPTX文件路径
            
        Returns:
            dict: 包含幻灯片到图片映射关系的字典
        """
        print(f"开始通过ZIP方式提取图片: {file_path}")
        
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        slide_image_mapping = {}
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. 解压PPTX文件
                print("正在解压PPTX文件...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # 2. 找到媒体目录
                media_dir = os.path.join(temp_dir, "ppt", "media")
                slides_dir = os.path.join(temp_dir, "ppt", "slides")
                rels_dir = os.path.join(temp_dir, "ppt", "slides", "_rels")
                
                if not os.path.exists(media_dir):
                    print("未找到media目录，可能没有图片")
                    return slide_image_mapping
                
                print(f"找到media目录: {media_dir}")
                print(f"媒体文件: {os.listdir(media_dir)}")
                
                # 3. 遍历所有幻灯片XML文件
                if os.path.exists(slides_dir):
                    for slide_file in os.listdir(slides_dir):
                        if slide_file.startswith("slide") and slide_file.endswith(".xml"):
                            slide_num = self._extract_slide_number(slide_file)
                            if slide_num is None:
                                continue
                                
                            print(f"处理幻灯片 {slide_num}: {slide_file}")
                            
                            # 解析幻灯片XML获取图片关系ID
                            slide_xml_path = os.path.join(slides_dir, slide_file)
                            image_rids = self._parse_slide_xml_for_images(slide_xml_path)
                            
                            if image_rids:
                                print(f"幻灯片 {slide_num} 中找到图片关系ID: {image_rids}")
                                
                                # 解析关系文件获取实际文件名
                                rels_file = slide_file + ".rels"
                                rels_path = os.path.join(rels_dir, rels_file)
                                
                                if os.path.exists(rels_path):
                                    image_files = self._parse_rels_file(rels_path, image_rids)
                                    
                                    if image_files:
                                        slide_image_mapping[slide_num] = image_files
                                        print(f"幻灯片 {slide_num} 映射到图片: {image_files}")
                                        
                                        # 复制图片到输出目录
                                        self._copy_images_to_output(media_dir, image_files, 
                                                                  base_filename, slide_num)
                
                print(f"图片提取完成，映射关系: {slide_image_mapping}")
                
            except Exception as e:
                print(f"ZIP方式图片提取失败: {e}")
                import traceback
                traceback.print_exc()
        
        return slide_image_mapping

    def _extract_slide_number(self, slide_filename):
        """从幻灯片文件名中提取编号"""
        try:
            # slide1.xml -> 1
            import re
            match = re.search(r'slide(\d+)\.xml', slide_filename)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def _parse_slide_xml_for_images(self, slide_xml_path):
        """
        解析幻灯片XML文件，查找图片引用
        
        Args:
            slide_xml_path: 幻灯片XML文件路径
            
        Returns:
            list: 图片关系ID列表
        """
        image_rids = []
        
        try:
            tree = ET.parse(slide_xml_path)
            root = tree.getroot()
            
            # 定义命名空间
            namespaces = {
                'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
                'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'
            }
            
            # 查找所有a:blip元素（图片引用）
            blip_elements = root.findall('.//a:blip', namespaces)
            
            for blip in blip_elements:
                embed_attr = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                if embed_attr:
                    image_rids.append(embed_attr)
                    print(f"找到图片引用ID: {embed_attr}")
            
        except Exception as e:
            print(f"解析幻灯片XML失败 {slide_xml_path}: {e}")
        
        return image_rids

    def _parse_rels_file(self, rels_path, image_rids):
        """
        解析关系文件，获取关系ID到文件名的映射
        
        Args:
            rels_path: 关系文件路径
            image_rids: 图片关系ID列表
            
        Returns:
            list: 对应的图片文件名列表
        """
        image_files = []
        
        try:
            tree = ET.parse(rels_path)
            root = tree.getroot()
            
            # 定义命名空间
            namespaces = {
                'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'
            }
            
            # 查找所有关系
            for relationship in root.findall('.//rel:Relationship', namespaces):
                rel_id = relationship.get('Id')
                target = relationship.get('Target')
                rel_type = relationship.get('Type')
                
                # 检查是否是图片关系
                if (rel_id in image_rids and 
                    target and 
                    rel_type and 
                    'image' in rel_type.lower()):
                    
                    # 提取文件名 (../media/image1.png -> image1.png)
                    filename = os.path.basename(target)
                    image_files.append(filename)
                    print(f"关系映射: {rel_id} -> {filename}")
            
        except Exception as e:
            print(f"解析关系文件失败 {rels_path}: {e}")
        
        return image_files

    def _convert_vector_to_png(self, image_path):
        """
        尝试将矢量图（EMF/WMF/SVG）转换为PNG
        
        Args:
            image_path: 原始图片路径
            
        Returns:
            str: 转换后的PNG路径，如果转换失败返回None
        """
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in ['.emf', '.wmf', '.svg']:
            return image_path  # 不需要转换
        
        # 生成PNG路径
        png_path = image_path.rsplit('.', 1)[0] + '_converted.png'
        
        # 方法1: 尝试使用Pillow + pillow-wmf插件
        try:
            from PIL import Image
            import subprocess
            
            # 检查是否安装了ImageMagick
            try:
                subprocess.run(['magick', '-version'], capture_output=True, check=True)
                has_imagemagick = True
            except:
                has_imagemagick = False
            
            if has_imagemagick:
                # 使用ImageMagick转换
                print(f"  尝试使用ImageMagick转换: {file_ext} -> PNG")
                cmd = ['magick', 'convert', image_path, '-background', 'white', '-alpha', 'remove', png_path]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and os.path.exists(png_path):
                    print(f"  [OK] 转换成功: {png_path}")
                    return png_path
                else:
                    print(f"  ImageMagick转换失败: {result.stderr}")
            
        except Exception as e:
            print(f"  转换失败: {e}")
        
        # 方法2: 尝试使用wand库（ImageMagick的Python绑定）
        try:
            from wand.image import Image as WandImage
            print(f"  尝试使用Wand库转换: {file_ext} -> PNG")
            
            with WandImage(filename=image_path, resolution=300) as img:
                img.background_color = 'white'
                img.alpha_channel = 'remove'
                img.format = 'png'
                img.save(filename=png_path)
            
            if os.path.exists(png_path):
                print(f"  [OK] Wand转换成功: {png_path}")
                return png_path
                
        except ImportError:
            print(f"  提示: 安装wand库可以支持矢量图转换 (pip install wand)")
        except Exception as e:
            print(f"  Wand转换失败: {e}")
        
        return None

    def _generate_image_description_with_deepseek(self, image_path):
        """
        使用DeepSeek-VL生成图片描述
        
        Args:
            image_path: 图片路径
            
        Returns:
            str: 描述文件路径
        """
        descriptions = []
        
        # 检查文件格式，尝试转换矢量图
        vector_formats = ['.emf', '.wmf', '.svg']
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext in vector_formats:
            print(f"检测到矢量图格式: {image_path} ({file_ext})")
            converted_path = self._convert_vector_to_png(image_path)
            
            if converted_path and converted_path != image_path:
                # 转换成功，使用转换后的PNG
                image_path = converted_path
                print(f"  使用转换后的PNG进行分析")
            else:
                # 转换失败，跳过
                print(f"  无法转换，跳过此图片")
                descriptions.append({
                    "source": "skipped",
                    "description": f"不支持的图片格式: {file_ext}（转换失败，建议在PPT中手动转换为PNG）"
                })
                # 保存并返回
                base, _ = os.path.splitext(os.path.basename(image_path))
                desc_path = os.path.join("output", "images", f"{base}_desc.json")
                try:
                    with open(desc_path, "w", encoding="utf-8") as f:
                        json.dump({
                            "image_path": image_path,
                            "descriptions": descriptions
                        }, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"保存图片描述失败: {e}")
                return desc_path
        
        # 使用DeepSeek-VL处理图片
        general_description = None
        if self.preprocessor.deepseek_vl is not None:
            try:
                print(f"使用DeepSeek-VL生成图片描述: {image_path}")
                description = self.preprocessor.deepseek_vl.describe_image(image_path)
                if description:
                    general_description = description
                    descriptions.append({
                        "source": "deepseek_vl",
                        "type": "general_description",
                        "description": description
                    })
                    
                    # 提取专业术语并加入术语库
                    self._extract_and_add_professional_terms(description)
                    
            except Exception as e:
                print(f"DeepSeek-VL描述生成失败: {e}")
                descriptions.append({
                    "source": "error",
                    "description": f"处理失败: {str(e)}"
                })
        else:
            print("DeepSeek-VL不可用，跳过图片描述生成")
        
        # 智能识别：根据描述内容判断是否需要进一步识别公式或代码
        if general_description and self.preprocessor.deepseek_vl is not None:
            # 检测公式关键词
            formula_keywords = ['公式', '方程', '数学', '计算式', 'formula', 'equation', 'math', 'LaTeX', '积分', '微分', '求导', '函数']
            has_formula = any(keyword in general_description for keyword in formula_keywords)
            
            # 检测代码关键词
            code_keywords = ['代码', '程序', 'code', 'program', 'function', 'def ', 'class ', 'import', 'return', '编程', '算法']
            has_code = any(keyword in general_description for keyword in code_keywords)
            
            # 如果检测到公式，进行专门的公式识别
            if has_formula:
                formula_recognized = False
                try:
                    print(f"  ↳ 检测到公式内容，启动专门公式识别...")
                    formula_result = self.preprocessor.deepseek_vl.recognize_formula(image_path)
                    if formula_result and formula_result.get("latex"):
                        descriptions.append({
                            "source": "deepseek_vl",
                            "type": "formula_recognition",
                            "latex": formula_result.get("latex", ""),
                            "formula_details": formula_result
                        })
                        print(f"  [OK] 公式识别完成")
                        formula_recognized = True
                        
                        # 将公式保存到 output/formulas/ 文件夹
                        try:
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            formula_json_path = os.path.join("output", "formulas", f"{base_name}_formula.json")
                            
                            formula_output = {
                                "source_image": image_path,
                                "extraction_method": "deepseek_vl_智能识别",
                                "latex": formula_result.get("latex", ""),
                                "formula_name": formula_result.get("名称", ""),
                                "meaning": formula_result.get("含义", ""),
                                "variables": formula_result.get("变量", ""),
                                "conditions": formula_result.get("条件", ""),
                                "discipline": formula_result.get("学科", ""),
                                "related_concepts": formula_result.get("关联", ""),
                                "full_details": formula_result
                            }
                            
                            with open(formula_json_path, "w", encoding="utf-8") as f:
                                json.dump(formula_output, f, ensure_ascii=False, indent=2)
                            
                            print(f"  → 公式已保存至: {formula_json_path}")
                        except Exception as save_error:
                            print(f"  [WARN] 保存公式文件失败: {save_error}")
                    else:
                        print(f"  ✗ 公式识别未返回有效结果")
                        
                except Exception as e:
                    print(f"  ✗ 公式识别失败: {e}")
                
                # 如果识别失败，记录到待人工补充列表
                if not formula_recognized:
                    self.failed_recognitions.append({
                        "type": "formula",
                        "image_path": image_path,
                        "description": general_description[:200] if general_description else "未获取到描述",
                        "reason": "自动识别失败或未返回有效结果"
                    })
            
            # 如果检测到代码，进行专门的代码识别
            if has_code:
                code_recognized = False
                try:
                    print(f"  ↳ 检测到代码内容，启动专门代码识别...")
                    code_result = self.preprocessor.deepseek_vl.recognize_code(image_path)
                    if code_result and code_result.get("code"):
                        descriptions.append({
                            "source": "deepseek_vl",
                            "type": "code_recognition",
                            "code": code_result.get("code", ""),
                            "language": code_result.get("language", ""),
                            "code_details": code_result
                        })
                        print(f"  [OK] 代码识别完成")
                        code_recognized = True
                        
                        # 将代码保存到 output/code/ 文件夹
                        try:
                            base_name = os.path.splitext(os.path.basename(image_path))[0]
                            language = code_result.get("language", "").lower()
                            
                            # 根据语言确定文件扩展名
                            language_extensions = {
                                "python": ".py",
                                "java": ".java",
                                "javascript": ".js",
                                "c++": ".cpp",
                                "c": ".c",
                                "go": ".go",
                                "rust": ".rs",
                                "typescript": ".ts",
                                "php": ".php",
                                "ruby": ".rb",
                                "swift": ".swift",
                                "kotlin": ".kt",
                                "matlab": ".m",
                                "r": ".r"
                            }
                            ext = language_extensions.get(language, ".txt")
                            
                            # 保存代码文件
                            code_file_path = os.path.join("output", "code", f"{base_name}_code{ext}")
                            with open(code_file_path, "w", encoding="utf-8") as f:
                                f.write(code_result.get("code", ""))
                            
                            # 保存代码元数据JSON
                            code_json_path = os.path.join("output", "code", f"{base_name}_code_metadata.json")
                            code_metadata = {
                                "source_image": image_path,
                                "extraction_method": "deepseek_vl_智能识别",
                                "code_file": code_file_path,
                                "language": code_result.get("language", ""),
                                "algorithm": code_result.get("算法", ""),
                                "concepts": code_result.get("概念", ""),
                                "functionality": code_result.get("功能", ""),
                                "knowledge_points": code_result.get("知识点", ""),
                                "difficulty": code_result.get("难度", ""),
                                "related_topics": code_result.get("关联", ""),
                                "full_details": code_result
                            }
                            
                            with open(code_json_path, "w", encoding="utf-8") as f:
                                json.dump(code_metadata, f, ensure_ascii=False, indent=2)
                            
                            print(f"  → 代码已保存至: {code_file_path}")
                            print(f"  → 元数据已保存至: {code_json_path}")
                        except Exception as save_error:
                            print(f"  [WARN] 保存代码文件失败: {save_error}")
                    else:
                        print(f"  ✗ 代码识别未返回有效结果")
                        
                except Exception as e:
                    print(f"  ✗ 代码识别失败: {e}")
                
                # 如果识别失败，记录到待人工补充列表
                if not code_recognized:
                    self.failed_recognitions.append({
                        "type": "code",
                        "image_path": image_path,
                        "description": general_description[:200] if general_description else "未获取到描述",
                        "reason": "自动识别失败或未返回有效结果"
                    })
        
        # 检查图片是否缺少专业内容（没有公式、代码、也没有专业术语）
        if general_description:
            has_professional_content = has_formula or has_code or self._contains_professional_terms(general_description)
            
            if not has_professional_content:
                print(f"  [WARN] 图片缺少专业内容，标记为需要人工补充")
                self.images_without_professional_content.append({
                    "image_path": image_path,
                    "description": general_description[:300] if general_description else "未获取到描述",
                    "reason": "未检测到公式、代码或专业术语"
                })
        
        # 保存描述到JSON
        base, _ = os.path.splitext(os.path.basename(image_path))
        desc_path = os.path.join("output", "images", f"{base}_desc.json")
        
        try:
            with open(desc_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image_path": image_path,
                    "descriptions": descriptions,
                    "智能识别": {
                        "检测到公式": any(d.get("type") == "formula_recognition" for d in descriptions),
                        "检测到代码": any(d.get("type") == "code_recognition" for d in descriptions)
                    }
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存图片描述失败: {e}")
        
        return desc_path

    def _recognize_formula_with_deepseek(self, formula_image_path):
        """
        使用DeepSeek-VL识别公式图片
        
        Args:
            formula_image_path: 公式图片路径
            
        Returns:
            dict: 包含latex和description的字典
        """
        # 检查DeepSeek-VL是否可用
        if self.preprocessor.deepseek_vl is None:
            print("DeepSeek-VL不可用，无法识别公式")
            return None
        
        try:
            print(f"使用DeepSeek-VL识别公式: {formula_image_path}")
            result = self.preprocessor.deepseek_vl.recognize_formula(formula_image_path)
            if result and result.get("latex"):
                print(f"  公式识别成功: {result['latex'][:100]}...")
                return result
            else:
                print(f"  公式识别未返回LaTeX结果")
                return None
        except Exception as e:
            print(f"DeepSeek-VL公式识别失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _copy_images_to_output(self, media_dir, image_files, base_filename, slide_num):
        """
        将图片复制到输出目录并生成描述
        
        Args:
            media_dir: 媒体文件源目录
            image_files: 图片文件名列表
            base_filename: 基础文件名
            slide_num: 幻灯片编号
        """
        for idx, image_file in enumerate(image_files, 1):
            try:
                source_path = os.path.join(media_dir, image_file)
                
                if os.path.exists(source_path):
                    # 生成输出文件名
                    file_ext = os.path.splitext(image_file)[1]
                    output_filename = f"{base_filename}_slide_{slide_num}_img_{idx}_zip{file_ext}"
                    output_path = os.path.join(self.output_img_dir, output_filename)
                    
                    # 复制图片
                    shutil.copy2(source_path, output_path)
                    print(f"复制图片: {source_path} -> {output_path}")
                    
                    # 使用DeepSeek-VL生成图片描述（根据模式决定是否生成）
                    desc_path = None
                    if not self.fast_mode:
                        try:
                            desc_path = self._generate_image_description_with_deepseek(output_path)
                        except Exception as e:
                            print(f"图片描述生成出错: {output_path}, {e}")
                    else:
                        print("快速模式：跳过图片描述生成")
                    
                    # 记录到结果中
                    self.preprocessor.results.append({
                        "type": "ppt_image_zip",
                        "page": slide_num,
                        "file": output_path,
                        "description_file": desc_path,
                        "extraction_method": "zip_xml_parsing",
                        "original_filename": image_file
                    })
                    
                else:
                    print(f"源图片文件不存在: {source_path}")
                    
            except Exception as e:
                print(f"复制图片失败 {image_file}: {e}")

    def extract_and_convert_equations(self, slide, slide_number):
        """
        处理幻灯片中的公式
        
        Args:
            slide: python-pptx的Slide对象
            slide_number: 幻灯片编号
            
        Returns:
            list: 包含公式信息的列表
        """
        equations = []
        
        for shape_index, shape in enumerate(slide.shapes):
            try:
                # 检查形状是否包含文本框
                if hasattr(shape, 'text_frame') and shape.text_frame:
                    # 获取形状的XML内容
                    shape_xml = self._get_shape_xml(shape)
                    if shape_xml:
                        # 检查是否包含OMML公式标签
                        omml_content = self._extract_omml_from_xml(shape_xml)
                        if omml_content:
                            print(f"在幻灯片 {slide_number} 形状 {shape_index} 中发现OMML公式")
                            
                            # 尝试转换OMML到LaTeX
                            latex_content = self._convert_omml_to_latex(omml_content)
                            
                            equation_info = {
                                "slide_number": slide_number,
                                "shape_index": shape_index,
                                "type": "omml_formula",
                                "original_omml": omml_content[:500] + "..." if len(omml_content) > 500 else omml_content,
                                "latex": latex_content,
                                "conversion_success": latex_content is not None
                            }
                            
                            equations.append(equation_info)
                            
                            # 添加到结果中
                            self.preprocessor.results.append({
                                "type": "formula",
                                "page": slide_number,
                                "formula_type": "omml",
                                "latex": latex_content,
                                "source": f"slide_{slide_number}_shape_{shape_index}",
                                "conversion_method": "omml_to_latex"
                            })
                            
                            continue
                
                # 如果没有找到OMML，检查是否为可能的公式图片
                if self._is_potential_formula_image(shape):
                    print(f"在幻灯片 {slide_number} 形状 {shape_index} 中发现潜在公式图片")
                    
                    # 使用图片处理流程处理公式图片
                    formula_image_path = self._process_formula_image(shape, slide_number, shape_index)
                    
                    if formula_image_path:
                        # 使用DeepSeek-VL识别公式
                        latex_result = self._recognize_formula_with_deepseek(formula_image_path)
                        
                        equation_info = {
                            "slide_number": slide_number,
                            "shape_index": shape_index,
                            "type": "formula_image",
                            "image_path": formula_image_path,
                            "latex": latex_result.get("latex", None) if latex_result else None,
                            "description": latex_result.get("description", None) if latex_result else None,
                            "conversion_success": bool(latex_result and latex_result.get("latex"))
                        }
                        
                        equations.append(equation_info)
                        
                        # 添加到结果中
                        self.preprocessor.results.append({
                            "type": "formula",
                            "page": slide_number,
                            "formula_type": "image",
                            "image_path": formula_image_path,
                            "latex": equation_info["latex"],
                            "source": f"slide_{slide_number}_shape_{shape_index}",
                            "conversion_method": "deepseek_vl"
                        })
                        
            except Exception as e:
                print(f"处理幻灯片 {slide_number} 形状 {shape_index} 时出错: {e}")
                continue
        
        return equations

    def _get_shape_xml(self, shape):
        """获取形状的XML内容"""
        try:
            # 尝试获取形状的内部XML
            if hasattr(shape, '_element'):
                return ET.tostring(shape._element, encoding='unicode')
        except Exception as e:
            print(f"获取形状XML失败: {e}")
        return None

    def _extract_omml_from_xml(self, xml_string):
        """从XML中提取OMML内容"""
        try:
            # 查找OMML数学标签
            omml_patterns = [
                r'<m:oMath[^>]*>.*?</m:oMath>',
                r'<m:oMathPara[^>]*>.*?</m:oMathPara>',
                r'<math[^>]*>.*?</math>'  # 也检查标准MathML
            ]
            
            for pattern in omml_patterns:
                matches = re.findall(pattern, xml_string, re.DOTALL | re.IGNORECASE)
                if matches:
                    return matches[0]
                    
        except Exception as e:
            print(f"提取OMML失败: {e}")
        return None

    def _convert_omml_to_latex(self, omml_content):
        """将OMML转换为LaTeX"""
        try:
            # 方法1: 尝试使用pandoc
            latex_result = self._convert_via_pandoc(omml_content)
            if latex_result:
                return latex_result
                
            # 方法2: 简单的文本替换作为备选方案
            latex_result = self._simple_omml_to_latex(omml_content)
            if latex_result:
                return latex_result
                
        except Exception as e:
            print(f"OMML转LaTeX失败: {e}")
        
        return None

    def _convert_via_pandoc(self, omml_content):
        """使用pandoc转换OMML到LaTeX"""
        try:
            # 检查pandoc是否可用
            subprocess.run(['pandoc', '--version'], 
                         capture_output=True, check=True)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
                temp_file.write(f'<root>{omml_content}</root>')
                temp_file_path = temp_file.name
            
            try:
                # 使用pandoc转换
                result = subprocess.run([
                    'pandoc', 
                    '-f', 'docx',
                    '-t', 'latex',
                    temp_file_path
                ], capture_output=True, text=True, check=True)
                
                return result.stdout.strip()
                
            finally:
                os.unlink(temp_file_path)
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Pandoc不可用，跳过pandoc转换")
        except Exception as e:
            print(f"Pandoc转换失败: {e}")
        
        return None

    def _simple_omml_to_latex(self, omml_content):
        """简单的OMML到LaTeX转换（基本文本替换）"""
        try:
            # 移除XML标签，提取纯文本
            text_content = re.sub(r'<[^>]+>', '', omml_content)
            text_content = text_content.strip()
            
            if not text_content:
                return None
            
            # 基本的数学符号替换
            replacements = {
                '≈': r'\approx',
                '≠': r'\neq',
                '≤': r'\leq',
                '≥': r'\geq',
                '∞': r'\infty',
                'α': r'\alpha',
                'β': r'\beta',
                'γ': r'\gamma',
                'δ': r'\delta',
                'θ': r'\theta',
                'λ': r'\lambda',
                'μ': r'\mu',
                'π': r'\pi',
                'σ': r'\sigma',
                'φ': r'\phi',
                'ω': r'\omega',
                '∑': r'\sum',
                '∫': r'\int',
                '√': r'\sqrt',
                '±': r'\pm',
                '×': r'\times',
                '÷': r'\div'
            }
            
            for symbol, latex in replacements.items():
                text_content = text_content.replace(symbol, latex)
            
            # 包装在数学环境中
            return f"${text_content}$"
            
        except Exception as e:
            print(f"简单转换失败: {e}")
        
        return None

    def _is_potential_formula_image(self, shape):
        """判断形状是否可能是公式图片"""
        try:
            # 检查是否为图片类型
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                return True
            
            # 检查是否为包含复杂路径的形状（可能是矢量公式）
            if shape.shape_type in [MSO_SHAPE_TYPE.FREEFORM, MSO_SHAPE_TYPE.AUTO_SHAPE]:
                return True
            
            # 检查形状大小（小的形状可能是公式）
            if hasattr(shape, 'width') and hasattr(shape, 'height'):
                # 假设公式通常比较小（宽度和高度都小于某个阈值）
                max_formula_size = 200000  # EMU单位
                if shape.width < max_formula_size and shape.height < max_formula_size:
                    return True
                    
        except Exception as e:
            print(f"检查潜在公式图片失败: {e}")
        
        return False

    def _process_formula_image(self, shape, slide_number, shape_index):
        """处理公式图片"""
        try:
            # 如果是图片类型，尝试导出图片
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                image_filename = f"formula_slide_{slide_number}_shape_{shape_index}.png"
                image_path = os.path.join(self.output_img_dir, image_filename)
                
                # 导出图片
                try:
                    image = shape.image
                    image_bytes = image.blob
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    print(f"公式图片已导出: {image_path}")
                    return image_path
                except Exception as export_error:
                    print(f"公式图片导出失败: {export_error}")
                    # 返回None，表示导出失败
                    return None
                
        except Exception as e:
            print(f"处理公式图片失败: {e}")
        
        return None

    def process_pptx_file_advanced(self, file_path):
        """
        高级PPTX处理：结合传统方法和ZIP解析
        
        Args:
            file_path: PPTX文件路径
        """
        print(f"开始高级PPTX处理: {file_path}")
        self.logger.info(f"开始高级PPTX处理: {file_path}")
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # 重置结果记录，为当前PPTX文件单独记录
        current_results = []
        original_results = self.preprocessor.results
        self.preprocessor.results = current_results
        
        try:
            # 1. 使用传统python-pptx方法处理文本和表格
            self.logger.info("开始处理文本和表格（传统方法）")
            self._process_text_and_tables_traditional(file_path, base_filename)
            
            # 2. 使用ZIP方法提取所有图片
            self.logger.info("开始提取图片（ZIP方法）")
            slide_image_mapping = self.extract_all_images_via_zip(file_path)
            self.logger.info(f"成功提取 {sum(len(images) for images in slide_image_mapping.values())} 张图片")
            
            # 3. 生成PPTX专用元数据
            self.logger.info("生成PPTX元数据")
            self._save_pptx_metadata(file_path, base_filename, slide_image_mapping)
            
            print(f"高级PPTX处理完成: {file_path}")
            print(f"PPTX元数据已保存: output/{base_filename}_pptx_metadata.json")
            self.logger.info(f"高级PPTX处理完成: {file_path}")
            
        except Exception as e:
            print(f"高级PPTX处理失败: {e}")
            self.logger.error(f"高级PPTX处理失败: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
        finally:
            # 恢复原始结果列表并合并当前结果
            self.preprocessor.results = original_results
            self.preprocessor.results.extend(current_results)
            self.logger.debug(f"合并结果，共 {len(current_results)} 项")
    
    def interactive_supplement(self):
        """
        交互式补充识别失败的公式和代码
        教师可以手动填写未能自动识别的内容
        """
        if not self.failed_recognitions:
            print("\n[OK] 所有公式和代码都已成功识别，无需人工补充！")
            return
        
        print("\n" + "="*70)
        print("🔍 发现有些内容自动识别失败，需要您的帮助！")
        print("="*70)
        print(f"共发现 {len(self.failed_recognitions)} 项需要人工补充\n")
        
        for idx, item in enumerate(self.failed_recognitions, 1):
            print(f"\n【{idx}/{len(self.failed_recognitions)}】 类型: {item['type']}")
            print(f"图片路径: {item['image_path']}")
            print(f"AI描述: {item['description']}")
            print(f"失败原因: {item['reason']}")
            print("-" * 70)
            
            # 询问是否要补充
            while True:
                choice = input(f"\n是否要补充这项内容？(y=是, n=跳过, v=查看图片): ").strip().lower()
                
                if choice == 'v':
                    # 提示如何查看图片
                    print(f"\n📷 请在文件管理器中打开查看：{item['image_path']}")
                    continue
                elif choice == 'n':
                    print("已跳过")
                    break
                elif choice == 'y':
                    # 根据类型进行不同的补充
                    if item['type'] == 'formula':
                        self._manual_formula_input(item)
                    elif item['type'] == 'code':
                        self._manual_code_input(item)
                    break
                else:
                    print("无效输入，请输入 y、n 或 v")
        
        print("\n" + "="*70)
        print("[OK] 人工补充完成！")
        print("="*70)
    
    def _manual_formula_input(self, item):
        """手动输入公式"""
        print("\n 请输入公式信息（直接按Enter跳过某项）：")
        
        latex = input("  LaTeX公式: ").strip()
        if not latex:
            print("  [WARN] 未输入LaTeX，已跳过")
            return
        
        formula_name = input("  公式名称（可选）: ").strip()
        meaning = input("  物理/数学含义（可选）: ").strip()
        variables = input("  变量说明（可选）: ").strip()
        
        # 保存到文件
        try:
            base_name = os.path.splitext(os.path.basename(item['image_path']))[0]
            formula_json_path = os.path.join("output", "formulas", f"{base_name}_formula_manual.json")
            
            formula_output = {
                "source_image": item['image_path'],
                "extraction_method": "人工补充",
                "latex": latex,
                "formula_name": formula_name,
                "meaning": meaning,
                "variables": variables,
                "original_description": item['description']
            }
            
            with open(formula_json_path, "w", encoding="utf-8") as f:
                json.dump(formula_output, f, ensure_ascii=False, indent=2)
            
            print(f"  [OK] 公式已保存至: {formula_json_path}")
        except Exception as e:
            print(f"  [ERROR] 保存失败: {e}")
    
    def _extract_and_add_professional_terms(self, description):
        """
        从描述中提取专业术语并加入术语库
        
        使用DeepSeek-VL识别专业术语
        """
        if not self.preprocessor.deepseek_vl:
            return
        
        try:
            # 简单的专业术语识别：检测核心概念、学科领域等关键字
            # 从描述中提取可能的专业术语
            if "核心概念：" in description or "学科领域：" in description:
                # 提取核心概念
                import re
                concept_match = re.search(r'核心概念[：:](.*?)(?:\n|$)', description)
                if concept_match:
                    concepts = concept_match.group(1).strip()
                    # 分割专业术语（按逗号、顿号等）
                    terms = re.split(r'[,，、；;]', concepts)
                    for term in terms:
                        term = term.strip()
                        if term and len(term) > 1:  # 过滤单字符
                            self.professional_terms.add(term)
        except Exception as e:
            pass  # 静默失败，不影响主流程
    
    def _contains_professional_terms(self, description):
        """
        检查描述中是否包含专业术语
        
        Args:
            description: 图片描述
            
        Returns:
            bool: 是否包含专业术语
        """
        if not description:
            return False
        
        # 检查是否包含已知的专业术语
        for term in self.professional_terms:
            if term in description:
                return True
        
        # 检查是否包含专业内容的标志性词汇
        professional_indicators = [
            '原理', '示意图', '结构图', '流程图', '系统', '模型', 
            '理论', '定律', '定理', '机制', '架构', '算法',
            '概念', '学科', '技术', '方法', '装置', '设备',
            'principle', 'diagram', 'structure', 'system', 'model',
            'theory', 'mechanism', 'architecture', 'concept'
        ]
        
        return any(indicator in description for indicator in professional_indicators)
    
    def supplement_professional_images(self):
        """
        补充缺少专业内容的图片
        教师手动输入图片的专业含义
        """
        if not self.images_without_professional_content:
            print("\n[OK] 所有图片都包含专业内容，无需补充！")
            return
        
        print("\n" + "="*70)
        print(" 发现一些图片可能缺少专业内容解释")
        print("="*70)
        print(f"共发现 {len(self.images_without_professional_content)} 张图片需要专业解释\n")
        
        for idx, item in enumerate(self.images_without_professional_content, 1):
            print(f"\n【{idx}/{len(self.images_without_professional_content)}】")
            print(f"图片路径: {item['image_path']}")
            print(f"AI描述: {item['description']}")
            print(f"原因: {item['reason']}")
            print("-" * 70)
            
            # 询问是否要补充
            while True:
                choice = input(f"\n是否要添加专业解释？(y=是, n=跳过, v=查看图片): ").strip().lower()
                
                if choice == 'v':
                    print(f"\n📷 请在文件管理器中打开查看：{item['image_path']}")
                    continue
                elif choice == 'n':
                    print("已跳过")
                    break
                elif choice == 'y':
                    self._manual_professional_explanation(item)
                    break
                else:
                    print("无效输入，请输入 y、n 或 v")
        
        print("\n" + "="*70)
        print("[OK] 专业内容补充完成！")
        print("="*70)
    
    def _manual_professional_explanation(self, item):
        """手动输入图片的专业解释"""
        print("\n 请输入图片的专业解释（直接按Enter跳过某项）：")
        
        professional_explanation = input("  专业解释（必填）: ").strip()
        if not professional_explanation:
            print("  [WARN] 未输入专业解释，已跳过")
            return
        
        key_concepts = input("  关键概念（用逗号分隔）: ").strip()
        discipline = input("  学科领域: ").strip()
        knowledge_points = input("  相关知识点: ").strip()
        teaching_purpose = input("  教学目的: ").strip()
        
        # 保存到文件
        try:
            base_name = os.path.splitext(os.path.basename(item['image_path']))[0]
            explanation_path = os.path.join("output", "images", f"{base_name}_professional_manual.json")
            
            explanation_data = {
                "source_image": item['image_path'],
                "extraction_method": "教师人工标注",
                "professional_explanation": professional_explanation,
                "key_concepts": [c.strip() for c in key_concepts.split(',') if c.strip()],
                "discipline": discipline,
                "knowledge_points": knowledge_points,
                "teaching_purpose": teaching_purpose,
                "original_ai_description": item['description']
            }
            
            with open(explanation_path, "w", encoding="utf-8") as f:
                json.dump(explanation_data, f, ensure_ascii=False, indent=2)
            
            print(f"  [OK] 专业解释已保存至: {explanation_path}")
            
            # 将关键概念加入术语库
            if key_concepts:
                for concept in key_concepts.split(','):
                    concept = concept.strip()
                    if concept:
                        self.professional_terms.add(concept)
                        
        except Exception as e:
            print(f"  [ERROR] 保存失败: {e}")
    
    def generate_professional_terms_library(self):
        """
        生成并保存专业术语库
        """
        if not self.professional_terms:
            print("\n[WARN] 未提取到专业术语")
            return
        
        try:
            library_path = os.path.join("output", "professional_terms_library.json")
            
            terms_data = {
                "course_name": "自动提取",
                "total_terms": len(self.professional_terms),
                "terms": sorted(list(self.professional_terms)),
                "extraction_method": "DeepSeek-VL智能识别 + 教师标注",
                "categories": {
                    "automatically_extracted": [t for t in self.professional_terms if len(t) > 1],
                }
            }
            
            with open(library_path, "w", encoding="utf-8") as f:
                json.dump(terms_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n 专业术语库已生成: {library_path}")
            print(f"   共收录 {len(self.professional_terms)} 个专业术语")
            
        except Exception as e:
            print(f"\n[ERROR] 专业术语库生成失败: {e}")
    
    def _manual_code_input(self, item):
        """手动输入代码"""
        print("\n 请输入代码信息：")
        
        language = input("  编程语言: ").strip()
        if not language:
            language = "unknown"
        
        print("  代码内容（输入完成后输入 END 单独一行结束）:")
        code_lines = []
        while True:
            line = input("    ")
            if line.strip() == "END":
                break
            code_lines.append(line)
        
        code_content = "\n".join(code_lines)
        if not code_content.strip():
            print("  [WARN] 未输入代码，已跳过")
            return
        
        functionality = input("  代码功能说明（可选）: ").strip()
        
        # 保存到文件
        try:
            base_name = os.path.splitext(os.path.basename(item['image_path']))[0]
            
            # 确定文件扩展名
            language_extensions = {
                "python": ".py",
                "java": ".java",
                "javascript": ".js",
                "c++": ".cpp",
                "c": ".c",
                "matlab": ".m"
            }
            ext = language_extensions.get(language.lower(), ".txt")
            
            # 保存代码文件
            code_file_path = os.path.join("output", "code", f"{base_name}_code_manual{ext}")
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            # 保存元数据
            code_json_path = os.path.join("output", "code", f"{base_name}_code_manual_metadata.json")
            code_metadata = {
                "source_image": item['image_path'],
                "extraction_method": "人工补充",
                "code_file": code_file_path,
                "language": language,
                "functionality": functionality,
                "original_description": item['description']
            }
            
            with open(code_json_path, "w", encoding="utf-8") as f:
                json.dump(code_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"  [OK] 代码已保存至: {code_file_path}")
            print(f"  [OK] 元数据已保存至: {code_json_path}")
        except Exception as e:
            print(f"  [ERROR] 保存失败: {e}")

    def _process_text_and_tables_traditional(self, file_path, base_filename):
        """使用传统python-pptx方法处理文本和表格"""
        prs = Presentation(file_path)
        
        for slide_index, slide in enumerate(prs.slides, start=1):
            # 处理公式
            equations = self.extract_and_convert_equations(slide, slide_index)
            if equations:
                print(f"在幻灯片 {slide_index} 中找到 {len(equations)} 个公式")
                
                # 保存公式信息到JSON文件
                formulas_json_path = f"output/formulas/{base_filename}_slide_{slide_index}_formulas.json"
                os.makedirs("output/formulas", exist_ok=True)
                
                formulas_output = {
                    "slide_number": slide_index,
                    "source_file": base_filename,
                    "equations_count": len(equations),
                    "equations": equations,
                    "processing_date": str(pd.Timestamp.now())
                }
                
                with open(formulas_json_path, "w", encoding="utf-8") as f:
                    json.dump(formulas_output, f, ensure_ascii=False, indent=2)
                
                print(f"公式信息已保存到: {formulas_json_path}")
            
            # 提取文本
            slide_text_items = []
            for shape in slide.shapes:
                if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                    text_content = []
                    for paragraph in shape.text_frame.paragraphs:
                        runs_text = ''.join(run.text for run in paragraph.runs)
                        text_content.append(runs_text if runs_text else paragraph.text)
                    final_text = "\n".join([t for t in text_content if t is not None])
                    if final_text.strip():
                        slide_text_items.append(final_text.strip())

            if slide_text_items:
                text_output = {
                    "type": "ppt_text",
                    "page": slide_index,
                    "source": f"{base_filename}_slide_{slide_index}",
                    "raw_text": "\n\n".join(slide_text_items)
                }
                text_json_path = f"{self.output_text_dir}/{base_filename}_slide_{slide_index}.json"
                with open(text_json_path, "w", encoding="utf-8") as f:
                    json.dump(text_output, f, ensure_ascii=False, indent=2)
                self.preprocessor.results.append({
                    "type": "ppt_text",
                    "page": slide_index,
                    "file": text_json_path
                })

            # 提取表格（优化版本）
            table_counter = 0
            for shape in slide.shapes:
                if hasattr(shape, "has_table") and shape.has_table:
                    table_counter += 1
                    table = shape.table
                    
                    # 获取表格位置信息（增强功能）
                    table_position = {
                        "left": float(shape.left.inches) if shape.left else 0,
                        "top": float(shape.top.inches) if shape.top else 0,
                        "width": float(shape.width.inches) if shape.width else 0,
                        "height": float(shape.height.inches) if shape.height else 0
                    }
                    
                    # 提取表格数据
                    data_matrix = []
                    for row in table.rows:
                        row_values = []
                        for cell in row.cells:
                            # 优化：使用 text_frame.text 获取纯文本
                            try:
                                if cell.text_frame and cell.text_frame.text:
                                    cell_text = cell.text_frame.text.strip()
                                else:
                                    cell_text = cell.text.strip() if cell.text else ""
                            except Exception as e:
                                print(f"提取单元格文本失败: {e}")
                                cell_text = ""
                            row_values.append(cell_text)
                        data_matrix.append(row_values)
                    
                    # 检查是否有有效数据
                    if data_matrix and any(any(cell for cell in row) for row in data_matrix):
                        # 使用pandas DataFrame保存为CSV
                        df = pd.DataFrame(data_matrix)
                        csv_path = f"{self.output_table_dir}/{base_filename}_slide_{slide_index}_table_{table_counter}.csv"
                        df.to_csv(csv_path, index=False, header=False, encoding="utf-8")
                        
                        # 创建表格JSON元数据文件
                        table_metadata = {
                            "type": "ppt_table",
                            "source": f"{base_filename}_slide_{slide_index}",
                            "slide_number": slide_index,
                            "table_index": table_counter,
                            "dimensions": {
                                "rows": len(data_matrix),
                                "columns": len(data_matrix[0]) if data_matrix else 0
                            },
                            "position": table_position,
                            "data_preview": data_matrix[:3] if len(data_matrix) > 0 else [],  # 前3行预览
                            "csv_file": csv_path
                        }
                        
                        # 保存表格元数据JSON
                        json_path = f"{self.output_table_dir}/{base_filename}_slide_{slide_index}_table_{table_counter}.json"
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(table_metadata, f, ensure_ascii=False, indent=2)
                        
                        # 记录到主结果中（增强元数据）
                        self.preprocessor.results.append({
                            "type": "ppt_table",
                            "page": slide_index,
                            "table_index": table_counter,
                            "file": csv_path,
                            "metadata_file": json_path,
                            "dimensions": {
                                "rows": len(data_matrix),
                                "columns": len(data_matrix[0]) if data_matrix else 0
                            },
                            "position": table_position,
                            "extraction_method": "python_pptx_optimized"
                        })
                        
                        print(f"[OK] 提取表格 {table_counter}: {len(data_matrix)}行 x {len(data_matrix[0]) if data_matrix else 0}列")
                        print(f"  位置: left={table_position['left']:.2f}in, top={table_position['top']:.2f}in")
                    else:
                        print(f"[WARN] 跳过空表格 {table_counter}")

    def _save_pptx_metadata(self, file_path, base_filename, slide_image_mapping):
        """
        保存PPTX专用元数据文件
        
        Args:
            file_path: 原始PPTX文件路径
            base_filename: 基础文件名
            slide_image_mapping: 幻灯片到图片的映射关系
        """
        import datetime
        
        # 统计各类型文件
        text_files = [r for r in self.preprocessor.results if r["type"] == "ppt_text"]
        table_files = [r for r in self.preprocessor.results if r["type"] == "ppt_table"]
        image_files_traditional = [r for r in self.preprocessor.results if r["type"] == "ppt_image"]
        image_files_zip = [r for r in self.preprocessor.results if r["type"] == "ppt_image_zip"]
        
        # 计算幻灯片统计
        total_slides = len(set(r["page"] for r in self.preprocessor.results if "page" in r))
        slides_with_images = len(slide_image_mapping)
        total_images_zip = sum(len(images) for images in slide_image_mapping.values())
        
        # 计算表格统计
        total_tables = len(table_files)
        total_table_rows = sum(r.get("dimensions", {}).get("rows", 0) for r in table_files)
        total_table_columns = sum(r.get("dimensions", {}).get("columns", 0) for r in table_files)
        table_positions = [r.get("position", {}) for r in table_files if r.get("position")]
        
        # 构建元数据
        metadata = {
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": base_filename,
            "source_path": file_path,
            "file_type": "PPTX",
            "processing_method": "advanced_zip_xml_parsing",
            "statistics": {
                "total_slides": total_slides,
                "slides_with_text": len(text_files),
                "slides_with_tables": len([r for r in table_files if r.get("dimensions", {}).get("rows", 0) > 0]),
                "total_tables": total_tables,
                "total_table_rows": total_table_rows,
                "total_table_columns": total_table_columns,
                "slides_with_images": slides_with_images,
                "total_images_extracted": len(image_files_traditional) + len(image_files_zip),
                "images_via_traditional": len(image_files_traditional),
                "images_via_zip_parsing": len(image_files_zip),
                "total_images_in_media": total_images_zip
            },
            "slide_image_mapping": slide_image_mapping,
            "files": {
                "text_files": text_files,
                "table_files": table_files,
                "image_files_traditional": image_files_traditional,
                "image_files_zip": image_files_zip
            },
            "table_analysis": {
                "positions": table_positions,
                "position_stats": {
                    "avg_left": sum(p.get("left", 0) for p in table_positions) / len(table_positions) if table_positions else 0,
                    "avg_top": sum(p.get("top", 0) for p in table_positions) / len(table_positions) if table_positions else 0,
                    "avg_width": sum(p.get("width", 0) for p in table_positions) / len(table_positions) if table_positions else 0,
                    "avg_height": sum(p.get("height", 0) for p in table_positions) / len(table_positions) if table_positions else 0
                }
            },
            "processing_info": {
                "extraction_methods": ["python-pptx", "zip_xml_parsing"],
                "image_formats_supported": ["PNG", "WMF", "EMF", "JPEG"],
                "table_extraction_enhanced": True,
                "table_position_tracking": True,
                "clip_descriptions_generated": True,
                "output_format": "JSON/CSV"
            }
        }
        
        # 保存元数据
        metadata_path = f"output/{base_filename}_pptx_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata_path


def process_pptx_file_advanced(preprocessor, file_path, fast_mode=False):
    """
    高级PPTX处理的入口函数
    
    Args:
        preprocessor: MultimodalPreprocessor实例
        file_path: PPTX文件路径
        fast_mode: 快速模式，跳过耗时处理
    """
    processor = AdvancedPPTProcessor(preprocessor, fast_mode=fast_mode)
    processor.process_pptx_file_advanced(file_path)
