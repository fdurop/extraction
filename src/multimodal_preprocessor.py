import sys
import os

# ========================================
# ONNX 屏蔽（必需！transformers会触发ONNX导入）
# ========================================
class _FakeONNX:
    """简单的假ONNX模块"""
    def __getattr__(self, name):
        return _FakeONNX()
    def __call__(self, *args, **kwargs):
        return _FakeONNX()

# 批量屏蔽所有可能的ONNX模块
_onnx_modules = [
    'torch.onnx', 'torch.onnx._internal', 'torch.onnx._internal.exporter',
    'torch.onnx.symbolic_helper', 'torch.onnx.utils', 'torch.onnx.operators',
    'torch.onnx.symbolic_opset9', 'torch.onnx.symbolic_opset11',
    'torch.onnx.symbolic_opset12', 'torch.onnx.symbolic_opset13',
    'torch.onnx.symbolic_opset14', 'torch.onnx.symbolic_opset15',
    'torch.onnx.symbolic_registry'
]
for _mod in _onnx_modules:
    sys.modules[_mod] = _FakeONNX()

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加路径
sys.path.append(parent_dir)  # 项目根目录
sys.path.append(current_dir)  # src目录
import json
from logger_config import get_logger, LoggerSetup
import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image, ImageEnhance
from transformers import CLIPProcessor, CLIPModel
import datetime
import pandas as pd
import cv2
from paddleocr import PaddleOCR
import re
import pdfplumber
import camelot
import csv
try:
    # 懒加载高级PPTX处理器（若不可用则忽略）
    from advanced_pptx_processor import process_pptx_file_advanced
except Exception:
    process_pptx_file_advanced = None

try:
    # 懒加载DeepSeek-VL封装器
    from deepseek_vl_wrapper import DeepSeekVLWrapper
except Exception as e:
    print(f"警告: DeepSeek-VL导入失败: {e}")
    DeepSeekVLWrapper = None

class MultimodalPreprocessor:
    def __init__(self, use_deepseek=True, use_clip=False):
        """
        初始化多模态预处理工具
        
        Args:
            use_deepseek: 是否使用DeepSeek-VL（推荐，功能更强大）
            use_clip: 是否使用CLIP（仅用于备份）
        """
        # 初始化日志系统
        self.logger = get_logger("MultimodalPreprocessor")
        LoggerSetup.log_session_start(self.logger)
        
        print("\n" + "=" * 60)
        print("🚀 多模态数据提取系统 - 初始化开始")
        print("=" * 60)
        print(f"   配置: DeepSeek-VL={use_deepseek}, CLIP={use_clip}")
        
        self.logger.info("开始初始化多模态预处理工具")
        self.logger.info(f"使用DeepSeek-VL: {use_deepseek}, 使用CLIP: {use_clip}")
        
        # 检测设备
        print("\n📱 [步骤 1/4] 检测计算设备...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ 检测到 CUDA GPU: {gpu_name}")
        else:
            print(f"   ⚠️  未检测到GPU，将使用CPU（速度较慢）")
        self.logger.info(f"检测到计算设备: {self.device}")
        
        # 设置模型使用选项
        self.use_deepseek = use_deepseek
        self.use_clip = use_clip
        
        # 创建输出目录
        print("\n📁 [步骤 2/4] 创建输出目录...")
        dirs = ["text", "images", "formulas", "tables", "code", "logs"]
        for d in dirs:
            os.makedirs(f"output/{d}", exist_ok=True)
        print(f"   ✅ 输出目录创建完成: {', '.join(dirs)}")
        self.logger.info("输出目录创建完成: text, images, formulas, tables, code, logs")
        
        # 初始化DeepSeek-VL模型
        print("\n🤖 [步骤 3/4] 初始化 AI 模型...")
        self.deepseek_vl = None
        if self.use_deepseek and DeepSeekVLWrapper is not None:
            try:
                print("   ⏳ 正在初始化 DeepSeek-VL（这是最耗时的步骤）...")
                self.logger.info("开始初始化DeepSeek-VL模型")
                self.deepseek_vl = DeepSeekVLWrapper(
                    model_path="./models/deepseek-vl-7b-chat",
                    device=self.device
                )
                self.logger.info("DeepSeek-VL初始化成功")
            except Exception as e:
                print(f"   [错误] DeepSeek-VL初始化失败: {e}")
                self.logger.error(f"DeepSeek-VL初始化失败: {e}", exc_info=True)
                import traceback
                traceback.print_exc()
                print("   将使用备用方案")
                self.use_deepseek = False
        else:
            if not self.use_deepseek:
                print("   [INFO] DeepSeek-VL已被禁用")
                self.logger.info("DeepSeek-VL已被用户禁用")
            elif DeepSeekVLWrapper is None:
                print("   [错误] DeepSeekVLWrapper类未找到，请检查导入")
                self.logger.error("DeepSeekVLWrapper类未找到")
        
        # 初始化CLIP模型（可选）
        self.clip_model = None
        self.clip_processor = None
        if self.use_clip:
            print("[CLIP] 正在加载CLIP模型...")
            print("   [INFO] 本地模型加载中，请稍候...")
            self.logger.info("开始加载CLIP模型")
            try:
                self.clip_model = CLIPModel.from_pretrained("./clip-model").to(self.device)
                print("   [OK] CLIP模型加载完成")
                self.logger.info("CLIP模型加载成功（本地）")
            except Exception as e:
                print(f"   [ERROR] 本地CLIP模型加载失败: {e}")
                self.logger.warning(f"本地CLIP模型加载失败: {e}")
                print("   [INFO] 尝试在线下载CLIP模型，这可能需要几分钟...")
                try:
                    self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                    print("   [OK] 在线CLIP模型下载并加载完成")
                    self.logger.info("CLIP模型加载成功（在线）")
                except Exception as e2:
                    print(f"   [ERROR] CLIP模型加载完全失败: {e2}")
                    self.logger.error(f"CLIP模型加载完全失败: {e2}")
                    self.use_clip = False
            
            if self.use_clip:
                print("[CLIP] 正在加载CLIP处理器...")
                print("   [INFO] 处理器加载中（可能需要下载）...")
                try:
                    self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                    print("   [OK] CLIP处理器加载完成")
                    self.logger.info("CLIP处理器加载成功")
                except Exception as e:
                    print(f"   [ERROR] CLIP处理器加载失败: {e}")
                    self.logger.error(f"CLIP处理器加载失败: {e}")
                    self.use_clip = False
        
        # 初始化OCR引擎（首次运行较慢）
        print("\n📝 [步骤 4/4] 初始化 OCR 引擎...")
        print("   ⏳ 正在加载 PaddleOCR 中文+英文模型...")
        print("      （首次运行需要下载，可能需要1-3分钟）")
        self.logger.info("开始初始化OCR引擎")
        
        # 根据设备自动选择GPU/CPU
        use_gpu = (self.device == "cuda")
        try:
            # PaddleOCR 初始化
            # use_angle_cls=True: 支持文字方向检测
            # lang='ch': 中文+英文
            # show_log=False: 减少日志输出
            self.ocr_reader = PaddleOCR(
                use_angle_cls=True, 
                lang='ch', 
                use_gpu=use_gpu,
                show_log=False
            )
            device_info = "GPU加速" if use_gpu else "CPU模式"
            print(f"   ✅ OCR引擎初始化完成 (PaddleOCR {device_info})")
            self.logger.info(f"OCR引擎初始化成功 (PaddleOCR, gpu={use_gpu})")
        except Exception as e:
            print(f"   ⚠️  OCR初始化失败: {e}")
            print("      → 将继续运行，但跳过OCR公式识别功能")
            self.logger.warning(f"OCR初始化失败: {e}")
            self.ocr_reader = None
        
        # 存储处理结果
        self.results = []
        
        print("\n" + "=" * 60)
        print("✅ 初始化完成！系统已就绪")
        print("=" * 60)
        log_path = LoggerSetup.get_log_file_path()
        print(f"📄 日志文件: {log_path}\n")
        self.logger.info("多模态预处理工具初始化完成")
        self.logger.info(f"日志文件路径: {log_path}")

    def process_pdf(self, file_path):
        """处理PDF文件，提取文本和图像"""
        print(f"开始处理PDF文件: {file_path}")
        self.logger.info(f"开始处理PDF文件: {file_path}")
        doc = fitz.open(file_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        self.logger.info(f"PDF文件名: {base_filename}, 总页数: {len(doc)}")
        
        # 为当前PDF文件单独记录结果
        current_results = []
        original_results = self.results
        self.results = current_results
        
        try:
            for page_num in range(len(doc)):
                print(f"处理第 {page_num + 1}/{len(doc)} 页...")
                page = doc.load_page(page_num)
                page_text = page.get_text()
                
                # 处理页面图像
                image_list = page.get_images(full=True)
                page_images = []
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # 保存原始图像
                    img_path = f"output/images/{base_filename}_p{page_num+1}_img{img_index+1}.{base_image['ext']}"
                    with open(img_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # 处理图像并保存
                    image_data = self.process_image(img_path, page_text)
                    self.save_image_data(image_data, base_filename, page_num, img_index)
                    page_images.append(img_path)
                
                # 处理页面文本
                text_data = self.process_text(page_text, page_num, page_images)
                text_data["source"] = f"{base_filename}_page{page_num+1}"
                self.save_text_data(text_data, base_filename, page_num)
                
                # 提取页面中的公式、表格、代码
                self.extract_formulas_from_page(page, page_text, base_filename, page_num)
                self.extract_tables_from_page(page, page_text, base_filename, page_num)
                self.extract_code_from_page(page_text, base_filename, page_num, page=page)
            
            # 保存PDF专用元数据
            self.save_pdf_metadata(file_path, base_filename)
            print(f"PDF处理完成！结果保存在output/{base_filename}_pdf_metadata.json")
            
        finally:
            # 恢复原始结果列表并合并当前结果
            self.results = original_results
            self.results.extend(current_results)

    def process_text(self, text, page_num, page_images):
        """处理文本内容"""
        # 清理文本
        cleaned_text = text.strip()
        if not cleaned_text:
            cleaned_text = "[页面无文本内容]"
        
        # 使用CLIP生成文本语义向量
        try:
            inputs = self.clip_processor(text=cleaned_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                text_vector = text_features.cpu().numpy()[0]
        except Exception as e:
            print(f"文本向量化失败: {e}")
            text_vector = np.zeros(512)  # CLIP默认向量维度
        
        return {
            "type": "text",
            "page": page_num + 1,
            "raw_text": cleaned_text,
            "word_count": len(cleaned_text),
            "associated_images": page_images,
            "text_vector": text_vector.tolist()
        }

    def process_image(self, image_path, page_text):
        """处理图像（优先使用DeepSeek-VL，备用CLIP）"""
        # 图像增强
        enhanced_path = self.enhance_image(image_path)
        
        # 获取图像基本信息
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_type = img.format
                mode = img.mode
        except Exception as e:
            print(f"读取图像信息失败: {e}")
            width = height = 0
            format_type = mode = "unknown"
        
        # 优先使用DeepSeek-VL生成描述
        image_description = ""
        description_tags = []
        image_vector = None
        
        if self.use_deepseek and self.deepseek_vl is not None:
            try:
                # 使用DeepSeek-VL生成详细描述
                image_description = self.deepseek_vl.describe_image(enhanced_path)
                description_tags = [{"description": image_description, "source": "deepseek-vl"}]
            except Exception as e:
                print(f"DeepSeek-VL图像描述失败: {e}")
        
        # 备用：使用CLIP生成向量和描述
        if self.use_clip and self.clip_model is not None:
            try:
                image = Image.open(enhanced_path)
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                    image_vector = image_features.cpu().numpy()[0]
                
                # 如果DeepSeek-VL没有生成描述，使用CLIP描述
                if not description_tags:
                    description_tags = self.generate_image_descriptions(enhanced_path)
                
            except Exception as e:
                print(f"CLIP图像处理失败: {e}")
                image_vector = np.zeros(512) if image_vector is None else image_vector
        
        result = {
            "type": "image",
            "image_path": image_path,
            "enhanced_path": enhanced_path,
            "width": width,
            "height": height,
            "format": format_type,
            "mode": mode,
            "page_text_context": page_text[:200] + "..." if len(page_text) > 200 else page_text,
            "deepseek_description": image_description,
            "clip_descriptions": description_tags
        }
        
        # 只有使用CLIP时才添加向量
        if image_vector is not None:
            result["image_vector"] = image_vector.tolist()
        
        return result

    def enhance_image(self, image_path):
        """图像增强处理"""
        img = Image.open(image_path)
        
        # 对比度增强
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # 锐度增强
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)
        
        # 保存增强后的图像
        enhanced_path = image_path.replace(".", "_enhanced.")
        img.save(enhanced_path)
        
        return enhanced_path

    def clip_generate_description(self, image_path: str) -> str:
        """基于CLIP为图片生成描述文本并保存为JSON，返回描述文件路径。"""
        try:
            descriptions = self.generate_image_descriptions(image_path)
        except Exception as e:
            print(f"生成图片描述失败: {e}")
            descriptions = []

        base, _ = os.path.splitext(os.path.basename(image_path))
        desc_path = os.path.join("output", "images", f"{base}_desc.json")
        try:
            with open(desc_path, "w", encoding="utf-8") as f:
                json.dump({
                    "image_path": image_path,
                    "clip_descriptions": descriptions
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存图片描述失败: {e}")
        return desc_path

    def generate_image_descriptions(self, image_path):
        """使用CLIP生成图像描述标签"""
        try:
            image = Image.open(image_path)
            image_input = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # 预定义的描述候选列表
            text_descriptions = [
                "科学图表", "数学公式", "数据图表", "流程图", 
                "实验装置", "分子结构", "几何图形", "统计图表",
                "技术示意图", "概念图", "网络图", "系统架构图",
                "照片", "插图", "示例图", "对比图",
                "文本图像", "表格", "代码", "截图"
            ]
            
            text_inputs = self.clip_processor(
                text=text_descriptions, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_input)
                text_features = self.clip_model.get_text_features(**text_inputs)
                
                # 归一化特征向量
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(5)  # 取前5个最相似的描述
                
                descriptions = []
                for value, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
                    descriptions.append({
                        "description": text_descriptions[idx],
                        "confidence": float(value)
                    })
            
            return descriptions
            
        except Exception as e:
            print(f"图像描述生成错误: {image_path}, {str(e)}")
            return []

    def save_text_data(self, data, filename, page_num):
        """保存文本处理结果"""
        output_path = f"output/text/{filename}_p{page_num+1}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.results.append({
            "type": "text",
            "page": page_num + 1,
            "file": output_path
        })

    def save_image_data(self, data, filename, page_num, img_index):
        """保存图像处理结果"""
        output_path = f"output/images/{filename}_p{page_num+1}_img{img_index+1}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        self.results.append({
            "type": "image",
            "page": page_num + 1,
            "file": output_path
        })

    def save_pdf_metadata(self, file_path, filename):
        """保存PDF专用元数据文件"""
        # 统计信息
        text_files = [r for r in self.results if r["type"] == "text"]
        image_files = [r for r in self.results if r["type"] == "image"]
        formula_files = [r for r in self.results if r["type"] == "formula"]
        table_files = [r for r in self.results if r["type"] == "table"]
        code_files = [r for r in self.results if r["type"] == "code"]
        
        # 计算表格统计
        total_table_rows = sum(r.get("rows", 0) for r in table_files)
        total_table_columns = sum(r.get("columns", 0) for r in table_files)
        
        # 计算代码统计
        total_code_lines = sum(r.get("line_count", 0) for r in code_files)
        code_languages = list(set(r.get("language", "unknown") for r in code_files))
        
        metadata = {
            "processing_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source_file": filename,
            "source_path": file_path,
            "file_type": "PDF",
            "processing_method": "pymupdf_ocr_camelot",
            "statistics": {
                "total_pages": len(set(r["page"] for r in self.results)),
                "total_text_blocks": len(text_files),
                "total_images": len(image_files),
                "total_formulas": len(formula_files),
                "total_tables": len(table_files),
                "total_table_rows": total_table_rows,
                "total_table_columns": total_table_columns,
                "total_code_blocks": len(code_files),
                "total_code_lines": total_code_lines,
                "code_languages": code_languages
            },
            "files": {
                "text_files": text_files,
                "image_files": image_files,
                "formula_files": formula_files,
                "table_files": table_files,
                "code_files": code_files
            },
            "processing_info": {
                "models_used": {
                    "deepseek_vl": self.use_deepseek,
                    "clip": self.use_clip,
                    "ocr": self.ocr_reader is not None
                },
                "deepseek_model": "deepseek-vl-7b-chat" if self.use_deepseek else None,
                "clip_model": "openai/clip-vit-base-patch32" if self.use_clip else None,
                "device": self.device,
                "output_format": "JSON/CSV",
                "extraction_features": ["text", "images", "formulas", "tables", "code"],
                "enhanced_features": {
                    "image_description": "deepseek-vl" if self.use_deepseek else "clip",
                    "formula_recognition": "deepseek-vl" if self.use_deepseek else "ocr",
                    "code_recognition": "deepseek-vl" if self.use_deepseek else "pattern"
                }
            }
        }
        
        with open(f"output/{filename}_pdf_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def extract_formulas_from_page(self, page, page_text, filename, page_num):
        """从页面中提取数学公式"""
        formulas = []
        
        # 1. 从文本中提取LaTeX格式的公式
        latex_patterns = [
            r'\$\$([^$]+)\$\$',  # 块级公式 $$...$$
            r'\$([^$]+)\$',      # 行内公式 $...$
            r'\\begin\{equation\}(.*?)\\end\{equation\}',  # equation环境
            r'\\begin\{align\}(.*?)\\end\{align\}',        # align环境
            r'\\begin\{math\}(.*?)\\end\{math\}',          # math环境
        ]
        
        for i, pattern in enumerate(latex_patterns):
            matches = re.findall(pattern, page_text, re.DOTALL)
            for j, match in enumerate(matches):
                formula_data = {
                    "type": "formula",
                    "page": page_num + 1,
                    "formula_id": f"{filename}_p{page_num+1}_formula{len(formulas)+1}",
                    "content": match.strip(),
                    "format": "latex",
                    "extraction_method": f"pattern_{i+1}",
                    "context": self.get_text_context(page_text, match, 100)
                }
                formulas.append(formula_data)
        
        # 2. 使用DeepSeek-VL识别图像中的公式（推荐）
        if self.use_deepseek and self.deepseek_vl is not None and len(formulas) < 10:
            try:
                # 获取页面中的图像列表
                image_list = page.get_images(full=True)
                
                # 对每个图像尝试识别公式（限制数量）
                for img_index, img in enumerate(image_list[:3]):  # 只处理前3个图像
                    try:
                        xref = img[0]
                        base_image = page.document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # 保存临时图像
                        temp_img_path = f"output/images/temp_formula_{filename}_p{page_num+1}_img{img_index}.png"
                        with open(temp_img_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # 使用DeepSeek-VL识别公式
                        formula_result = self.deepseek_vl.recognize_formula(temp_img_path)
                        
                        if formula_result["latex"]:
                            formula_data = {
                                "type": "formula",
                                "page": page_num + 1,
                                "formula_id": f"{filename}_p{page_num+1}_formula{len(formulas)+1}",
                                "content": formula_result["latex"],
                                "format": "latex",
                                "description": formula_result.get("description", ""),
                                "extraction_method": "deepseek_vl",
                                "source_image": temp_img_path
                            }
                            formulas.append(formula_data)
                        
                        # 删除临时图像（可选）
                        # os.remove(temp_img_path)
                        
                    except Exception as e:
                        print(f"DeepSeek-VL公式识别失败 (图像 {img_index}): {e}")
                        continue
            
            except Exception as e:
                print(f"DeepSeek-VL公式识别失败 (页面 {page_num+1}): {e}")
        
        # 3. 备用：使用OCR识别公式
        elif self.ocr_reader and len(formulas) < 5:  # 限制OCR处理，避免卡住
            try:
                # 获取页面图像
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # PaddleOCR识别
                # PaddleOCR返回格式: [[line_result], ...]
                # line_result = [bbox, (text, confidence)]
                ocr_result = self.ocr_reader.ocr(img, cls=True)
                
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0][:3]:  # 只处理前3个结果，避免过多处理
                        bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text = line[1][0]  # 文本
                        confidence = line[1][1]  # 置信度
                        
                        # 检查是否包含数学符号
                        math_symbols = ['∑', '∫', '∂', '∆', '∇', '∞', '±', '≠', '≤', '≥', 'α', 'β', 'γ', 'δ', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω']
                        if any(symbol in text for symbol in math_symbols) and confidence > 0.6:
                            if any(char.isdigit() or char in '+-*/=()[]{}^_' for char in text):
                                formula_data = {
                                    "type": "formula",
                                    "page": page_num + 1,
                                    "formula_id": f"{filename}_p{page_num+1}_formula{len(formulas)+1}",
                                    "content": text,
                                    "format": "ocr_text",
                                    "confidence": float(confidence),
                                    "extraction_method": "paddleocr",
                                    "bbox": [[float(pt[0]), float(pt[1])] for pt in bbox]  # PaddleOCR bbox格式
                                }
                                formulas.append(formula_data)
            
            except Exception as e:
                print(f"OCR公式识别失败 (页面 {page_num+1}): {e}")
        
        # 保存公式数据
        if formulas:
            self.save_formulas_data(formulas, filename, page_num)
        
        return formulas

    def extract_tables_from_page(self, page, page_text, filename, page_num):
        """从页面中提取表格"""
        tables = []
        
        try:
            # 使用camelot提取表格 - 限制处理时间，避免卡住
            pdf_path = None
            for file in os.listdir("input"):
                if file.lower().endswith('.pdf') and filename in file:
                    pdf_path = os.path.join("input", file)
                    break
            
            if pdf_path and os.path.exists(pdf_path) and page_num < 10:  # 只处理前10页，避免卡住
                # 提取当前页面的表格，限制处理
                camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num+1), flavor='lattice')
                
                for i, table in enumerate(camelot_tables[:2]):  # 只处理前2个表格
                    if table.df is not None and not table.df.empty and len(table.df) > 0:
                        table_data = {
                            "type": "table",
                            "page": page_num + 1,
                            "table_id": f"{filename}_p{page_num+1}_table{i+1}",
                            "rows": len(table.df),
                            "columns": len(table.df.columns),
                            "data": table.df.to_dict('records'),
                            "extraction_method": "camelot",
                            "accuracy": getattr(table, 'accuracy', 0.0)
                        }
                        tables.append(table_data)
        
        except Exception as e:
            print(f"Camelot表格提取跳过 (页面 {page_num+1}): {e}")
        
        # 备用方法：从文本中识别表格模式
        table_patterns = self.detect_text_tables(page_text)
        for i, pattern in enumerate(table_patterns):
            table_data = {
                "type": "table",
                "page": page_num + 1,
                "table_id": f"{filename}_p{page_num+1}_texttable{i+1}",
                "content": pattern,
                "extraction_method": "text_pattern",
                "context": self.get_text_context(page_text, pattern, 50)
            }
            tables.append(table_data)
        
        # 保存表格数据
        if tables:
            self.save_tables_data(tables, filename, page_num)
        
        return tables

    def extract_code_from_page(self, page_text, filename, page_num, page=None):
        """从页面文本和图像中提取代码块"""
        code_blocks = []
        
        # 1. 从文本中提取代码块（原有逻辑）
        code_patterns = [
            r'```(\w*)\n(.*?)```',  # Markdown代码块
            r'`([^`]+)`',           # 行内代码
            r'(?:^|\n)((?:    |\t)[^\n]+(?:\n(?:    |\t)[^\n]+)*)',  # 缩进代码块
        ]
        
        # 编程语言关键字
        programming_keywords = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'else:', 'for ', 'while ', 'return',
            'function', 'var ', 'let ', 'const ', 'console.log', 'print(', 'println',
            'public ', 'private ', 'static ', 'void ', 'int ', 'String ', 'boolean',
            '#include', 'using namespace', 'int main(', 'printf(', 'cout <<'
        ]
        
        for i, pattern in enumerate(code_patterns):
            matches = re.findall(pattern, page_text, re.DOTALL | re.MULTILINE)
            
            for j, match in enumerate(matches):
                if isinstance(match, tuple):
                    language = match[0] if match[0] else "unknown"
                    content = match[1] if len(match) > 1 else match[0]
                else:
                    content = match
                    language = "unknown"
                
                # 检查是否包含编程关键字
                if any(keyword in content for keyword in programming_keywords) or len(content.strip()) > 20:
                    code_data = {
                        "type": "code",
                        "page": page_num + 1,
                        "code_id": f"{filename}_p{page_num+1}_code{len(code_blocks)+1}",
                        "content": content.strip(),
                        "language": language,
                        "extraction_method": f"pattern_{i+1}",
                        "line_count": len(content.strip().split('\n')),
                        "context": self.get_text_context(page_text, content, 100)
                    }
                    code_blocks.append(code_data)
        
        # 2. 使用DeepSeek-VL从图像中识别代码
        if self.use_deepseek and self.deepseek_vl is not None and page is not None and len(code_blocks) < 10:
            try:
                # 获取页面中的图像列表
                image_list = page.get_images(full=True)
                
                # 对每个图像尝试识别代码（限制数量）
                for img_index, img in enumerate(image_list[:3]):  # 只处理前3个图像
                    try:
                        xref = img[0]
                        base_image = page.document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # 保存临时图像
                        temp_img_path = f"output/images/temp_code_{filename}_p{page_num+1}_img{img_index}.png"
                        with open(temp_img_path, "wb") as f:
                            f.write(image_bytes)
                        
                        # 使用DeepSeek-VL识别代码
                        code_result = self.deepseek_vl.recognize_code(temp_img_path)
                        
                        if code_result["code"]:
                            code_data = {
                                "type": "code",
                                "page": page_num + 1,
                                "code_id": f"{filename}_p{page_num+1}_code{len(code_blocks)+1}",
                                "content": code_result["code"],
                                "language": code_result.get("language", "unknown"),
                                "description": code_result.get("description", ""),
                                "extraction_method": "deepseek_vl",
                                "line_count": len(code_result["code"].split('\n')),
                                "source_image": temp_img_path
                            }
                            code_blocks.append(code_data)
                        
                        # 删除临时图像（可选）
                        # os.remove(temp_img_path)
                        
                    except Exception as e:
                        print(f"DeepSeek-VL代码识别失败 (图像 {img_index}): {e}")
                        continue
            
            except Exception as e:
                print(f"DeepSeek-VL代码识别失败 (页面 {page_num+1}): {e}")
        
        # 保存代码数据
        if code_blocks:
            self.save_code_data(code_blocks, filename, page_num)
        
        return code_blocks

    def get_text_context(self, full_text, target_text, context_length=100):
        """获取目标文本的上下文"""
        try:
            index = full_text.find(target_text)
            if index == -1:
                return target_text
            
            start = max(0, index - context_length)
            end = min(len(full_text), index + len(target_text) + context_length)
            return full_text[start:end]
        except:
            return target_text

    def detect_text_tables(self, text):
        """从文本中检测表格模式"""
        tables = []
        lines = text.split('\n')
        
        # 寻找包含多个制表符或空格分隔的行
        table_lines = []
        for line in lines:
            # 检查是否包含表格特征：多个制表符、竖线分隔符等
            if '\t' in line and line.count('\t') >= 2:
                table_lines.append(line)
            elif '|' in line and line.count('|') >= 2:
                table_lines.append(line)
            elif re.search(r'\s{3,}', line) and len(line.split()) >= 3:
                table_lines.append(line)
            else:
                if table_lines and len(table_lines) >= 2:
                    tables.append('\n'.join(table_lines))
                table_lines = []
        
        # 检查最后一组
        if table_lines and len(table_lines) >= 2:
            tables.append('\n'.join(table_lines))
        
        return tables

    def save_formulas_data(self, formulas, filename, page_num):
        """保存公式数据"""
        # JSON格式保存
        json_path = f"output/formulas/{filename}_p{page_num+1}_formulas.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(formulas, f, ensure_ascii=False, indent=2)
        
        # CSV格式保存
        csv_path = f"output/formulas/{filename}_p{page_num+1}_formulas.csv"
        if formulas:
            df = pd.DataFrame(formulas)
            df.to_csv(csv_path, index=False, encoding="utf-8")
        
        # 记录到结果
        for formula in formulas:
            self.results.append({
                "type": "formula",
                "page": page_num + 1,
                "file": json_path,
                "formula_id": formula["formula_id"]
            })

    def save_tables_data(self, tables, filename, page_num):
        """保存表格数据"""
        # JSON格式保存
        json_path = f"output/tables/{filename}_p{page_num+1}_tables.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)
        
        # 为每个表格单独保存CSV
        for i, table in enumerate(tables):
            if table.get("data") and isinstance(table["data"], list):
                csv_path = f"output/tables/{table['table_id']}.csv"
                try:
                    df = pd.DataFrame(table["data"])
                    df.to_csv(csv_path, index=False, encoding="utf-8")
                except Exception as e:
                    print(f"保存表格CSV失败: {e}")
        
        # 记录到结果
        for table in tables:
            self.results.append({
                "type": "table",
                "page": page_num + 1,
                "file": json_path,
                "table_id": table["table_id"],
                "rows": table.get("rows", 0),
                "columns": table.get("columns", 0)
            })

    def save_code_data(self, code_blocks, filename, page_num):
        """保存代码数据"""
        # JSON格式保存
        json_path = f"output/code/{filename}_p{page_num+1}_code.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(code_blocks, f, ensure_ascii=False, indent=2)
        
        # CSV格式保存
        csv_path = f"output/code/{filename}_p{page_num+1}_code.csv"
        if code_blocks:
            df = pd.DataFrame(code_blocks)
            df.to_csv(csv_path, index=False, encoding="utf-8")
        
        # 为每个代码块单独保存文件
        for code in code_blocks:
            if code.get("language") and code.get("language") != "unknown":
                ext = self.get_file_extension(code["language"])
                code_file_path = f"output/code/{code['code_id']}.{ext}"
                with open(code_file_path, "w", encoding="utf-8") as f:
                    f.write(code["content"])
        
        # 记录到结果
        for code in code_blocks:
            self.results.append({
                "type": "code",
                "page": page_num + 1,
                "file": json_path,
                "code_id": code["code_id"],
                "language": code.get("language", "unknown"),
                "line_count": code.get("line_count", 0)
            })

    def get_file_extension(self, language):
        """根据编程语言获取文件扩展名"""
        extensions = {
            "python": "py",
            "javascript": "js",
            "java": "java",
            "cpp": "cpp",
            "c": "c",
            "csharp": "cs",
            "php": "php",
            "ruby": "rb",
            "go": "go",
            "rust": "rs",
            "swift": "swift",
            "kotlin": "kt",
            "typescript": "ts",
            "html": "html",
            "css": "css",
            "sql": "sql",
            "shell": "sh",
            "bash": "sh",
            "powershell": "ps1"
        }
        return extensions.get(language.lower(), "txt")

if __name__ == "__main__":
    # 初始化主日志
    main_logger = get_logger("Main")
    LoggerSetup.log_session_start(main_logger)
    
    main_logger.info("多模态PDF数据预处理器启动")
    
    try:
        # 初始化处理器（内部已有详细进度）
        processor = MultimodalPreprocessor()
        main_logger.info("处理器初始化完成")
        
        # 检查输入目录中的PDF文件
        input_dir = "input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            print(f"[ERROR] 输入目录不存在，已创建: {input_dir}")
            print("请将PDF文件放入input目录后重新运行")
        else:
            # 查找所有支持的文件，过滤掉临时文件
            input_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.pdf', '.pptx')) and not f.startswith('~$')]

            if not input_files:
                print(f"\n❌ 错误：在 {input_dir} 目录中未找到PDF/PPTX文件")
                print("   请将PDF或PPTX文件放入input目录后重新运行")
            else:
                pdf_count = sum(1 for f in input_files if f.lower().endswith('.pdf'))
                pptx_count = sum(1 for f in input_files if f.lower().endswith('.pptx'))
                print("\n" + "=" * 60)
                print(f"📂 扫描到文件: {pdf_count} 个PDF, {pptx_count} 个PPTX")
                print("=" * 60)
                main_logger.info(f"找到 {pdf_count} 个PDF文件, {pptx_count} 个PPTX文件")

                # 用于跟踪PPTX处理器（用于交互式补充）
                pptx_processors = []
                
                for idx, in_file in enumerate(input_files, 1):
                    input_path = os.path.join(input_dir, in_file)
                    print(f"\n📄 [{idx}/{len(input_files)}] 正在处理: {in_file}")
                    main_logger.info(f"开始处理文件 [{idx}/{len(input_files)}]: {in_file}")
                    
                    if in_file.lower().endswith('.pdf'):
                        print("   [PDF] 使用PDF处理器（PyMuPDF + OCR + 表格提取）...")
                        processor.process_pdf(input_path)
                    elif in_file.lower().endswith('.pptx'):
                        if process_pptx_file_advanced is None:
                            print("   [WARN] 未安装PPTX处理依赖或导入失败，跳过PPTX文件")
                            main_logger.warning(f"PPTX处理器不可用，跳过文件: {in_file}")
                        else:
                            print("   [PPTX] 使用高级PPTX处理器（ZIP+XML解析 + 优化表格提取）...")
                            from advanced_pptx_processor import AdvancedPPTProcessor
                            pptx_processor = AdvancedPPTProcessor(processor, fast_mode=False)
                            pptx_processor.process_pptx_file_advanced(input_path)
                            pptx_processors.append(pptx_processor)
                    
                    print(f"   ✅ [{idx}/{len(input_files)}] 完成: {in_file}")
                    main_logger.info(f"完成处理文件 [{idx}/{len(input_files)}]: {in_file}")

                print("\n" + "=" * 60)
                print("🎉 所有文件处理完成！")
                print("=" * 60)
                print(f"📁 结果保存目录: output/")
                print(f"   ├─ text/      (文本内容)")
                print(f"   ├─ images/    (图片及描述)")
                print(f"   ├─ formulas/  (公式)")
                print(f"   ├─ tables/    (表格)")
                print(f"   └─ code/      (代码)")
                main_logger.info("所有文件处理完成")
                
                # === 人工补充环节 ===
                if pptx_processors:
                    # 1. 检查识别失败的公式和代码
                    total_failed = sum(len(p.failed_recognitions) for p in pptx_processors)
                    if total_failed > 0:
                        print(f"\n[INFO] 发现 {total_failed} 项公式/代码识别失败")
                        choice = input("是否现在补充识别失败的公式/代码？(y/n): ").strip().lower()
                        if choice == 'y':
                            for pptx_processor in pptx_processors:
                                if pptx_processor.failed_recognitions:
                                    pptx_processor.interactive_supplement()
                    
                    # 2. 检查缺少专业内容的图片
                    total_unprofessional = sum(len(p.images_without_professional_content) for p in pptx_processors)
                    if total_unprofessional > 0:
                        print(f"\n[INFO] 发现 {total_unprofessional} 张图片缺少专业内容解释")
                        choice = input("是否现在添加专业解释？(y/n): ").strip().lower()
                        if choice == 'y':
                            for pptx_processor in pptx_processors:
                                if pptx_processor.images_without_professional_content:
                                    pptx_processor.supplement_professional_images()
                    
                    # 3. 生成专业术语库
                    total_terms = sum(len(p.professional_terms) for p in pptx_processors)
                    if total_terms > 0:
                        print(f"\n[INFO] 从课程中提取到 {total_terms} 个专业术语")
                        # 合并所有处理器的术语
                        all_terms = set()
                        for p in pptx_processors:
                            all_terms.update(p.professional_terms)
                        
                        # 使用第一个处理器生成术语库
                        pptx_processors[0].professional_terms = all_terms
                        pptx_processors[0].generate_professional_terms_library()
                    else:
                        print("\n[INFO] 未提取到专业术语（可能是描述格式问题）")
                
                # 会话成功结束
                LoggerSetup.log_session_end(main_logger, success=True)
                print(f"\n📋 完整日志已保存至: {LoggerSetup.get_log_file_path()}")
                
    except Exception as e:
        print(f"[ERROR] 处理过程中发生错误: {e}")
        main_logger.error(f"处理过程中发生严重错误: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        LoggerSetup.log_session_end(main_logger, success=False)
        print(f"\n📋 错误日志已保存至: {LoggerSetup.get_log_file_path()}")