import sys
import os

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加路径
sys.path.append(parent_dir)  # 项目根目录
sys.path.append(current_dir)  # src目录
import json
import fitz  # PyMuPDF
import torch
import numpy as np
from PIL import Image, ImageEnhance
from transformers import CLIPProcessor, CLIPModel
import datetime
import pandas as pd
import cv2
import easyocr
import re
import pdfplumber
import camelot
import csv
try:
    # 懒加载高级PPTX处理器（若不可用则忽略）
    from advanced_pptx_processor import process_pptx_file_advanced
except Exception:
    process_pptx_file_advanced = None

class MultimodalPreprocessor:
    def __init__(self):
        """初始化多模态预处理工具"""
        print("🚀 开始初始化多模态预处理工具...")
        
        # 检测设备
        print("📱 检测计算设备...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✓ 使用设备: {self.device}")
        
        # 创建输出目录
        print("📁 创建输出目录...")
        os.makedirs("output/text", exist_ok=True)
        os.makedirs("output/images", exist_ok=True)
        os.makedirs("output/formulas", exist_ok=True)
        os.makedirs("output/tables", exist_ok=True)
        os.makedirs("output/code", exist_ok=True)
        print("✓ 输出目录创建完成")
        
        # 初始化CLIP模型（可能较慢）
        print("🤖 正在加载CLIP模型...")
        print("   ⏳ 本地模型加载中，请稍候...")
        try:
            self.clip_model = CLIPModel.from_pretrained("./clip-model").to(self.device)
            print("   ✓ CLIP模型加载完成")
        except Exception as e:
            print(f"   ❌ 本地CLIP模型加载失败: {e}")
            print("   ⏳ 尝试在线下载CLIP模型，这可能需要几分钟...")
            try:
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
                print("   ✓ 在线CLIP模型下载并加载完成")
            except Exception as e2:
                print(f"   ❌ CLIP模型加载完全失败: {e2}")
                raise e2
        
        print("🔧 正在加载CLIP处理器...")
        print("   ⏳ 处理器加载中（可能需要下载）...")
        try:
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("   ✓ CLIP处理器加载完成")
        except Exception as e:
            print(f"   ❌ CLIP处理器加载失败: {e}")
            raise e
        
        # 初始化OCR引擎（首次运行较慢）
        print("👁 正在初始化OCR引擎...")
        print("   ⏳ 首次运行需要下载模型文件，这可能需要几分钟，请耐心等待...")
        print("   📥 正在下载中文和英文OCR模型...")
        try:
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)  # 强制使用CPU避免GPU问题
            print("   ✓ OCR引擎初始化完成")
        except Exception as e:
            print(f"   ⚠ OCR初始化失败，将跳过OCR功能: {e}")
            print("   将继续运行，但跳过OCR公式识别功能")
            self.ocr_reader = None
        
        # 存储处理结果
        self.results = []
        
        print("🎉 多模态预处理工具初始化完成！")
        print("=" * 50)

    def process_pdf(self, file_path):
        """处理PDF文件，提取文本和图像"""
        print(f"开始处理PDF文件: {file_path}")
        doc = fitz.open(file_path)
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
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
                self.extract_code_from_page(page_text, base_filename, page_num)
            
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
        """处理图像（使用CLIP）"""
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
        
        # 使用CLIP生成图像向量和描述
        try:
            image = Image.open(enhanced_path)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_vector = image_features.cpu().numpy()[0]
            
            # 生成图像描述标签
            description_tags = self.generate_image_descriptions(enhanced_path)
            
        except Exception as e:
            print(f"图像处理失败: {e}")
            image_vector = np.zeros(512)
            description_tags = []
        
        return {
            "type": "image",
            "image_path": image_path,
            "enhanced_path": enhanced_path,
            "width": width,
            "height": height,
            "format": format_type,
            "mode": mode,
            "page_text_context": page_text[:200] + "..." if len(page_text) > 200 else page_text,
            "image_vector": image_vector.tolist(),
            "clip_descriptions": description_tags
        }

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
                "clip_model": "openai/clip-vit-base-patch32",
                "device": self.device,
                "output_format": "JSON/CSV",
                "ocr_enabled": self.ocr_reader is not None,
                "extraction_features": ["text", "images", "formulas", "tables", "code"]
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
        
        # 2. 从图像中识别公式（使用OCR）- 简化版本避免卡住
        if self.ocr_reader and len(formulas) < 5:  # 限制OCR处理，避免卡住
            try:
                # 获取页面图像
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # OCR识别，设置超时
                results = self.ocr_reader.readtext(img, width_ths=0.7, height_ths=0.7)
                
                for result in results[:3]:  # 只处理前3个结果，避免过多处理
                    text = result[1]
                    confidence = result[2]
                    
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
                                "extraction_method": "ocr",
                                "bbox": [[float(pt[0]), float(pt[1])] for pt in result[0]]  # 转换为Python原生类型
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

    def extract_code_from_page(self, page_text, filename, page_num):
        """从页面文本中提取代码块"""
        code_blocks = []
        
        # 代码块模式
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
    print("=" * 60)
    print("多模态PDF数据预处理器")
    print("=" * 60)
    print("开始初始化...")  # 添加调试信息
    
    # 确保输出目录存在
    os.makedirs("output/text", exist_ok=True)
    os.makedirs("output/images", exist_ok=True)
    os.makedirs("output/formulas", exist_ok=True)
    os.makedirs("output/tables", exist_ok=True)
    os.makedirs("output/code", exist_ok=True)
    
    try:
        # 初始化处理器
        print("⚙️ 正在初始化处理器...")
        print("⚠️ 注意：首次运行可能需要下载模型，请耐心等待...")
        processor = MultimodalPreprocessor()
        print("✅ 处理器初始化完成!")
        
        # 检查输入目录中的PDF文件
        input_dir = "input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            print(f"❌ 输入目录不存在，已创建: {input_dir}")
            print("请将PDF文件放入input目录后重新运行")
        else:
            # 查找所有支持的文件，过滤掉临时文件
            input_files = [f for f in os.listdir(input_dir) 
                          if f.lower().endswith(('.pdf', '.pptx')) and not f.startswith('~$')]

            if not input_files:
                print(f"❌ 在 {input_dir} 目录中未找到PDF/PPTX文件")
                print("请将PDF或PPTX文件放入input目录后重新运行")
            else:
                pdf_count = sum(1 for f in input_files if f.lower().endswith('.pdf'))
                pptx_count = sum(1 for f in input_files if f.lower().endswith('.pptx'))
                print(f"✓ 找到 {pdf_count} 个PDF文件, {pptx_count} 个PPTX文件")

                for idx, in_file in enumerate(input_files, 1):
                    input_path = os.path.join(input_dir, in_file)
                    print(f"\n📄 [{idx}/{len(input_files)}] 开始处理: {in_file}")
                    
                    if in_file.lower().endswith('.pdf'):
                        print("   📚 使用PDF处理器（PyMuPDF + OCR + 表格提取）...")
                        processor.process_pdf(input_path)
                    elif in_file.lower().endswith('.pptx'):
                        if process_pptx_file_advanced is None:
                            print("   ⚠ 未安装PPTX处理依赖或导入失败，跳过PPTX文件")
                        else:
                            print("   📊 使用高级PPTX处理器（ZIP+XML解析 + 优化表格提取）...")
                            process_pptx_file_advanced(processor, input_path)
                    
                    print(f"   ✅ [{idx}/{len(input_files)}] 完成处理: {in_file}")

                print("\n🎉 所有文件处理完成！")
                print("📁 结果保存在: output/ 目录")
                
    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()