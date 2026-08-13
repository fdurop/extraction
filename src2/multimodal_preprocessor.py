"""
多模态数据提取系统 - 重构版本
模块化架构：分离核心处理逻辑与文档格式解析
"""

import sys
import os
import shutil
import re

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 查找项目根目录并切换工作目录（确保input/output路径正确）
def find_and_set_project_root():
    """查找项目根目录并设置为工作目录"""
    current = os.path.abspath(current_dir)
    for _ in range(5):
        if os.path.isfile(os.path.join(current, 'run_src2.py')) and os.path.isdir(
            os.path.join(current, 'src2')
        ):
            os.chdir(current)
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    parent = os.path.dirname(current_dir)
    if os.path.isfile(os.path.join(parent, 'run_src2.py')):
        os.chdir(parent)
        return parent
    raise RuntimeError("无法定位项目根目录：需要 run_src2.py 和 src2/ 目录")

project_root = find_and_set_project_root()

# 添加src2目录到路径（这样可以直接导入core、parsers、utils）
sys.path.insert(0, current_dir)

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None

# 导入模块
from core import ImageProcessor, TextProcessor, FormulaExtractor, TableExtractor, CodeExtractor
from parsers import PDFParser, PPTXParser
from utils import OutputManager, FileUtils, create_vlm_client, load_vlm_config

# 导入日志配置（src2目录下已有logger_config.py）
from logger_config import get_logger, LoggerSetup


def _build_output_base_dir(vlm_config):
    provider = str(vlm_config.get("provider") or "qwen").strip().lower()
    model = str(vlm_config.get("model") or "model").strip().lower()

    def safe(name):
        chars = []
        for ch in name:
            if ch.isalnum() or ch in {"-", "_", "."}:
                chars.append(ch)
            else:
                chars.append("_")
        cleaned = "".join(chars).strip("._")
        return cleaned or "model"

    base_name = f"{safe(provider)}_{safe(model)}"
    output_root = "output"
    os.makedirs(output_root, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    max_index = 0
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        if not os.path.isdir(path):
            continue
        match = pattern.match(name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return os.path.join(output_root, f"{base_name}_{max_index + 1}")


class MultimodalPreprocessor:
    """多模态数据预处理器 - 重构版"""
    
    def __init__(self):
        """初始化基于远程多模态 API 的预处理工具。"""
        # 初始化日志系统
        self.logger = get_logger("MultimodalPreprocessor")
        LoggerSetup.log_session_start(self.logger)
        
        print("\n" + "=" * 60)
        print("多模态数据提取系统 - 重构版 v2.0")
        print("=" * 60)
        print(f"   工作目录: {os.getcwd()}")
        self.vlm_config = load_vlm_config()
        self.output_base_dir = _build_output_base_dir(self.vlm_config)
        print(
            f"   配置: API-VLM={self.vlm_config.get('provider', 'qwen')}/"
            f"{self.vlm_config.get('model', 'model')}"
        )
        print(f"   输出目录: {self.output_base_dir}")
        
        self.logger.info("开始初始化多模态预处理工具（重构版）")
        self.logger.info("运行模式: remote_api")
        print("\n[步骤 1/4] 检查远程 API 配置...")
        
        # 初始化OCR引擎
        print("\n[步骤 2/4] 初始化OCR引擎...")
        try:
            if PaddleOCR is None:
                raise ImportError("paddleocr is not installed")
            # 尝试使用show_log参数（新版本）
            try:
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
            except TypeError:
                # 如果不支持show_log参数，使用旧版本方式
                self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='ch')
            print("   [OK] PaddleOCR初始化成功")
            self.logger.info("PaddleOCR初始化成功")
        except Exception as e:
            print(f"   [WARN] PaddleOCR初始化失败: {e}")
            print("   [INFO] OCR功能将不可用，但不影响其他功能")
            self.logger.warning(f"PaddleOCR初始化失败: {e}")
            self.ocr_engine = None
        
        print("\n[步骤 3/4] 初始化多模态 API...")
        self.vlm_client = create_vlm_client(logger=self.logger)
        if self.vlm_client:
            print(f"   [OK] API多模态模型已启用: {self.vlm_client.provider}/{self.vlm_client.model}")
            self.logger.info(f"API VLM enabled: {self.vlm_client.provider}/{self.vlm_client.model}")
        elif self.vlm_config.get("enabled", True):
            print(f"   [WARN] API多模态模型未启用：请填写 {self.vlm_config.get('config_path', 'config/vlm_api.yaml')} 中的 api_key")
            self.logger.warning("API VLM is not available; image descriptions will be limited.")

        print("\n[步骤 4/4] 初始化核心处理与输出模块...")
        self.image_processor = ImageProcessor(
            logger=self.logger,
            vlm_client=self.vlm_client
        )
        self.text_processor = TextProcessor(logger=self.logger)
        self.formula_extractor = FormulaExtractor(
            ocr_engine=self.ocr_engine,
            deepseek_wrapper=self.vlm_client,
            logger=self.logger
        )
        self.table_extractor = TableExtractor(logger=self.logger)
        self.code_extractor = CodeExtractor(logger=self.logger)
        print("   [OK] 核心处理模块初始化完成")
        
        self.output_manager = OutputManager(base_dir=self.output_base_dir, logger=self.logger)
        print("   [OK] 输出目录创建完成")
        
        print("\n" + "=" * 60)
        print(" 初始化完成！系统就绪")
        print("=" * 60)
        log_path = os.path.join(self.output_base_dir, "debug", "logs", "app.log")
        if os.path.exists(log_path):
            print(f" 日志文件: {log_path}")
        print("")
        
        self.logger.info("多模态预处理工具初始化完成")
        
        # 专业术语库
        self.professional_terms = set()

    def _vlm_source_metadata(self):
        if self.vlm_client:
            return {
                "extraction_method": f"{self.vlm_client.provider}_api_智能识别",
                "backend": "api",
                "provider": self.vlm_client.provider,
                "model": self.vlm_client.model,
            }
        return {
            "extraction_method": "none",
            "backend": "none",
            "provider": None,
            "model": None,
        }
    
    def process_file(self, file_path: str):
        """
        处理文件（自动识别格式）
        
        Args:
            file_path: 文件路径
        """
        self.logger.info(f"开始处理文件: {file_path}")
        print(f"\n{'='*60}")
        print(f" 处理文件: {file_path}")
        print(f"{'='*60}")
        
        if FileUtils.is_pdf(file_path):
            self.process_pdf(file_path)
        elif FileUtils.is_pptx(file_path):
            self.process_pptx(file_path)
        else:
            print(f"   [WARN] 不支持的文件格式: {file_path}")
            self.logger.warning(f"不支持的文件格式: {file_path}")
        
        print(f"   [OK] 文件处理完成\n")
    
    def process_pdf(self, file_path: str):
        """
        处理PDF文件
        
        Args:
            file_path: PDF文件路径
        """
        print(f"\n[PDF处理器] 开始处理: {file_path}")
        self.logger.info(f"开始处理PDF: {file_path}")
        
        # 使用PDF解析器
        parser = PDFParser(
            logger=self.logger,
            output_images_dir=os.path.join(self.output_base_dir, "debug", "images"),
        )
        result = parser.parse(file_path)
        
        if not result.get("success"):
            print(f"[ERROR] PDF解析失败")
            return
        
        original_filename = result["filename"]
        filename = f"{original_filename}_pdf"
        pages = result["pages"]
        result["metadata"].update({
            "file_name": original_filename,
            "file_type": "pdf",
            "file_path": file_path,
        })
        
        print(f"   总页数: {len(pages)}")
        
        # 处理每一页
        for page_data in pages:
            page_num = page_data["page_num"]
            print(f"\n   处理第 {page_num + 1}/{len(pages)} 页...")
            
            # 1. 文本处理
            text_result = self.text_processor.process_text(
                page_data["text"],
                page_num,
                page_data["images"]
            )
            self.output_manager.save_text(text_result, filename, page_num)
            
            # 更新术语库
            self.professional_terms.update(text_result.get("technical_terms", []))
            
            # 2. 图像处理
            for img_idx, img_path in enumerate(page_data["images"]):
                img_result = self.image_processor.process_image(img_path, page_data["text"])
                self.output_manager.save_image_metadata(img_result, filename, page_num, img_idx)
            
            # 3. 公式提取
            formulas = self.formula_extractor.extract_formulas_from_text(
                page_data["text"],
                context=page_data["text"][:500]
            )
            if formulas:
                self.output_manager.save_formulas(formulas, filename, page_num)
            
            # 4. 表格提取
            if page_data.get("raw_page"):
                tables = self.table_extractor.extract_tables_from_pdf_page(
                    page_data["raw_page"],
                    page_data["text"]
                )
                if tables:
                    self.output_manager.save_tables(tables, filename, page_num)
            
            # 5. 代码提取
            code_blocks = self.code_extractor.extract_code_from_text(page_data["text"])
            if code_blocks:
                self.output_manager.save_code(code_blocks, filename, page_num)
        
        # 保存元数据
        self.output_manager.save_metadata(result["metadata"], filename)
        
        print(f"\n[OK] PDF处理完成: {filename}")
        self.logger.info(f"PDF处理完成: {filename}")
    
    def process_pptx(self, file_path: str):
        """
        处理PPTX文件
        
        Args:
            file_path: PPTX文件路径
        """
        print(f"\n[PPTX处理器] 开始处理: {file_path}")
        self.logger.info(f"开始处理PPTX: {file_path}")
        
        # 使用PPTX解析器
        parser = PPTXParser(
            logger=self.logger,
            output_images_dir=os.path.join(self.output_base_dir, "debug", "images"),
        )
        result = parser.parse(file_path)
        
        if not result.get("success"):
            print(f"[ERROR] PPTX解析失败")
            return
        
        original_filename = result["filename"]
        filename = f"{original_filename}_pptx"
        pages = result["pages"]
        result["metadata"].update({
            "file_name": original_filename,
            "file_type": "pptx",
            "file_path": file_path,
        })
        image_mapping = result["metadata"].get("image_mapping", {})
        
        print(f"   总幻灯片数: {len(pages)}")
        
        # 处理每一张幻灯片
        for page_data in pages:
            page_num = page_data["page_num"]
            print(f"\n   处理第 {page_num + 1}/{len(pages)} 张幻灯片...")
            
            # 1. 文本处理
            images = image_mapping.get(page_num + 1, [])
            text_result = self.text_processor.process_text(
                page_data["text"],
                page_num,
                images
            )
            self.output_manager.save_text(text_result, filename, page_num)
            
            # 更新术语库
            self.professional_terms.update(text_result.get("technical_terms", []))
            
            # 2. 图像处理（使用ZIP方法提取的图像）
            for img_idx, img_path in enumerate(images):
                # 跳过不存在的图像
                if not os.path.exists(img_path):
                    continue
                
                print(f"   → 处理图像 {img_idx + 1}/{len(images)}: {os.path.basename(img_path)}")
                print(f"      ✓ 图片已保存: {os.path.basename(img_path)}")
                    
                # 处理图片（生成描述）
                img_result = self.image_processor.process_image(img_path, page_data["text"])
                
                # 只保存成功处理的图像
                if img_result.get("error"):
                    print(f"      ✗ 图像处理失败: {img_result.get('error')}")
                    continue
                
                # 保存图像元数据JSON（图片已在parser中保存，这里只保存JSON）
                self.output_manager.save_image_metadata(img_result, filename, page_num, img_idx)
                print(f"      ✓ JSON已保存: {filename}_slide_{page_num+1}_img_{img_idx+1}_metadata.json")
                
                # === 智能识别：根据图像描述判断是否需要进一步识别公式或代码 ===
                general_description = img_result.get("description", "")
                
                if general_description and self.deepseek_wrapper is not None:
                    # 检测公式关键词（更严格）
                    formula_keywords = ['公式', '方程', '数学表达式', '计算式', 'formula', 'equation', 
                                       'LaTeX', '积分', '微分', '导数', '求导', '函数式', '数学式']
                    has_formula = any(keyword in general_description for keyword in formula_keywords)
                    
                    # 检测代码关键词（更严格，排除误判）
                    # 必须包含明确的代码相关词汇
                    code_keywords_strong = ['代码', '程序', 'code snippet', 'program', '编程', '源代码', 
                                           'void setup', 'void loop', 'int main', 'def ', 'class ', 
                                           'function ', '#include', 'import ', 'pinMode', 'digitalWrite']
                    has_code = any(keyword in general_description for keyword in code_keywords_strong)
                    
                    # 排除假阳性：如果只是提到"Arduino"但没有明确的代码词汇，不算
                    weak_indicators = ['Arduino', '单片机', 'microcontroller']
                    if not has_code and any(weak in general_description for weak in weak_indicators):
                        # 二次检查：是否真的提到了代码
                        secondary_check = ['代码', 'code', '程序', 'program', '语句', 'statement']
                        has_code = any(check in general_description for check in secondary_check)
                    
                    # 检测表格关键词
                    table_keywords = ['表格', 'table', '数据表', '统计表', '对比表']
                    has_table = any(keyword in general_description for keyword in table_keywords)
                    
                    # 如果检测到公式，进行专门的公式识别
                    if has_formula:
                        try:
                            print(f"      ↳ [智能检测] 发现公式关键词，启动专门公式识别...")
                            formula_result = self.deepseek_wrapper.recognize_formula(img_path)
                            
                            if formula_result and formula_result.get("latex"):
                                # 保存公式结果
                                source_metadata = self._vlm_source_metadata()
                                formula_data = {
                                    "source_image": img_path,
                                    **source_metadata,
                                    "latex": formula_result.get("latex", ""),
                                    "description": formula_result.get("description", ""),
                                    "raw_response": formula_result.get("raw_response", ""),
                                    "page_num": page_num,
                                    "image_index": img_idx
                                }
                                self.output_manager.save_formulas([formula_data], filename, page_num)
                                print(f"      [OK] 公式识别并保存完成")
                            else:
                                print(f"      ✗ 公式识别未返回有效结果")
                        except Exception as e:
                            self.logger.error(f"公式识别失败: {e}")
                            print(f"      ✗ 公式识别失败: {e}")
                    
                    # 如果检测到代码，进行专门的代码识别
                    if has_code:
                        try:
                            print(f"      ↳ [智能检测] 发现代码关键词，启动专门代码识别...")
                            code_result = self.deepseek_wrapper.recognize_code(img_path)
                            
                            # 三重检测：has_code=True 且 code非空
                            if code_result and code_result.get("has_code") and code_result.get("code"):
                                # 保存代码结果
                                source_metadata = self._vlm_source_metadata()
                                code_data = {
                                    "source_image": img_path,
                                    **source_metadata,
                                    "code": code_result.get("code", ""),
                                    "language": code_result.get("language", "txt"),
                                    "description": code_result.get("description", ""),
                                    "raw_response": code_result.get("raw_response", ""),
                                    "page_num": page_num,
                                    "image_index": img_idx
                                }
                                self.output_manager.save_code([code_data], filename, page_num)
                                print(f"      [OK] 代码识别并保存完成")
                            else:
                                print(f"      [跳过] 图片中无实际代码内容")
                        except Exception as e:
                            self.logger.error(f"代码识别失败: {e}")
                            print(f"      ✗ 代码识别失败: {e}")
                    
                    # 如果检测到表格，进行专门的表格识别
                    if has_table:
                        try:
                            print(f"      ↳ [智能检测] 发现表格关键词，启动专门表格识别...")
                            table_result = self.deepseek_wrapper.recognize_table(img_path)
                            
                            if table_result and table_result.get("table_data"):
                                # 保存表格结果（简化格式）
                                source_metadata = self._vlm_source_metadata()
                                table_data = {
                                    "source_image": img_path,
                                    **source_metadata,
                                    "headers": table_result.get("headers", ""),
                                    "content": table_result.get("table_data", ""),
                                    "description": table_result.get("description", ""),
                                    "raw_response": table_result.get("raw_response", ""),
                                    "json": [],  # 可以扩展为结构化数据
                                    "page_num": page_num,
                                    "image_index": img_idx
                                }
                                self.output_manager.save_tables([table_data], filename, page_num)
                                print(f"      [OK] 表格识别并保存完成")
                            else:
                                print(f"      ✗ 表格识别未返回有效结果")
                        except Exception as e:
                            self.logger.error(f"表格识别失败: {e}")
                            print(f"      ✗ 表格识别失败: {e}")
            
            # 3. 公式提取
            formulas = self.formula_extractor.extract_formulas_from_text(
                page_data["text"],
                context=page_data["text"][:500]
            )
            if formulas:
                self.output_manager.save_formulas(formulas, filename, page_num)
            
            # 4. 表格提取
            for table_info in page_data.get("tables", []):
                table_data = self.table_extractor.extract_tables_from_pptx_shape(
                    table_info["shape"]
                )
                if table_data:
                    self.output_manager.save_tables([table_data], filename, page_num)
            
            # 5. 代码提取
            code_blocks = self.code_extractor.extract_code_from_text(page_data["text"])
            if code_blocks:
                self.output_manager.save_code(code_blocks, filename, page_num)
        
        # 保存元数据
        self.output_manager.save_metadata(result["metadata"], filename)
        
        # 保存专业术语库
        if self.professional_terms:
            self.output_manager.save_professional_terms(self.professional_terms)
            print(f"   [OK] 专业术语库已保存 ({len(self.professional_terms)} 个术语)")
        
        print(f"\n   [OK] PPTX处理完成: {filename}")
        self.logger.info(f"PPTX处理完成: {filename}")
    
    def get_processing_summary(self):
        """获取处理摘要"""
        summary = self.output_manager.get_output_summary()
        summary["professional_terms_count"] = len(self.professional_terms)
        return summary


def main():
    """主函数"""
    # 初始化主日志
    main_logger = get_logger("Main")
    LoggerSetup.log_session_start(main_logger)
    
    main_logger.info("多模态数据预处理器启动（重构版）")
    
    try:
        max_files_raw = os.getenv("EXTRACTION_MAX_FILES", "").strip()
        max_files = int(max_files_raw) if max_files_raw.isdigit() else None

        # 初始化处理器
        processor = MultimodalPreprocessor()
        main_logger.info("处理器初始化完成")
        
        # 检查输入目录中的文件
        input_dir = "input"
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            print(f"[ERROR] 输入目录不存在，已创建: {input_dir}")
            print("请将PDF或PPTX文件放入input目录后重新运行")
            return
        
        # 查找所有支持的文件
        input_files = FileUtils.get_files(input_dir, extensions=['.pdf', '.pptx'])
        if max_files is not None:
            input_files = input_files[:max_files]
        
        if not input_files:
            print(f"[ERROR] 未在 {input_dir} 目录找到PDF或PPTX文件")
            print("请将文件放入input目录后重新运行")
            return
        
        print(f"\n找到 {len(input_files)} 个待处理文件")
        main_logger.info(f"找到 {len(input_files)} 个待处理文件")
        
        # 处理所有文件
        for idx, file_path in enumerate(input_files, 1):
            print(f"\n[{idx}/{len(input_files)}] 正在处理: {os.path.basename(file_path)}")
            main_logger.info(f"开始处理文件 [{idx}/{len(input_files)}]: {file_path}")
            
            processor.process_file(file_path)
            
            print(f"   [OK] [{idx}/{len(input_files)}] 完成")
            main_logger.info(f"完成处理文件 [{idx}/{len(input_files)}]: {file_path}")
        
        # 保存专业术语库
        if processor.professional_terms:
            processor.output_manager.save_professional_terms(processor.professional_terms)
        
        # 显示处理摘要
        print("\n" + "=" * 60)
        print("所有文件处理完成！")
        print("=" * 60)
        
        summary = processor.get_processing_summary()
        print(f"\n输出摘要:")
        for name, info in summary.items():
            if name == "professional_terms_count":
                print(f"   专业术语: {info} 个")
            else:
                print(f"   {name}: {info['count']} 个文件")
        
        print(f"\n结果保存目录: {processor.output_base_dir}/")
        print("   ├─ kg_data/  (下游知识图谱和向量检索交付文件)")
        print("   └─ debug/    (人工检查和中间调试文件)")

        log_file = LoggerSetup.get_log_file_path()
        if log_file and os.path.exists(log_file):
            output_log_dir = os.path.join(processor.output_base_dir, "debug", "logs")
            os.makedirs(output_log_dir, exist_ok=True)
            output_log_file = os.path.join(output_log_dir, os.path.basename(log_file))
            shutil.copy2(log_file, output_log_file)
            print(f"   日志副本: {output_log_file}")
            main_logger.info(f"日志副本已保存: {output_log_file}")
        
        main_logger.info("所有文件处理完成")
        
    except Exception as e:
        print(f"\n[ERROR] 处理过程出错: {e}")
        main_logger.error(f"处理过程出错: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
