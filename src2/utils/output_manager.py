"""
输出管理器
统一管理所有类型数据的输出
"""

import os
import json
import csv
from typing import Dict, Any, List
from datetime import datetime

from .knowledge_exporter import KnowledgeExporter


def _relpath(value):
    if not value:
        return ""
    try:
        return os.path.relpath(str(value), os.getcwd()).replace(os.sep, "/")
    except Exception:
        return str(value).replace("\\", "/")


def _normalize_paths(value):
    if isinstance(value, dict):
        return {key: _normalize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item) for item in value]
    if isinstance(value, str):
        looks_like_path = (
            "\\" in value
            or value.startswith("input/")
            or value.startswith("output/")
            or value.startswith("input\\")
            or value.startswith("output\\")
        )
        return _relpath(value) if looks_like_path else value
    return value


class OutputManager:
    """统一的输出管理器"""
    
    def __init__(self, base_dir: str = "output", logger=None):
        """
        初始化输出管理器
        
        Args:
            base_dir: 输出基础目录
            logger: 日志记录器
        """
        self.base_dir = base_dir
        self.debug_dir = os.path.join(base_dir, "debug")
        self.kg_dir = os.path.join(base_dir, "kg_data")
        self.logger = logger
        self.knowledge_exporter = KnowledgeExporter(base_dir=self.kg_dir, logger=logger)
        
        # 创建输出目录结构
        self.dirs = {
            "text": os.path.join(self.debug_dir, "text"),
            "images": os.path.join(self.debug_dir, "images"),
            "formulas": os.path.join(self.debug_dir, "formulas"),
            "tables": os.path.join(self.debug_dir, "tables"),
            "code": os.path.join(self.debug_dir, "code"),
            "logs": os.path.join(self.debug_dir, "logs"),
            "document_metadata": os.path.join(self.debug_dir, "document_metadata")
        }
        
        self._create_directories()
    
    def _create_directories(self):
        """创建所有输出目录"""
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def save_text(self, data: Dict[str, Any], filename: str, page_num: int):
        """
        保存文本数据
        
        Args:
            data: 文本数据字典
            filename: 文件名（不含扩展名）
            page_num: 页码
        """
        try:
            output_file = os.path.join(
                self.dirs["text"],
                f"{filename}_page_{page_num+1}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(data), f, ensure_ascii=False, indent=2)
            self.knowledge_exporter.add_text(data, filename, page_num)
            
            if self.logger:
                self.logger.debug(f"保存文本: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存文本失败: {e}")
    
    def save_image_metadata(self, data: Dict[str, Any], filename: str, page_num: int, img_index: int):
        """
        保存图像元数据
        
        Args:
            data: 图像元数据字典
            filename: 文件名
            page_num: 页码
            img_index: 图像索引
        """
        try:
            # 使用与图片一致的命名格式，确保排序时在一起
            output_file = os.path.join(
                self.dirs["images"],
                f"{filename}_slide_{page_num+1}_img_{img_index+1}_metadata.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(data), f, ensure_ascii=False, indent=2)
            self.knowledge_exporter.add_image(data, filename, page_num, img_index)
            
            if self.logger:
                self.logger.debug(f"保存图像元数据: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存图像元数据失败: {e}")
    
    def save_formulas(self, formulas: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存公式数据（同时保存JSON和CSV）
        
        Args:
            formulas: 公式列表
            filename: 文件名
            page_num: 页码
        """
        if not formulas:
            return
        
        try:
            # 1. 保存JSON格式
            json_file = os.path.join(
                self.dirs["formulas"],
                f"{filename}_page_{page_num+1}_formulas.json"
            )
            
            output_data = {
                "page": page_num + 1,
                "formula_count": len(formulas),
                "formulas": formulas,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(output_data), f, ensure_ascii=False, indent=2)
            
            # 2. 保存CSV格式
            csv_file = os.path.join(
                self.dirs["formulas"],
                f"{filename}_page_{page_num+1}_formulas.csv"
            )
            
            with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                # 定义CSV字段
                fieldnames = ['序号', 'LaTeX公式', '描述', '来源图像', '提取方法']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for idx, formula in enumerate(formulas, 1):
                    writer.writerow({
                        '序号': idx,
                        'LaTeX公式': formula.get('latex', ''),
                        '描述': formula.get('description', ''),
                        '来源图像': _relpath(formula.get('source_image', '')),
                        '提取方法': formula.get('extraction_method', '')
                    })
            
            if self.logger:
                self.logger.info(f"保存 {len(formulas)} 个公式: JSON={json_file}, CSV={csv_file}")
            self.knowledge_exporter.add_formulas(formulas, filename, page_num)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存公式失败: {e}")
    
    def save_tables(self, tables: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存表格数据（同时保存JSON和CSV）
        
        Args:
            tables: 表格列表
            filename: 文件名
            page_num: 页码
        """
        if not tables:
            return
        
        try:
            for table_idx, table in enumerate(tables):
                # 1. 保存JSON格式
                json_file = os.path.join(
                    self.dirs["tables"],
                    f"{filename}_page_{page_num+1}_table_{table_idx+1}.json"
                )
                
                # 移除DataFrame对象（不能序列化）
                table_copy = table.copy()
                if 'dataframe' in table_copy:
                    del table_copy['dataframe']
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(_normalize_paths(table_copy), f, ensure_ascii=False, indent=2)
                
                # 2. 保存CSV格式
                csv_file = os.path.join(
                    self.dirs["tables"],
                    f"{filename}_page_{page_num+1}_table_{table_idx+1}.csv"
                )
                
                # 使用table中的json数据来生成CSV
                if 'json' in table and table['json']:
                    with open(csv_file, 'w', encoding='utf-8-sig', newline='') as f:
                        if len(table['json']) > 0:
                            fieldnames = list(table['json'][0].keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(table['json'])
            
            if self.logger:
                self.logger.info(f"保存 {len(tables)} 个表格 (JSON + CSV)")
            self.knowledge_exporter.add_tables(tables, filename, page_num)
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存表格失败: {e}")
    
    def save_code(self, code_blocks: List[Dict[str, Any]], filename: str, page_num: int):
        """
        保存代码数据（只保存2个文件：纯源代码文件 + JSON元数据）
        
        Args:
            code_blocks: 代码块列表
            filename: 文件名
            page_num: 页码
        """
        if not code_blocks:
            return
        
        try:
            for idx, code_block in enumerate(code_blocks):
                language = code_block.get('language', 'txt').lower()
                
                # 根据语言确定文件扩展名
                language_extensions = {
                    "python": ".py",
                    "java": ".java",
                    "javascript": ".js",
                    "c++": ".cpp",
                    "cpp": ".cpp",
                    "c": ".c",
                    "go": ".go",
                    "rust": ".rs",
                    "typescript": ".ts",
                    "php": ".php",
                    "ruby": ".rb",
                    "swift": ".swift",
                    "kotlin": ".kt",
                    "matlab": ".m",
                    "r": ".r",
                    "sql": ".sql",
                    "html": ".html",
                    "css": ".css",
                    "arduino": ".ino",
                    "ino": ".ino",
                    "shell": ".sh",
                    "bash": ".sh",
                    "txt": ".txt"
                }
                
                # 如果语言无法识别或为unknown，统一使用.txt
                if language == "unknown" or language not in language_extensions:
                    ext = ".txt"
                else:
                    ext = language_extensions.get(language, ".txt")
                
                # 1. 保存纯源代码文件（不添加任何额外内容）
                code_file = os.path.join(
                    self.dirs["code"],
                    f"{filename}_page_{page_num+1}_code_{idx+1}{ext}"
                )
                
                # 只写入纯代码，不添加注释或其他内容
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code_block.get('code', ''))
                
                # 2. 保存JSON元数据（包含所有辅助信息）
                metadata_file = os.path.join(
                    self.dirs["code"],
                    f"{filename}_page_{page_num+1}_code_{idx+1}_metadata.json"
                )
                
                metadata = {
                    "source_image": _relpath(code_block.get('source_image', '')),
                    "extraction_method": code_block.get('extraction_method', ''),
                    "language": language,
                    "description": code_block.get('description', ''),
                    "code_file": os.path.basename(code_file),
                    "page": page_num + 1,
                    "index": idx + 1,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": code_block.get('raw_response', '')
                }
                
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(_normalize_paths(metadata), f, ensure_ascii=False, indent=2)
                
                if self.logger:
                    self.logger.info(f"保存代码: {os.path.basename(code_file)} + metadata.json")
            self.knowledge_exporter.add_code(code_blocks, filename, page_num)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存代码失败: {e}")
    
    def save_metadata(self, metadata: Dict[str, Any], filename: str):
        """
        保存文档元数据
        
        Args:
            metadata: 元数据字典
            filename: 文件名
        """
        try:
            output_file = os.path.join(self.dirs["document_metadata"], f"{filename}_metadata.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(_normalize_paths(metadata), f, ensure_ascii=False, indent=2)
            self.knowledge_exporter.register_document(filename, metadata)
            # Refresh structured JSON after each document. Paid embeddings are
            # generated once, after every input document has been processed.
            self.knowledge_exporter.finalize(build_vectors=False)
            
            if self.logger:
                self.logger.info(f"保存元数据: {output_file}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存元数据失败: {e}")
    
    def save_professional_terms(self, terms: set, filename: str = "professional_terms_library"):
        """
        保存专业术语库
        
        Args:
            terms: 术语集合
            filename: 文件名
        """
        try:
            output_file = os.path.join(self.debug_dir, f"{filename}.json")
            
            terms_data = {
                "total_terms": len(terms),
                "terms": sorted(list(terms)),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(terms_data, f, ensure_ascii=False, indent=2)
            
            if self.logger:
                self.logger.info(f"保存专业术语库: {output_file} ({len(terms)} 个术语)")
        except Exception as e:
            if self.logger:
                self.logger.error(f"保存专业术语库失败: {e}")
    
    def _get_code_extension(self, language: str) -> str:
        """获取代码文件扩展名"""
        extensions = {
            'python': 'py',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'javascript': 'js',
            'typescript': 'ts',
            'matlab': 'm',
            'r': 'r',
            'sql': 'sql',
        }
        return extensions.get(language.lower(), 'txt')
    
    def finalize(self):
        """Write the final handoff files and build embeddings exactly once."""
        self.knowledge_exporter.finalize(build_vectors=True)

    def get_output_summary(self) -> Dict[str, Any]:
        """
        获取输出摘要
        
        Returns:
            输出统计信息
        """
        summary = {}
        
        for name, dir_path in self.dirs.items():
            try:
                files = os.listdir(dir_path)
                summary[name] = {
                    "count": len(files),
                    "path": dir_path
                }
            except Exception:
                summary[name] = {
                    "count": 0,
                    "path": dir_path
                }
        summary["kg_data"] = {
            "count": len(os.listdir(self.kg_dir)) if os.path.exists(self.kg_dir) else 0,
            "path": self.kg_dir,
        }
        summary["debug"] = {
            "count": len(os.listdir(self.debug_dir)) if os.path.exists(self.debug_dir) else 0,
            "path": self.debug_dir,
        }
        return summary
