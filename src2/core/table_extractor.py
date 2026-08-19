"""
表格提取模块
功能：从PDF、PPTX中提取表格数据
"""

import re
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from .table_validator import TableValidator


class TableExtractor:
    def __init__(self, logger=None, vlm_client=None):
        """
        初始化表格提取器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.vlm_client = vlm_client
        self.validator = TableValidator()
    
    def extract_tables_from_pdf_page(self, page, page_text: str = "", method: str = "auto") -> List[Dict[str, Any]]:
        """
        从PDF页面提取表格
        
        Args:
            page: PyMuPDF页面对象
            page_text: 页面文本
            method: 提取方法 (auto/camelot/pdfplumber/text)
            
        Returns:
            表格列表
        """
        candidates = []

        if method == "auto":
            candidates.extend(self._extract_with_pymupdf(page))
        if method in {"auto", "camelot"}:
            candidates.extend(self._extract_with_camelot(page))
        if method in {"auto", "pdfplumber"}:
            candidates.extend(self._extract_with_pdfplumber(page))
        if method in {"auto", "text"} and not candidates:
            candidates.extend(self._extract_from_text(page_text))

        validated = self.validator.validate_many(candidates)
        accepted = [table for table in validated if not table.get("review_required")]
        return accepted or validated
    
    def extract_tables_from_pptx_shape(self, shape) -> Optional[Dict[str, Any]]:
        """
        从PPTX形状中提取表格
        
        Args:
            shape: python-pptx表格形状
            
        Returns:
            表格数据字典
        """
        try:
            if not hasattr(shape, 'table'):
                return None
            
            table = shape.table
            rows = len(table.rows)
            cols = len(table.columns)
            
            # 提取表格数据
            data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    row_data.append(cell_text)
                data.append(row_data)
            
            # Preserve the original cell matrix; use the first row as the
            # default header while retaining the raw matrix for review.
            if data:
                if len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                else:
                    df = pd.DataFrame(data)
                
                return {
                    "type": "pptx_table",
                    "rows": rows,
                    "cols": cols,
                    "data": data,
                    "headers": [data[0]] if len(data) > 1 else [],
                    "cells": data[1:] if len(data) > 1 else data,
                    "raw_cells": data,
                    "source_type": "pptx_native_table",
                    "extraction_method": "python_pptx",
                    "dataframe": df,
                    "csv": df.to_csv(index=False),
                    "json": df.to_dict(orient='records')
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"PPTX表格提取失败: {e}")
            return None

    def should_analyze_image(self, description: str) -> bool:
        description = str(description or "")
        keywords = ("表格", "数据表", "统计表", "对比表", "参数表", "真值表", "table", "tabular")
        return any(keyword.lower() in description.lower() for keyword in keywords)

    def extract_tables_from_vlm_image(
        self,
        image_path: str,
        context: str = "",
        image_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not self.vlm_client:
            return []
        try:
            response = self.vlm_client.recognize_table(image_path, context=context)
            if not response.get("has_table", True):
                return []
            table = {
                **response,
                "source_image": image_path,
                "source_type": "image_vlm_table",
                "extraction_method": f"{getattr(self.vlm_client, 'provider', 'vlm')}_api_table",
                "backend": "api",
                "provider": getattr(self.vlm_client, "provider", None),
                "model": getattr(self.vlm_client, "model", None),
                "image_index": image_index,
            }
            return [table]
        except Exception as exc:
            if self.logger:
                self.logger.error(f"表格视觉识别失败: {exc}")
            return [{
                "source_image": image_path,
                "source_type": "image_vlm_table",
                "extraction_method": "vlm_table",
                "headers": [],
                "cells": [],
                "api_error": str(exc),
                "review_required": True,
                "review_reason": "api_failed",
                "image_index": image_index,
            }]

    def finalize_page(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.validator.validate_many(tables)

    def _extract_with_pymupdf(self, page) -> List[Dict[str, Any]]:
        """Use PyMuPDF's native table finder when available."""
        try:
            finder = page.find_tables()
            result = []
            for table in getattr(finder, "tables", []):
                matrix = table.extract() or []
                if not matrix:
                    continue
                result.append({
                    "type": "pymupdf",
                    "source_type": "pdf_native_table",
                    "extraction_method": "pymupdf_find_tables",
                    "headers": [matrix[0]] if len(matrix) > 1 else [],
                    "cells": matrix[1:] if len(matrix) > 1 else matrix,
                    "raw_cells": matrix,
                    "bbox": list(table.bbox) if getattr(table, "bbox", None) else None,
                    "accuracy": 0.92,
                })
            return result
        except Exception as exc:
            if self.logger:
                self.logger.debug(f"PyMuPDF表格提取失败: {exc}")
            return []
    
    def _extract_with_camelot(self, page) -> List[Dict[str, Any]]:
        """使用Camelot提取表格"""
        tmp_name = None
        try:
            import camelot
            import tempfile
            import fitz

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            doc = fitz.open()
            doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            doc.save(tmp_name)
            doc.close()

            result = []
            for flavor in ("lattice", "stream"):
                try:
                    tables = camelot.read_pdf(tmp_name, pages='1', flavor=flavor)
                    for table in tables:
                        matrix = table.df.fillna("").astype(str).values.tolist()
                        result.append({
                            "type": "camelot",
                            "source_type": "pdf_native_table",
                            "extraction_method": f"camelot_{flavor}",
                            "method": flavor,
                            "accuracy": table.accuracy,
                            "headers": [matrix[0]] if len(matrix) > 1 else [],
                            "cells": matrix[1:] if len(matrix) > 1 else matrix,
                            "raw_cells": matrix,
                            "bbox": list(getattr(table, "_bbox", [])) or None,
                        })
                except Exception as exc:
                    if self.logger:
                        self.logger.debug(f"Camelot {flavor}提取失败: {exc}")
            return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Camelot提取失败: {e}")
            return []
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    
    def _extract_with_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """使用pdfplumber提取表格"""
        tmp_name = None
        try:
            import pdfplumber
            import tempfile
            import fitz

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            single_doc = fitz.open()
            single_doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
            single_doc.save(tmp_name)
            single_doc.close()

            with pdfplumber.open(tmp_name) as pdf:
                tables = pdf.pages[0].extract_tables()
                result = []
                for table in tables:
                    if table:
                        result.append({
                            "type": "pdfplumber",
                            "source_type": "pdf_native_table",
                            "extraction_method": "pdfplumber",
                            "headers": [table[0]] if len(table) > 1 else [],
                            "cells": table[1:] if len(table) > 1 else table,
                            "raw_cells": table,
                            "accuracy": 0.86,
                        })
            return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"pdfplumber提取失败: {e}")
            return []
        finally:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
    
    def _extract_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中检测并提取表格"""
        tables = []
        
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            # 检测表格行（包含多个分隔符）
            if re.search(r'[\|\t]{2,}', line) or re.search(r'\s{3,}', line):
                table_lines.append(line)
                in_table = True
            else:
                if in_table and len(table_lines) >= 2:
                    # 处理检测到的表格
                    table_data = self._parse_text_table(table_lines)
                    if table_data:
                        tables.append(table_data)
                table_lines = []
                in_table = False
        
        # 处理文件末尾的表格
        if in_table and len(table_lines) >= 2:
            table_data = self._parse_text_table(table_lines)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_text_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """解析文本表格"""
        try:
            # 尝试不同的分隔符
            separators = ['|', '\t', r'\s{2,}']
            
            for sep in separators:
                rows = []
                for line in lines:
                    if sep == r'\s{2,}':
                        # 使用多个空格分隔
                        cells = re.split(r'\s{2,}', line.strip())
                    else:
                        cells = [c.strip() for c in line.split(sep)]
                    
                    if len(cells) > 1:
                        rows.append(cells)
                
                if len(rows) >= 2 and len(set(len(r) for r in rows)) == 1:
                    # 所有行列数相同，可能是有效表格
                    df = pd.DataFrame(rows[1:], columns=rows[0])
                    return {
                        "type": "text_table",
                        "source_type": "pdf_text_table",
                        "extraction_method": "text_spacing_rule",
                        "separator": sep,
                        "headers": [rows[0]],
                        "cells": rows[1:],
                        "dataframe": df,
                        "csv": df.to_csv(index=False),
                        "json": df.to_dict(orient='records')
                    }
            
            return None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"文本表格解析失败: {e}")
            return None
