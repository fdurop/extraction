"""
表格提取模块
功能：从PDF、PPTX中提取表格数据
"""

import re
import pandas as pd
from typing import List, Dict, Any, Optional


class TableExtractor:
    def __init__(self, logger=None):
        """
        初始化表格提取器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
    
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
        tables = []
        
        if method == "auto" or method == "camelot":
            # 尝试使用camelot
            camelot_tables = self._extract_with_camelot(page)
            if camelot_tables:
                tables.extend(camelot_tables)
                return tables
        
        if method == "auto" or method == "pdfplumber":
            # 尝试使用pdfplumber
            pdfplumber_tables = self._extract_with_pdfplumber(page)
            if pdfplumber_tables:
                tables.extend(pdfplumber_tables)
                return tables
        
        if method == "auto" or method == "text":
            # 从文本中检测表格
            text_tables = self._extract_from_text(page_text)
            if text_tables:
                tables.extend(text_tables)
        
        return tables
    
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
            
            # 转换为DataFrame
            if data:
                # 假设第一行是表头
                if len(data) > 1:
                    df = pd.DataFrame(data[1:], columns=data[0])
                else:
                    df = pd.DataFrame(data)
                
                return {
                    "type": "pptx_table",
                    "rows": rows,
                    "cols": cols,
                    "data": data,
                    "dataframe": df,
                    "csv": df.to_csv(index=False),
                    "json": df.to_dict(orient='records')
                }
        except Exception as e:
            if self.logger:
                self.logger.error(f"PPTX表格提取失败: {e}")
            return None
    
    def _extract_with_camelot(self, page) -> List[Dict[str, Any]]:
        """使用Camelot提取表格"""
        try:
            import camelot
            import tempfile
            import fitz
            
            # Camelot需要文件路径，创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                # 从页面创建临时PDF
                doc = fitz.open()
                doc.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
                doc.save(tmp.name)
                doc.close()
                
                # 使用Camelot提取
                tables = camelot.read_pdf(tmp.name, pages='1', flavor='lattice')
                
                result = []
                for idx, table in enumerate(tables):
                    df = table.df
                    result.append({
                        "type": "camelot",
                        "method": "lattice",
                        "accuracy": table.accuracy,
                        "dataframe": df,
                        "csv": df.to_csv(index=False),
                        "json": df.to_dict(orient='records')
                    })
                
                return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Camelot提取失败: {e}")
            return []
    
    def _extract_with_pdfplumber(self, page) -> List[Dict[str, Any]]:
        """使用pdfplumber提取表格"""
        try:
            import pdfplumber
            import tempfile
            
            # 创建临时PDF
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                doc = page.parent
                # 保存单页
                single_doc = fitz.open()
                single_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
                single_doc.save(tmp.name)
                single_doc.close()
                
                # 使用pdfplumber
                with pdfplumber.open(tmp.name) as pdf:
                    first_page = pdf.pages[0]
                    tables = first_page.extract_tables()
                    
                    result = []
                    for table in tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            result.append({
                                "type": "pdfplumber",
                                "dataframe": df,
                                "csv": df.to_csv(index=False),
                                "json": df.to_dict(orient='records')
                            })
                    
                    return result
        except Exception as e:
            if self.logger:
                self.logger.debug(f"pdfplumber提取失败: {e}")
            return []
    
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
                        "separator": sep,
                        "dataframe": df,
                        "csv": df.to_csv(index=False),
                        "json": df.to_dict(orient='records')
                    }
            
            return None
        except Exception as e:
            if self.logger:
                self.logger.debug(f"文本表格解析失败: {e}")
            return None
