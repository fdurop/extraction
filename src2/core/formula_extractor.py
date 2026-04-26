"""
公式提取模块
功能：从文本和图像中提取数学公式
"""

import re
import os
from typing import List, Dict, Any, Optional
from PIL import Image


class FormulaExtractor:
    def __init__(self, ocr_engine=None, deepseek_wrapper=None, logger=None):
        """
        初始化公式提取器
        
        Args:
            ocr_engine: OCR引擎（如PaddleOCR）
            deepseek_wrapper: DeepSeek-VL包装器
            logger: 日志记录器
        """
        self.ocr_engine = ocr_engine
        self.deepseek_wrapper = deepseek_wrapper
        self.logger = logger
    
    def extract_formulas_from_text(self, text: str, context: str = "") -> List[Dict[str, Any]]:
        """
        从文本中提取公式
        
        Args:
            text: 文本内容
            context: 上下文信息
            
        Returns:
            公式列表
        """
        formulas = []
        
        # 1. 提取LaTeX格式公式
        latex_formulas = self._extract_latex_formulas(text)
        formulas.extend(latex_formulas)
        
        # 2. 提取简单数学表达式
        simple_formulas = self._extract_simple_formulas(text)
        formulas.extend(simple_formulas)
        
        return formulas
    
    def _extract_latex_formulas(self, text: str) -> List[Dict[str, Any]]:
        """提取LaTeX格式的公式"""
        formulas = []
        
        # 行内公式 $...$
        inline_pattern = r'\$([^\$]+)\$'
        for match in re.finditer(inline_pattern, text):
            formulas.append({
                "type": "latex_inline",
                "content": match.group(1),
                "latex": match.group(1),
                "position": match.span()
            })
        
        # 块公式 $$...$$
        block_pattern = r'\$\$([^\$]+)\$\$'
        for match in re.finditer(block_pattern, text):
            formulas.append({
                "type": "latex_block",
                "content": match.group(1),
                "latex": match.group(1),
                "position": match.span()
            })
        
        # LaTeX环境
        env_pattern = r'\\begin\{(equation|align|gather)\}(.*?)\\end\{\1\}'
        for match in re.finditer(env_pattern, text, re.DOTALL):
            formulas.append({
                "type": f"latex_{match.group(1)}",
                "content": match.group(2),
                "latex": match.group(0),
                "position": match.span()
            })
        
        return formulas
    
    def _extract_simple_formulas(self, text: str) -> List[Dict[str, Any]]:
        """提取简单数学表达式"""
        formulas = []
        
        # 匹配简单方程式：a = b + c
        equation_pattern = r'\b([a-zA-Z_]\w*)\s*=\s*([\w\s\+\-\*/\(\)\^\.]+)'
        for match in re.finditer(equation_pattern, text):
            # 验证右侧是否包含数学运算符
            right_side = match.group(2)
            if any(op in right_side for op in ['+', '-', '*', '/', '^', '(', ')']):
                formulas.append({
                    "type": "equation",
                    "content": match.group(0),
                    "variable": match.group(1),
                    "expression": right_side.strip(),
                    "position": match.span()
                })
        
        return formulas
    
    def extract_formulas_from_image(self, image_path: str, context: str = "") -> Optional[Dict[str, Any]]:
        """
        从图像中提取公式
        
        Args:
            image_path: 图像路径
            context: 上下文信息
            
        Returns:
            提取的公式信息
        """
        result = {
            "image_path": image_path,
            "formulas": [],
            "method": "none",
            "success": False
        }
        
        # 1. 优先使用DeepSeek-VL识别公式
        if self.deepseek_wrapper:
            try:
                formula = self._recognize_with_deepseek(image_path, context)
                if formula:
                    result["formulas"].append(formula)
                    result["method"] = "deepseek"
                    result["success"] = True
                    return result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"DeepSeek公式识别失败: {e}")
        
        # 2. 备用OCR识别
        if self.ocr_engine:
            try:
                formula = self._recognize_with_ocr(image_path)
                if formula:
                    result["formulas"].append(formula)
                    result["method"] = "ocr"
                    result["success"] = True
                    return result
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"OCR公式识别失败: {e}")
        
        return result
    
    def _recognize_with_deepseek(self, image_path: str, context: str = "") -> Optional[Dict[str, Any]]:
        """使用DeepSeek-VL识别公式"""
        if not self.deepseek_wrapper:
            return None
        
        prompt = """请识别图片中的数学公式，并以LaTeX格式输出。
要求：
1. 公式部分用LaTeX格式表示
2. 说明公式中各符号的含义
3. 简要说明公式的物理或数学意义

格式：
LaTeX: [公式]
符号说明: [说明]
意义: [意义]
"""
        
        response = self.deepseek_wrapper.generate_description(image_path, prompt)
        
        # 解析响应
        formula_data = {
            "latex": "",
            "symbols": {},
            "meaning": "",
            "raw_response": response
        }
        
        # 提取LaTeX部分
        latex_match = re.search(r'LaTeX:\s*(.+?)(?:\n|$)', response)
        if latex_match:
            formula_data["latex"] = latex_match.group(1).strip()
        
        # 提取意义部分
        meaning_match = re.search(r'意义:\s*(.+?)(?:\n\n|$)', response, re.DOTALL)
        if meaning_match:
            formula_data["meaning"] = meaning_match.group(1).strip()
        
        return formula_data if formula_data["latex"] else None
    
    def _recognize_with_ocr(self, image_path: str) -> Optional[Dict[str, Any]]:
        """使用OCR识别公式"""
        if not self.ocr_engine:
            return None
        
        try:
            result = self.ocr_engine.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                return None
            
            # 提取所有文本
            texts = [line[1][0] for line in result[0]]
            full_text = ' '.join(texts)
            
            return {
                "content": full_text,
                "confidence": sum(line[1][1] for line in result[0]) / len(result[0]),
                "raw_ocr": result
            }
        except Exception as e:
            if self.logger:
                self.logger.error(f"OCR识别失败: {e}")
            return None
    
    def is_formula_image(self, image_path: str) -> bool:
        """
        判断图像是否可能包含公式
        
        Args:
            image_path: 图像路径
            
        Returns:
            是否为公式图像
        """
        try:
            img = Image.open(image_path)
            width, height = img.size
            
            # 简单启发式判断
            # 公式图像通常：宽度适中、高度较小、横纵比大
            aspect_ratio = width / height if height > 0 else 0
            
            is_formula = (
                100 < width < 1000 and  # 宽度适中
                20 < height < 200 and   # 高度较小
                aspect_ratio > 2        # 横向长条形
            )
            
            return is_formula
        except Exception:
            return False
