"""
图像处理模块
功能：图像增强、描述生成（CLIP、DeepSeek-VL）
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageEnhance
from typing import Optional, Dict, Any


def _relative_path(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        return os.path.relpath(path, os.getcwd()).replace(os.sep, "/")
    except Exception:
        return str(path).replace("\\", "/")


class ImageProcessor:
    def __init__(self, use_deepseek=True, use_clip=False, device="cuda", logger=None, deepseek_wrapper=None, vlm_client=None):
        """
        初始化图像处理器
        
        Args:
            use_deepseek: 是否使用DeepSeek-VL模型
            use_clip: 是否使用CLIP模型
            device: 计算设备
            logger: 日志记录器
            deepseek_wrapper: 外部已加载的DeepSeek-VL实例（避免重复加载）
        """
        self.use_deepseek = use_deepseek
        self.use_clip = use_clip
        self.device = device
        self.logger = logger
        
        # 延迟加载模型
        self.deepseek_wrapper = deepseek_wrapper  # 优先使用传入的实例
        self.vlm_client = vlm_client
        self.clip_model = None
        self.clip_processor = None
        
        # 只有在没有传入实例且需要使用时才加载
        if use_deepseek and self.deepseek_wrapper is None:
            self._init_deepseek()
        
        if use_clip:
            self._init_clip()
    
    def _init_deepseek(self):
        """初始化DeepSeek-VL模型"""
        try:
            from utils.deepseek_vl_wrapper import DeepSeekVLWrapper
            self.deepseek_wrapper = DeepSeekVLWrapper()
            if self.logger:
                self.logger.info("DeepSeek-VL模型初始化成功")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"DeepSeek-VL模型初始化失败: {e}")
            self.use_deepseek = False
    
    def _init_clip(self):
        """初始化CLIP模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if self.logger:
                self.logger.info("CLIP模型初始化成功")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"CLIP模型初始化失败: {e}")
            self.use_clip = False
    
    def enhance_image(self, image_path: str) -> str:
        """
        增强图像质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            增强后的图像路径
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                if self.logger:
                    self.logger.warning(f"图像文件不存在: {image_path}")
                return image_path
            
            # 检查文件扩展名
            file_ext = os.path.splitext(image_path)[1].lower()
            if file_ext in ['.svg', '.emf', '.wmf']:
                # 矢量图无法用PIL增强，跳过
                if self.logger:
                    self.logger.info(f"跳过矢量图增强: {image_path}")
                return image_path
            
            img = Image.open(image_path)
            
            # 处理调色板模式（Palette）图像的透明度问题
            if img.mode == 'P':
                # 如果有透明度信息，转换为RGBA
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            
            # 转换为RGB模式（统一处理）
            if img.mode == 'RGBA':
                # RGBA需要处理透明度，创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])  # 使用alpha通道作为mask
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 增强对比度
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            
            # 增强锐度
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.3)
            
            # Only change the file name suffix. Replacing every "." also changes
            # model directories such as qwen3.7-plus.
            root, ext = os.path.splitext(image_path)
            enhanced_path = f"{root}_enhanced{ext}"
            img.save(enhanced_path)
            
            return enhanced_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"图像增强失败: {e}")
            return image_path
    
    def generate_description(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """
        生成图像描述（自动选择最佳方法）
        
        Args:
            image_path: 图像路径
            context: 上下文文本
            
        Returns:
            包含描述信息的字典
        """
        result = {
            "image_path": image_path,
            "description": "",
            "method": "none",
            "enhanced": False,
            "backend": "none",
            "provider": None,
            "model": None,
            "status": "not_started",
            "api_error": None,
        }
        
        # Prefer the configured remote VLM API.
        if self.vlm_client:
            try:
                description = self.vlm_client.generate_description(image_path, context)
                result["description"] = description
                result["method"] = getattr(self.vlm_client, "provider_name", "vlm_api")
                result["backend"] = "api"
                result["provider"] = getattr(self.vlm_client, "provider", None)
                result["model"] = getattr(self.vlm_client, "model", None)
                result["status"] = "success"
                return result
            except Exception as e:
                result["api_error"] = str(e)
                result["status"] = "api_failed"
                if self.logger:
                    self.logger.warning(f"VLM API描述生成失败，尝试本地/备用模型: {e}")

        # 优先使用DeepSeek-VL
        if self.use_deepseek and self.deepseek_wrapper:
            try:
                description = self._generate_with_deepseek(image_path, context)
                result["description"] = description
                result["method"] = getattr(self.deepseek_wrapper, "provider_name", "deepseek")
                result["backend"] = "local"
                result["provider"] = "deepseek"
                result["model"] = getattr(self.deepseek_wrapper, "model_name", None) or "deepseek-vl"
                result["status"] = "success"
                return result
            except Exception as e:
                result["status"] = "local_failed"
                if self.logger:
                    self.logger.warning(f"DeepSeek-VL描述生成失败，尝试CLIP: {e}")
        
        # 备用CLIP
        if self.use_clip and self.clip_model:
            try:
                description = self._generate_with_clip(image_path)
                result["description"] = description
                result["method"] = "clip"
                result["backend"] = "local"
                result["provider"] = "clip"
                result["model"] = "openai/clip-vit-base-patch32"
                result["status"] = "success"
                return result
            except Exception as e:
                result["status"] = "clip_failed"
                if self.logger:
                    self.logger.error(f"CLIP描述生成失败: {e}")
        
        if result["status"] in {"not_started", "api_failed"}:
            result["status"] = "no_model_succeeded"
        return result
    
    def _generate_with_deepseek(self, image_path: str, context: str = "") -> str:
        """使用DeepSeek-VL生成描述"""
        if not self.deepseek_wrapper:
            raise ValueError("DeepSeek-VL未初始化")
        
        # 构建提示词
        if context:
            prompt = f"这是一个教学材料中的图片。上下文：{context[:200]}。请详细描述这张图片的内容，包括核心概念、关键信息和教学价值。"
        else:
            prompt = "请详细描述这张图片的内容，包括核心概念、关键信息和可能的应用场景。"
        
        description = self.deepseek_wrapper.generate_description(image_path, prompt)
        return description
    
    def _generate_with_clip(self, image_path: str) -> str:
        """使用CLIP生成描述"""
        if not self.clip_model or not self.clip_processor:
            raise ValueError("CLIP未初始化")
        
        image = Image.open(image_path).convert('RGB')
        
        # 候选描述
        candidates = [
            "一张包含技术图表的图片",
            "一张包含数学公式的图片",
            "一张包含电路图的图片",
            "一张包含流程图的图片",
            "一张包含数据表格的图片",
            "一张包含代码示例的图片",
            "一张教学说明图片"
        ]
        
        inputs = self.clip_processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # 获取最可能的描述
        best_idx = probs.argmax().item()
        confidence = probs[0][best_idx].item()
        
        return f"{candidates[best_idx]} (置信度: {confidence:.2f})"
    
    def process_image(self, image_path: str, page_text: str = "") -> Dict[str, Any]:
        """
        完整的图像处理流程
        
        Args:
            image_path: 图像路径
            page_text: 页面文本（提供上下文）
            
        Returns:
            处理结果字典
        """
        result = {
            "original_path": _relative_path(image_path),
            "enhanced_path": None,
            "description": None,
            "metadata": {},
            "error": None,
            "method": "none",
            "backend": "none",
            "provider": None,
            "model": None,
            "status": "not_started",
            "api_error": None,
            "indexable": False,
        }
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            error_msg = f"图像文件不存在: {image_path}"
            if self.logger:
                self.logger.warning(error_msg)
            result["error"] = error_msg
            return result
        
        # 检查文件格式，跳过无法处理的格式
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext in ['.svg', '.emf', '.wmf']:
            error_msg = f"跳过矢量图格式: {image_path}"
            if self.logger:
                self.logger.info(error_msg)
            result["error"] = error_msg
            return result
        
        # 1. 图像增强
        try:
            enhanced_path = self.enhance_image(image_path)
            result["enhanced_path"] = _relative_path(enhanced_path)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"图像增强失败: {e}")
            result["enhanced_path"] = _relative_path(image_path)
        
        # 2. 生成描述
        try:
            desc_image_path = enhanced_path if "enhanced_path" in locals() else image_path
            desc_result = self.generate_description(desc_image_path, page_text)
            # desc_result 是字典 {"description": "...", "method": "..."}
            result["description"] = desc_result.get("description", "")
            result["method"] = desc_result.get("method", "unknown")
            result["backend"] = desc_result.get("backend", "unknown")
            result["provider"] = desc_result.get("provider")
            result["model"] = desc_result.get("model")
            result["status"] = desc_result.get("status", "unknown")
            result["api_error"] = desc_result.get("api_error")
            result["indexable"] = bool(result["description"])
            result["metadata"]["generation"] = {
                "method": result["method"],
                "backend": result["backend"],
                "provider": result["provider"],
                "model": result["model"],
                "status": result["status"],
                "api_error": result["api_error"],
            }
        except Exception as e:
            error_msg = f"图像描述生成失败: {e}"
            if self.logger:
                self.logger.error(error_msg)
            result["error"] = error_msg
            result["status"] = "failed"
        
        return result
