#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSeek-VL 模型封装类
用于图像理解、公式识别、代码识别等功能
"""

import sys
import os
import torch
from PIL import Image
from logger_config import get_logger

# Windows CPU兼容性设置（仅在CPU模式下使用）
# GPU模式下不需要这些设置
if not torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False

# 添加DeepSeek-VL代码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
deepseek_path = os.path.join(parent_dir, "DeepSeek-VL-main")
sys.path.insert(0, deepseek_path)

try:
    from transformers import AutoModelForCausalLM
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from deepseek_vl.utils.io import load_pil_images
except ImportError as e:
    print(f"警告: DeepSeek-VL导入失败: {e}")
    print("某些功能将不可用")


class DeepSeekVLWrapper:
    """DeepSeek-VL模型封装类"""
    
    def __init__(self, model_path="./models/deepseek-vl-7b-chat", device=None):
        """
        初始化DeepSeek-VL模型
        
        Args:
            model_path: 模型路径
            device: 设备 (cuda/cpu)，None则自动检测
        """
        self.logger = get_logger("DeepSeekVLWrapper")
        
        print("[DeepSeek-VL] 正在初始化DeepSeek-VL模型...")
        self.logger.info("开始初始化DeepSeek-VL模型")
        
        # 检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"   使用设备: {self.device}")
        self.logger.info(f"使用设备: {self.device}")
        
        # 检查模型路径
        if not os.path.exists(model_path):
            self.logger.error(f"模型路径不存在: {model_path}")
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        self.logger.info(f"模型路径: {model_path}")
        
        # 加载处理器
        print("   ⏳ [1/3] 正在加载 VLChatProcessor...")
        self.logger.info("加载VLChatProcessor")
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        print("   ✅ [1/3] VLChatProcessor 加载完成")
        self.logger.info("VLChatProcessor加载成功")
        
        # 加载模型 - 按照官方标准方式
        print("   ⏳ [2/3] 正在加载主模型 (13.7GB，需要1-3分钟)...")
        print("        └─ 正在从磁盘读取模型文件...")
        self.logger.info("开始加载主模型")
        
        # 标准加载
        if self.device == "cuda":
            self.logger.info("CUDA模式：加载为bfloat16")
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print("        └─ 模型文件加载完成，正在转换数据类型并移动到GPU...")
            self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()
            print("   ✅ [2/3] 主模型加载完成并已移至GPU")
        else:
            # CPU模式：强制float32（按照DeepSeek官方建议）
            print("        └─ CPU模式：强制加载为float32...")
            self.logger.info("CPU模式：强制加载为float32")
            self.vl_gpt = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32
            )
            print("        └─ 正在转换为float32格式...")
            # 二次确保所有参数为float32
            self.vl_gpt = self.vl_gpt.float()
            self.vl_gpt = self.vl_gpt.eval()
            print("   ✅ [2/3] 主模型加载完成（CPU float32模式）")
            self.logger.info("Float32强制转换完成")
        
        print("   ✅ [3/3] DeepSeek-VL 初始化完成！\n")
        self.logger.info("DeepSeek-VL模型加载完成")
    
    @torch.inference_mode()
    def _generate_response(self, conversation, max_new_tokens=512):
        """
        生成模型回复
        
        Args:
            conversation: 对话列表
            max_new_tokens: 最大生成token数
            
        Returns:
            str: 模型回复
        """
        try:
            # 加载图像
            pil_images = load_pil_images(conversation)
            
            # 准备输入（使用conversations参数）
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.vl_gpt.device)
            
            # 生成回复
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            outputs = self.vl_gpt.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
            
            answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            print(f"生成回复失败: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def describe_image(self, image_path, prompt=None):
        """
        生成图像描述
        
        Args:
            image_path: 图像路径
            prompt: 自定义提示词，None则使用默认
            
        Returns:
            str: 图像描述
        """
        self.logger.info(f"开始生成图像描述: {image_path}")
        if prompt is None:
            # 针对教育知识图谱优化的提示词
            prompt = (
                "请以教授《机电系统原理》课程的专业大学教师的身份，从构建教育知识图谱的视角分析这张图片，提供以下信息：\n"
                "1. 核心概念：识别图中的主要概念、术语、定义，对图片中的内容做一个整体的描述\n"
                "2. 知识类型：判断内容类型（原理图、流程图、示意图、数据图表、实验演示、实物照片等）\n"
                "3. 知识点：提取关键知识点和教学内容\n"
                "请用结构化、专业性的方式描述，便于知识提取和图谱构建。"
            )
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path]
            },
            {"role": "Assistant", "content": ""}
        ]
        
        result = self._generate_response(conversation, max_new_tokens=500)
        self.logger.info(f"图像描述生成完成: {image_path}")
        return result
    
    def recognize_formula(self, image_path):
        """
        识别图像中的数学公式并转换为LaTeX
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: {"latex": LaTeX格式公式, "description": 公式描述}
        """
        self.logger.info(f"开始识别公式: {image_path}")
        # 针对教育知识图谱优化的公式识别提示词
        prompt = (
            "这张图片中包含《机电系统原理》课程相关的数学/物理/工程公式。请进行知识提取：\n"
            "1. LaTeX表达：将公式精确转换为LaTeX格式\n"
            "2. 公式名称：如果是已知公式，给出标准名称（如：牛顿第二定律、欧拉公式等）\n"
            "3. 物理含义：解释公式的物理/数学意义\n"
            "4. 变量说明：列出公式中各变量的含义和单位\n"
            "5. 适用条件：说明公式的适用范围或前提条件\n"
            "请按照以下格式回复：\n"
            "LaTeX: [LaTeX代码]\n"
            "名称: [公式名称]\n"
            "含义: [物理/数学含义]\n"
            "变量: [各变量说明]\n"
            "条件: [适用条件]\n"
        )
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path]
            },
            {"role": "Assistant", "content": ""}
        ]
        
        response = self._generate_response(conversation, max_new_tokens=400)
        self.logger.info(f"公式识别完成: {image_path}")
        
        # 解析回复
        result = {
            "latex": "",
            "description": "",
            "raw_response": response
        }
        
        # 尝试提取LaTeX
        if "LaTeX:" in response or "latex:" in response:
            lines = response.split('\n')
            for line in lines:
                if 'latex:' in line.lower():
                    result["latex"] = line.split(':', 1)[1].strip()
                elif '含义:' in line or '意义:' in line or '说明:' in line:
                    result["description"] = line.split(':', 1)[1].strip()
        else:
            # 如果没有按格式返回，整个作为LaTeX
            result["latex"] = response
        
        return result
    
    def recognize_code(self, image_path):
        """
        识别图像中的代码
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: {"code": 代码内容, "language": 编程语言, "description": 代码说明}
        """
        self.logger.info(f"开始识别代码: {image_path}")
        # 针对教育知识图谱优化的代码识别提示词
        prompt = (
            "这张图片中包含《机电系统原理》课程相关程序代码。请进行教学知识提取：\n"
            "1. 代码重现：精确提取代码内容（保持缩进和格式）\n"
            "2. 编程语言：识别编程语言和版本特征\n"
            "3. 功能说明：描述代码的功能和应用场景\n"
            "4. 知识点：列出代码涉及的教学知识点\n"
            "请按照以下格式回复：\n"
            "语言: [编程语言]\n"
            "代码:\n```\n[代码内容]\n```\n"
            "功能: [功能说明]\n"
            "知识点: [教学知识点]\n"
        )
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path]
            },
            {"role": "Assistant", "content": ""}
        ]
        
        response = self._generate_response(conversation, max_new_tokens=500)
        self.logger.info(f"代码识别完成: {image_path}")
        
        # 解析回复
        result = {
            "code": "",
            "language": "unknown",
            "description": "",
            "raw_response": response
        }
        
        # 提取语言
        if "语言:" in response:
            lines = response.split('\n')
            for line in lines:
                if '语言:' in line:
                    result["language"] = line.split(':', 1)[1].strip()
                    break
        
        # 提取代码块
        import re
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            result["code"] = code_blocks[0][1].strip()
            if code_blocks[0][0]:  # 如果有语言标识
                result["language"] = code_blocks[0][0]
        else:
            # 如果没有代码块标记，尝试提取代码部分
            if "代码:" in response:
                parts = response.split("代码:", 1)
                if len(parts) > 1:
                    code_part = parts[1].split("功能:")[0] if "功能:" in parts[1] else parts[1]
                    result["code"] = code_part.strip()
        
        # 提取功能说明
        if "功能:" in response:
            result["description"] = response.split("功能:", 1)[1].strip()
        
        return result
    
    def analyze_image_comprehensive(self, image_path):
        """
        全面分析图像内容（描述+公式+代码）
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: 包含各类分析结果
        """
        prompt = (
            "请全面分析这张图片，包括：\n"
            "1. 图片的整体描述（场景、内容等）\n"
            "2. 如果包含数学公式，请转换为LaTeX格式\n"
            "3. 如果包含代码，请提取代码内容和编程语言\n"
            "4. 图片的主题和类型"
        )
        
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt}",
                "images": [image_path]
            },
            {"role": "Assistant", "content": ""}
        ]
        
        response = self._generate_response(conversation, max_new_tokens=600)
        
        return {
            "full_analysis": response,
            "image_path": image_path
        }
    
    def batch_process_images(self, image_paths, task="describe"):
        """
        批量处理图像
        
        Args:
            image_paths: 图像路径列表
            task: 任务类型 (describe/formula/code/comprehensive)
            
        Returns:
            list: 结果列表
        """
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"处理图像 {i}/{len(image_paths)}: {image_path}")
            
            try:
                if task == "describe":
                    result = self.describe_image(image_path)
                elif task == "formula":
                    result = self.recognize_formula(image_path)
                elif task == "code":
                    result = self.recognize_code(image_path)
                elif task == "comprehensive":
                    result = self.analyze_image_comprehensive(image_path)
                else:
                    result = {"error": f"未知任务类型: {task}"}
                
                results.append({
                    "image_path": image_path,
                    "result": result,
                    "task": task
                })
                
            except Exception as e:
                print(f"处理失败: {e}")
                results.append({
                    "image_path": image_path,
                    "error": str(e),
                    "task": task
                })
        
        return results
