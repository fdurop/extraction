"""
DeepSeek-VL模型封装器
从原src目录复制并适配
"""

import sys
import os

def find_project_root(start_path):
    """
    从当前目录向上查找项目根目录
    项目根目录标志：包含DeepSeek-VL-main目录
    """
    current = os.path.abspath(start_path)
    max_depth = 5  # 最多向上查找5层
    
    for _ in range(max_depth):
        # 检查是否包含DeepSeek-VL-main
        deepseek_path = os.path.join(current, 'DeepSeek-VL-main')
        if os.path.exists(deepseek_path):
            return current
        
        # 向上一级
        parent = os.path.dirname(current)
        if parent == current:  # 已到根目录
            break
        current = parent
    
    return None

# 查找项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = find_project_root(current_dir)

if project_root is None:
    print(f"[ERROR] 无法找到项目根目录（需要包含DeepSeek-VL-main目录）")
    print(f"[INFO] 从 {current_dir} 向上查找")
    DEEPSEEK_AVAILABLE = False
else:
    deepseek_path = os.path.join(project_root, 'DeepSeek-VL-main')
    sys.path.insert(0, deepseek_path)

try:
    from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
    from deepseek_vl.utils.io import load_pil_images
    import torch
    DEEPSEEK_AVAILABLE = True
except Exception as e:
    print(f"DeepSeek-VL导入失败: {e}")
    DEEPSEEK_AVAILABLE = False


class DeepSeekVLWrapper:
    """DeepSeek-VL模型封装"""
    
    def __init__(self, model_path: str = None):
        """
        初始化DeepSeek-VL
        
        Args:
            model_path: 模型路径
        """
        if not DEEPSEEK_AVAILABLE:
            raise RuntimeError("DeepSeek-VL不可用")
        
        if model_path is None:
            # 使用相对路径查找模型
            current_dir = os.path.dirname(os.path.abspath(__file__))
            proj_root = find_project_root(current_dir)
            
            if proj_root is None:
                raise RuntimeError("无法找到项目根目录")
            
            model_path = os.path.join(proj_root, "models", "deepseek-vl-7b-chat")
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"模型目录不存在: {model_path}")
        
        print(f"加载DeepSeek-VL模型: {model_path}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.vl_gpt = MultiModalityCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True
        ).to(torch.bfloat16).to(self.device).eval()
        
        print(f"DeepSeek-VL模型加载完成 (设备: {self.device})")
    
    def _generate_response(self, conversation, max_new_tokens=512):
        """
        通用的生成方法（内部方法）
        
        Args:
            conversation: 对话列表
            max_new_tokens: 最大生成token数
            
        Returns:
            生成的回复
        """
        try:
            # 加载图像
            pil_images = load_pil_images(conversation)
            
            # 准备输入
            prepare_inputs = self.vl_chat_processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.device)
            
            # 生成
            inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            
            with torch.inference_mode():
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
            print(f"DeepSeek-VL生成失败: {e}")
            return ""
    
    def generate_description(self, image_path: str, prompt: str = None) -> str:
        """
        生成图像描述
        
        Args:
            image_path: 图像路径
            prompt: 提示词，None则使用默认
            
        Returns:
            生成的描述
        """
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
        
        return self._generate_response(conversation, max_new_tokens=500)
    
    def recognize_formula(self, image_path: str):
        """
        识别图像中的数学公式并转换为LaTeX
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: {"latex": LaTeX格式公式, "description": 公式描述, "raw_response": 完整回复}
        """
        print(f"      [公式识别] 开始识别公式...")
        
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
        
        print(f"      [OK] 公式识别完成: {result['latex'][:50]}...")
        return result
    
    def recognize_code(self, image_path: str):
        """
        识别图像中的代码
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: {"code": 代码内容, "language": 编程语言, "description": 代码说明, "has_code": 是否包含代码}
        """
        print(f"      [代码识别] 开始识别代码...")
        
        # 先判断是否有代码
        prompt = (
            "请仔细观察这张图片，判断是否包含程序代码。\n"
            "如果包含代码，请提取：\n"
            "1. 代码内容：精确提取（保持原始格式和缩进）\n"
            "2. 编程语言：识别编程语言\n"
            "3. 功能说明：简要描述代码功能\n"
            "\n"
            "如果不包含代码或只是文字说明，请直接回复：无代码\n"
            "\n"
            "包含代码时请按以下格式回复：\n"
            "语言: [编程语言]\n"
            "```\n"
            "[代码内容]\n"
            "```\n"
            "功能: [功能说明]\n"
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
        
        # 解析回复
        result = {
            "code": "",
            "language": "txt",
            "description": "",
            "raw_response": response,
            "has_code": False
        }
        
        # 检查是否明确说明没有代码
        no_code_indicators = ['无代码', '没有代码', 'no code', '不包含代码', '无程序', 'not contain']
        if any(indicator in response.lower() for indicator in no_code_indicators):
            print(f"      [跳过] 图片中无代码内容")
            return result
        
        # 提取语言
        if "语言:" in response:
            lines = response.split('\n')
            for line in lines:
                if '语言:' in line:
                    result["language"] = line.split(':', 1)[1].strip()
                    break
        
        # 提取代码块（关键修复：遍历所有代码块，跳过标记块）
        import re
        code_blocks = re.findall(r'```(\w*)\n(.*?)```', response, re.DOTALL)
        
        if code_blocks:
            # 遍历所有代码块，找到真正的代码（而不是"语言: Arduino"这样的标记）
            for lang, code_content in code_blocks:
                code_content = code_content.strip()
                
                # 跳过太短的内容
                if len(code_content) < 5:
                    continue
                
                # 跳过只包含标记的代码块（如"语言: Arduino"）
                if code_content.startswith(('语言:', '功能:', '说明:', '[代码内容]')):
                    continue
                
                # 跳过纯标记行（只有一行且包含冒号）
                lines = code_content.split('\n')
                non_empty_lines = [line for line in lines if line.strip()]
                if len(non_empty_lines) == 1 and ':' in non_empty_lines[0]:
                    continue
                
                # 找到有效的代码块
                result["code"] = code_content
                result["has_code"] = True
                if lang:  # 如果有语言标识
                    result["language"] = lang
                print(f"      → 从```代码块提取成功 ({len(code_content)} 字符)")
                break  # 找到第一个有效代码块就停止
        else:
            # 如果没有代码块标记，尝试提取代码部分
            if "代码:" in response:
                parts = response.split("代码:", 1)
                if len(parts) > 1:
                    code_part = parts[1].split("功能:")[0] if "功能:" in parts[1] else parts[1]
                    code_content = code_part.strip()
                    if code_content and len(code_content) > 10:
                        result["code"] = code_content
                        result["has_code"] = True
        
        # 提取功能说明
        if "功能:" in response:
            result["description"] = response.split("功能:", 1)[1].strip()
        
        # 最终验证
        if not result["has_code"] or not result["code"] or len(result["code"]) < 10:
            result["has_code"] = False
            result["code"] = ""
            print(f"      [跳过] 未提取到有效代码内容")
            return result
        
        print(f"      [OK] 代码识别完成: {result['language']}, {len(result['code'])} 字符")
        return result
    
    def recognize_table(self, image_path: str):
        """
        识别图像中的表格
        
        Args:
            image_path: 图像路径
            
        Returns:
            dict: {"table_data": 表格内容, "description": 表格说明}
        """
        print(f"      [表格识别] 开始识别表格...")
        
        prompt = (
            "这张图片中包含表格。请提取表格内容：\n"
            "1. 表格结构：识别行数、列数、表头\n"
            "2. 表格内容：按行列提取所有单元格的内容\n"
            "3. 表格说明：描述表格的主题和用途\n"
            "请按照以下格式回复：\n"
            "表头: [列1, 列2, 列3, ...]\n"
            "数据:\n"
            "行1: [值1, 值2, 值3, ...]\n"
            "行2: [值1, 值2, 值3, ...]\n"
            "说明: [表格主题和用途]\n"
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
        
        # 解析回复
        result = {
            "table_data": [],
            "headers": [],
            "description": "",
            "raw_response": response
        }
        
        # 简单解析（实际应用中可能需要更复杂的解析）
        lines = response.split('\n')
        for line in lines:
            if '表头:' in line or 'Headers:' in line or '列:' in line:
                # 尝试提取表头
                header_text = line.split(':', 1)[1].strip()
                result["headers"] = header_text
            elif '说明:' in line or 'Description:' in line:
                result["description"] = line.split(':', 1)[1].strip()
        
        # 将整个回复作为表格数据
        result["table_data"] = response
        
        print(f"      [OK] 表格识别完成")
        return result
