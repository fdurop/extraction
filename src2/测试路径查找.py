"""
测试路径查找功能
确保能够正确找到项目根目录、DeepSeek-VL和模型
"""

import os
import sys

def find_project_root(start_path):
    """
    从当前目录向上查找项目根目录
    项目根目录标志：包含DeepSeek-VL-main目录
    """
    current = os.path.abspath(start_path)
    max_depth = 5
    
    print(f"开始查找项目根目录，起点: {current}")
    
    for level in range(max_depth):
        print(f"  [{level}] 检查: {current}")
        
        # 检查是否包含DeepSeek-VL-main
        deepseek_path = os.path.join(current, 'DeepSeek-VL-main')
        if os.path.exists(deepseek_path):
            print(f"  ✓ 找到DeepSeek-VL-main: {deepseek_path}")
            return current
        
        # 向上一级
        parent = os.path.dirname(current)
        if parent == current:  # 已到根目录
            print(f"  × 已到系统根目录")
            break
        current = parent
    
    print(f"  × 未找到项目根目录（查找了{max_depth}层）")
    return None

def main():
    print("="*60)
    print("路径查找测试")
    print("="*60)
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\n当前脚本位置: {current_dir}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 查找项目根目录
    print("\n[1/4] 查找项目根目录...")
    project_root = find_project_root(current_dir)
    
    if project_root is None:
        print("\n❌ 测试失败：无法找到项目根目录")
        print("\n请确保目录结构正确：")
        print("  your-project/")
        print("  ├── DeepSeek-VL-main/   ← 必须存在")
        print("  └── src2/")
        return False
    
    print(f"\n✓ 项目根目录: {project_root}")
    
    # 检查DeepSeek-VL
    print("\n[2/4] 检查DeepSeek-VL目录...")
    deepseek_path = os.path.join(project_root, 'DeepSeek-VL-main')
    if os.path.exists(deepseek_path):
        print(f"✓ DeepSeek-VL存在: {deepseek_path}")
        
        # 检查关键文件
        init_file = os.path.join(deepseek_path, 'deepseek_vl', '__init__.py')
        if os.path.exists(init_file):
            print(f"  ✓ deepseek_vl模块存在")
        else:
            print(f"  ⚠️ deepseek_vl模块不完整")
    else:
        print(f"❌ DeepSeek-VL不存在: {deepseek_path}")
        return False
    
    # 检查模型目录
    print("\n[3/4] 检查模型目录...")
    model_path = os.path.join(project_root, 'models', 'deepseek-vl-7b-chat')
    if os.path.exists(model_path):
        print(f"✓ 模型目录存在: {model_path}")
        
        # 检查模型文件
        config_file = os.path.join(model_path, 'config.json')
        if os.path.exists(config_file):
            print(f"  ✓ config.json存在")
        else:
            print(f"  ⚠️ config.json不存在")
    else:
        print(f"⚠️ 模型目录不存在: {model_path}")
        print(f"   （DeepSeek-VL功能将不可用）")
    
    # 检查input目录
    print("\n[4/4] 检查input目录...")
    input_path = os.path.join(project_root, 'input')
    if os.path.exists(input_path):
        print(f"✓ input目录存在: {input_path}")
        files = [f for f in os.listdir(input_path) 
                if f.lower().endswith(('.pdf', '.pptx')) and not f.startswith('~$')]
        print(f"  找到 {len(files)} 个待处理文件")
        for f in files[:5]:  # 最多显示5个
            print(f"    - {f}")
        if len(files) > 5:
            print(f"    ... 还有 {len(files)-5} 个文件")
    else:
        print(f"⚠️ input目录不存在: {input_path}")
        print(f"   请创建input目录并放入待处理文件")
    
    # 测试导入
    print("\n[额外] 测试模块导入...")
    sys.path.insert(0, current_dir)
    
    try:
        from core import ImageProcessor, TextProcessor
        print("✓ core模块导入成功")
    except Exception as e:
        print(f"❌ core模块导入失败: {e}")
        return False
    
    try:
        from parsers import PDFParser, PPTXParser
        print("✓ parsers模块导入成功")
    except Exception as e:
        print(f"❌ parsers模块导入失败: {e}")
        return False
    
    try:
        from utils import OutputManager, FileUtils
        print("✓ utils模块导入成功")
    except Exception as e:
        print(f"❌ utils模块导入失败: {e}")
        return False
    
    # 总结
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"✓ 项目根目录: {project_root}")
    print(f"✓ 所有关键路径已验证")
    print(f"✓ 所有模块可以正常导入")
    print("\n✅ 路径配置正确！可以运行 multimodal_preprocessor.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
