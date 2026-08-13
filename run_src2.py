"""API 多模态抽取流程入口。"""

import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
print(f"项目根目录: {project_root}")

# 添加必要的路径
src2_path = os.path.join(project_root, 'src2')
sys.path.insert(0, src2_path)
sys.path.insert(0, project_root)

# 验证关键路径
print("\n检查关键路径:")
print(f"  src2目录: {os.path.exists(src2_path)} - {src2_path}")

input_path = os.path.join(project_root, 'input')
print(f"  输入目录: {os.path.exists(input_path)} - {input_path}")
config_path = os.path.join(project_root, 'config', 'vlm_api.yaml')
print(f"  API配置: {os.path.exists(config_path)} - {config_path}")

os.chdir(project_root)
print(f"\n当前工作目录: {os.getcwd()}")

# 导入并运行主程序
print("\n" + "="*60)
print("开始运行 API 多模态抽取流程")
print("="*60 + "\n")

try:
    from multimodal_preprocessor import main
    main()
except Exception as e:
    print(f"\n[ERROR] 运行失败: {e}")
    import traceback
    traceback.print_exc()
