import os
import sys

# 获取当前脚本所在目录的父目录（项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 移除可能冲突的路径
sys.path = [p for p in sys.path if not p.endswith('GPT-SoVITS')]

# 添加当前项目路径
sys.path.insert(0, project_root)

# 导入并运行 API
from training_api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 