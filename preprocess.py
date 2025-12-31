"""
数据预处理入口脚本

使用方法:
    python preprocess.py              # 预处理所有数据集
    python preprocess.py twibot20     # 只预处理Twibot-20
    python preprocess.py misbot       # 只预处理Misbot
"""

import sys
from preprocess_unified import UnifiedPreprocessor

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    preprocessor = UnifiedPreprocessor()
    
    if dataset == "all":
        preprocessor.preprocess_all()
    else:
        preprocessor.preprocess(dataset)
