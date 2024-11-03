import os
from pathlib import Path

# 모델 경로 생성
model_path = os.path.join(Path(__file__).absolute().parents[2], 'ckpt/humanparsing/parsing_atr.onnx')

# 경로 출력
print(f"Model path: {model_path}")
