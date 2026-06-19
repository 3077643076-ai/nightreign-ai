"""零样本敌人检测：用 OWLv2 模型在游戏画面中找"怪物/敌人"。

无需标注数据，只需文本提示词即可检测。
首次运行会下载模型（~1.5GB）。

用法：
    python enemy_detector.py
    进游戏，看终端输出检测到的敌人数量和位置
"""

# 必须在 import transformers 之前设置镜像
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import time
import ctypes
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mss

# 检测提示词：要找的东西
QUERIES = ["enemy monster", "humanoid enemy", "giant creature", "boss"]

# 检测间隔（帧），不用每帧跑，太慢
DETECT_EVERY = 15  # ~1 秒检测一次

# 置信度阈值
CONFIDENCE_THRESHOLD = 0.15


def load_model():
    """加载 OWLv2 模型。"""
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    model_id = "google/owlv2-base-patch16-ensemble"
    print(f"加载 OWLv2 模型 (HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'default')})...")
    model = Owlv2ForObjectDetection.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to("cuda")
    processor = Owlv2Processor.from_pretrained(model_id)
    model.eval()
    print("模型就绪！")
    return model, processor


def detect(model, processor, frame, queries):
    """在画面中检测指定目标，返回 [(box, label, confidence), ...]"""
    h, w = frame.shape[:2]
    # BGR → RGB PIL
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(text=queries, images=pil_img, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=torch.tensor([(h, w)]), threshold=CONFIDENCE_THRESHOLD
    )[0]

    detections = []
    for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append(((x1, y1, x2, y2), queries[label], float(score)))

    return detections


def main():
    # 加载模型
    model, processor = load_model()

    sct = mss.MSS()
    monitor = sct.monitors[1]

    print(f"\n检测提示词: {QUERIES}")
    print(f"检测间隔: 每 {DETECT_EVERY} 帧 (~1秒)")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}")
    print("\n进游戏！终端会打印检测结果\n")

    cnt = 0
    while True:
        img = np.array(sct.grab(monitor))
        frame = img[:, :, :3].copy()

        cnt += 1
        if cnt % DETECT_EVERY != 0:
            continue

        # 降采样加速检测（分辨率减半）
        small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

        dets = detect(model, processor, small, QUERIES)
        if dets:
            print(f"\n[#{cnt}] 检测到 {len(dets)} 个目标:")
            for box, label, conf in dets[:5]:  # 最多显示 5 个
                x1, y1, x2, y2 = [v * 2 for v in box]  # 还原坐标
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = frame.shape[1]
                # 判断在屏幕的哪边
                if cx < w // 3:
                    side = "左"
                elif cx > 2 * w // 3:
                    side = "右"
                else:
                    side = "中"
                print(f"  {label}: conf={conf:.2f} 位置={side}({cx},{cy}) 大小={x2-x1}x{y2-y1}")
        else:
            print(f"[#{cnt}] 未检测到敌人", flush=True)

        if ctypes.windll.user32.GetAsyncKeyState(ord('Q')) & 0x8000:
            break

    print("退出")


if __name__ == "__main__":
    main()
