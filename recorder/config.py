from pathlib import Path

# 录制参数
FPS = 15                      # 行为克隆 15fps 够用
TARGET_WIDTH = 1280           # 720p 降质，None = 保持原始分辨率
TARGET_HEIGHT = 720

# 存储路径
DATA_ROOT = Path(__file__).parent.parent / "data"

# 手柄 Deadzone
STICK_DEADZONE = 0.15         # 摇杆死区
TRIGGER_DEADZONE = 0.05       # 扳机死区

# 热键
HOTKEY_START = "f8"
HOTKEY_STOP = "f9"

# 手柄索引 (第一个手柄 = 0)
GAMEPAD_INDEX = 0

# 截图参数
IMAGE_FORMAT = "jpg"          # jpg / png
JPEG_QUALITY = 85             # JPEG 质量 0-100
DXCAM_MAX_BUFFER_LEN = 64
DXCAM_OUTPUT_COLOR = "BGR"    # BGR for OpenCV compatibility
