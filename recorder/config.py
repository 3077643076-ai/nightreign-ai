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
GAMEPAD_INDEX = 3  # Xbox 手柄被 Steam 挤到索引 3

# 截图参数
IMAGE_FORMAT = "jpg"          # jpg / png
JPEG_QUALITY = 85             # JPEG 质量 0-100
DXCAM_MAX_BUFFER_LEN = 64
DXCAM_OUTPUT_COLOR = "BGR"    # BGR for OpenCV compatibility

# 状态机 / 规划脑按键
SUPERVISOR_DRY_RUN = False
CONFIRM_KEY = "f"              # 死亡后再战 / 菜单确认
LOCK_ON_KEY = "q"              # 锁定目标，按个人键位修改
SUPERVISOR_INTERVAL_SEC = 0.5
RETRY_CONFIRM_DELAY_SEC = 0.8
