import ctypes
import threading
import time

from . import config

# XInput button flags
_XINPUT_BUTTONS = {
    0x0001: "DPAD_U",  0x0002: "DPAD_D",  0x0004: "DPAD_L",  0x0008: "DPAD_R",
    0x0010: "START",   0x0020: "BACK",
    0x0040: "LS",      0x0080: "RS",
    0x0100: "LB",      0x0200: "RB",
    0x1000: "A",       0x2000: "B",        0x4000: "X",       0x8000: "Y",
}

_BUTTON_NAMES = ("A", "B", "X", "Y", "LB", "RB", "BACK", "START", "LS", "RS", "GUIDE", "LT", "RT", "DPAD_U", "DPAD_D", "DPAD_L", "DPAD_R")
_AXIS_NAMES = ("LX", "LY", "RX", "RY", "DPAD_X", "DPAD_Y")

ERROR_DEVICE_NOT_CONNECTED = 1167


class _XinputGamepad(ctypes.Structure):
    _fields_ = [
        ("buttons", ctypes.c_ushort),
        ("left_trigger", ctypes.c_ubyte),
        ("right_trigger", ctypes.c_ubyte),
        ("l_thumb_x", ctypes.c_short),
        ("l_thumb_y", ctypes.c_short),
        ("r_thumb_x", ctypes.c_short),
        ("r_thumb_y", ctypes.c_short),
    ]


class _XinputState(ctypes.Structure):
    _fields_ = [
        ("packet_number", ctypes.c_ulong),
        ("gamepad", _XinputGamepad),
    ]


def _load_xinput():
    for name in ("XInput1_4.dll", "XInput9_1_0.dll", "XInput1_3.dll"):
        try:
            dll = ctypes.windll.LoadLibrary(name)
            dll.XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(_XinputState)]
            dll.XInputGetState.restype = ctypes.c_uint
            return dll
        except OSError:
            continue
    raise OSError("No XInput DLL found")


class InputReader:
    """后台线程持续读取手柄状态，维护当前状态快照。"""

    def __init__(self, gamepad_index: int = config.GAMEPAD_INDEX):
        self._gamepad_index = gamepad_index
        self._buttons = {k: 0 for k in _BUTTON_NAMES}
        self._axes = {k: 0.0 for k in _AXIS_NAMES}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._connected = False
        self._xinput = None

    def start(self) -> "InputReader":
        try:
            self._xinput = _load_xinput()
        except OSError:
            self._connected = False
            return self

        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _poll(self):
        last_packet = None
        while self._running:
            state = _XinputState()
            res = self._xinput.XInputGetState(self._gamepad_index, ctypes.byref(state))
            if res == ERROR_DEVICE_NOT_CONNECTED:
                self._connected = False
                last_packet = None
                time.sleep(1.0)
                continue
            if res != 0:
                time.sleep(0.5)
                continue

            self._connected = True
            if last_packet is not None and state.packet_number == last_packet:
                time.sleep(0.001)
                continue
            last_packet = state.packet_number

            g = state.gamepad
            self._update_from_gamepad(g)

    def _update_from_gamepad(self, g: _XinputGamepad):
        with self._lock:
            # buttons
            for flag, name in _XINPUT_BUTTONS.items():
                self._buttons[name] = 1 if (g.buttons & flag) else 0

            # triggers → LT/RT as buttons (二值化)
            lt_raw = g.left_trigger / 255.0
            rt_raw = g.right_trigger / 255.0
            self._buttons["LT"] = 1 if lt_raw >= 0.5 else 0
            self._buttons["RT"] = 1 if rt_raw >= 0.5 else 0

            # sticks
            def _stick(raw):
                val = raw / 32767.0
                if abs(val) < config.STICK_DEADZONE:
                    return 0.0
                return round(val, 4)

            self._axes["LX"] = _stick(g.l_thumb_x)
            self._axes["LY"] = _stick(g.l_thumb_y)
            self._axes["RX"] = _stick(g.r_thumb_x)
            self._axes["RY"] = _stick(g.r_thumb_y)

            # dpad → axes
            self._axes["DPAD_X"] = (
                -1.0 if (g.buttons & 0x0004) else
                1.0 if (g.buttons & 0x0008) else
                0.0
            )
            self._axes["DPAD_Y"] = (
                -1.0 if (g.buttons & 0x0002) else
                1.0 if (g.buttons & 0x0001) else
                0.0
            )

    def get_state(self) -> dict:
        with self._lock:
            return {
                "buttons": dict(self._buttons),
                "axes": dict(self._axes),
            }

    @property
    def connected(self) -> bool:
        return self._connected
