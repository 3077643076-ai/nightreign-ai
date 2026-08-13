import threading
from collections import defaultdict

from pynput import keyboard, mouse


class KeyboardMouseReader:
    def __init__(self, start_listeners: bool = True):
        self._start_listeners = start_listeners
        self._keys = defaultdict(int)
        self._mouse_buttons = defaultdict(int)
        self._mouse_delta = {"dx": 0, "dy": 0, "wheel": 0}
        self._last_mouse_pos = None
        self._lock = threading.Lock()
        self._keyboard_listener = None
        self._mouse_listener = None

    def start(self) -> "KeyboardMouseReader":
        if not self._start_listeners:
            return self
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._mouse_listener = mouse.Listener(
            on_move=self._on_move,
            on_click=self._on_click,
            on_scroll=self._on_scroll,
        )
        self._keyboard_listener.daemon = True
        self._mouse_listener.daemon = True
        self._keyboard_listener.start()
        self._mouse_listener.start()
        return self

    def stop(self):
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None

    def record_key_down(self, name: str):
        with self._lock:
            self._keys[self._normalize_key_name(name)] = 1

    def record_key_up(self, name: str):
        with self._lock:
            self._keys[self._normalize_key_name(name)] = 0

    def record_mouse_move(self, dx: int, dy: int):
        with self._lock:
            self._mouse_delta["dx"] += int(dx)
            self._mouse_delta["dy"] += int(dy)

    def record_mouse_wheel(self, dy: int):
        with self._lock:
            self._mouse_delta["wheel"] += int(dy)

    def record_mouse_button(self, name: str, pressed: bool):
        with self._lock:
            self._mouse_buttons[self._normalize_mouse_button(name)] = 1 if pressed else 0

    def get_state(self) -> dict:
        with self._lock:
            state = {
                "keys": dict(self._keys),
                "mouse_buttons": dict(self._mouse_buttons),
                "mouse_delta": dict(self._mouse_delta),
            }
            self._mouse_delta = {"dx": 0, "dy": 0, "wheel": 0}
            return state

    def _on_press(self, key):
        self.record_key_down(self._key_to_name(key))

    def _on_release(self, key):
        self.record_key_up(self._key_to_name(key))

    def _on_move(self, x, y):
        with self._lock:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (x, y)
                return
            last_x, last_y = self._last_mouse_pos
            self._last_mouse_pos = (x, y)
        self.record_mouse_move(x - last_x, y - last_y)

    def _on_click(self, _x, _y, button, pressed):
        self.record_mouse_button(str(button).replace("Button.", ""), pressed)

    def _on_scroll(self, _x, _y, _dx, dy):
        self.record_mouse_wheel(dy)

    def _key_to_name(self, key) -> str:
        if hasattr(key, "char") and key.char:
            return key.char.lower()
        if hasattr(key, "name"):
            return key.name.lower()
        return str(key).replace("Key.", "").lower()

    def _normalize_key_name(self, name: str) -> str:
        return name.replace("Key.", "").lower()

    def _normalize_mouse_button(self, name: str) -> str:
        return name.replace("Button.", "").lower()
