"""内存读取器：从游戏进程读取 HP/FP/耐力/坐标 等精确数值。

用于训练时提供 ground truth 标签，推理时不使用。
需要先绕过 EAC（用 Hexinton 离线启动器启动游戏）。

偏移来源：Hexinton v1.1.0 CT 表 (2025-12-17)
用法：
    from memory_reader import MemoryReader
    mr = MemoryReader()
    mr.open()
    state = mr.read()  # dict
"""

import struct
import ctypes
from ctypes import wintypes


PROCESS_NAME = "nightreign.exe"

# AOB 签名 → 用于定位 WorldChrMan / GameDataMan / LockTgtMan 指针
# Elden Ring Nightreign 1.1.0
AOB_SIGNATURES = {
    # "48 8B 05 ?? ?? ?? ?? 0F 28 F1 48 85 C0" = mov rax, [rip+XXXXXXXX]
    "WorldChrMan": bytes([0x48, 0x8B, 0x05, 0x00, 0x00, 0x00, 0x00,
                           0x0F, 0x28, 0xF1, 0x48, 0x85, 0xC0]),
    # "48 8B 0D ?? ?? ?? ?? F3 48 0F 2C C0" = mov rcx, [rip+XXXXXXXX]
    "GameDataMan": bytes([0x48, 0x8B, 0x0D, 0x00, 0x00, 0x00, 0x00,
                           0xF3, 0x48, 0x0F, 0x2C, 0xC0]),
    # "48 8B 35 ?? ?? ?? ?? 48 81 C6 ?? ?? ?? ?? 4C 8B 2D ?? ?? ?? ?? ..." = mov r13, [rip+...]
    # rip_offset=0xE 指示从第14字节的指令(4C 8B 2D)解析RIP地址
    "LockTgtMan": bytes([0x48, 0x8B, 0x35, 0x00, 0x00, 0x00, 0x00,
                          0x48, 0x81, 0xC6, 0x00, 0x00, 0x00, 0x00,
                          0x4C, 0x8B, 0x2D, 0x00, 0x00, 0x00, 0x00,
                          0x4C, 0x89, 0x6C, 0x24, 0x00,
                          0x4D, 0x85, 0xED]),
}

# 偏移常量（来自 CT 表 Lua 脚本 + registerSymbol("MainPlayerOffset","174e8")）
MAIN_PLAYER_OFFSET = 0x174E8

# 玩家属性偏移（从 chrAsm + 0 开始，即 statsBase）
STAT_HP           = 0x140   # int32
STAT_MAX_HP       = 0x144   # int32
STAT_FP           = 0x150   # int32
STAT_MAX_FP       = 0x154   # int32
STAT_STAMINA      = 0x15C   # int32
STAT_MAX_STAMINA  = 0x160   # int32

# 坐标偏移（从 chrAsm + 0x68 开始，即 posBase）
POS_X = 0x70   # float
POS_Y = 0x74   # float
POS_Z = 0x78   # float

# 卢恩（从 GameDataMan + 8 开始）
RUNES_OFFSET = 0x2F4  # int32, chain: [[GameDataMan] + 8] + 0x2F4

# ── 动画偏移（从 chrAsm 开始）─────────────────────
# 当前动画ID: [[chrAsm + 0x80] + 0x98]  (GetAnim)
# 强制播放动画: [[chrAsm + 0x58] + 0x18]  (PlayAnimByNumericalId)
ANIM_PTR_OFFSET    = 0x80   # → 动画状态指针
ANIM_ID_OFFSET     = 0x98   # → 当前动画ID (int32)
ANIM_CTRL_OFFSET   = 0x58   # → 动画控制指针（用于强制播放）
ANIM_PLAY_OFFSET   = 0x18   # → 写入动画ID (int32)

# 死亡动画ID范围（来自 Inf Revivify 脚本）
DEATH_ANIMS = {
    4000, 4100, 202040,
}
DEATH_ANIM_RANGES = [
    (17000, 19000),
    (60000, 99000),
]

# ── 复活道具 ─────────────────────────────────
# 道具 ID: 0x400002BC (来自 CT "Drop Revive Item")
# 默认 4 个格子，训练中低于 3 个就手动补货
# 地址需要回家用 CE 确认：塞 8 个 → 搜索 8 → 用掉 1 个 → 搜索 7 → 找地址
REVIVE_ITEM_ID = 0x400002BC
# 物品栏起始地址（待验证，通常是 [[GameDataMan] + 0x8] + 某偏移）
# INVENTORY_GOODS_OFFSET = 0x???  # 回家确认

# ── Boss/锁定目标偏移（需要在游戏上验证）────────
# LockTgtMan → [+0x8] → TargetHandle → [+0x190] → ChrIns → [+0x190] → ChrAsm
LOCKTGTOFF_TARGET_HANDLE = 0x8
TARGET_TO_CHRINS        = 0x190  # 待验证
CHRINS_TO_CHRASM        = 0x190  # 待验证（玩家是0x1B8，敌人可能不同）


class MemoryReader:
    """读 Elden Ring Nightreign 内存，返回精确的游戏状态。"""

    def __init__(self, process_name: str = PROCESS_NAME):
        self._process_name = process_name
        self._handle = None
        self._pid = None
        self._base_addr = 0

        # 缓存的解析地址
        self._world_chr_man_ptr = 0   # WorldChrMan 指针所在地址
        self._game_data_man_ptr = 0   # GameDataMan 指针所在地址
        self._lock_tgt_man_ptr = 0    # LockTgtMan 指针所在地址（读Boss用）

    # ============================================================
    # 连接
    # ============================================================

    def open(self) -> bool:
        """打开游戏进程，AOB 扫描定位 WorldChrMan / GameDataMan。"""
        PROCESS_ALL_ACCESS = 0x1F0FFF

        self._pid = self._find_pid(self._process_name)
        if self._pid is None:
            print(f"[MemoryReader] 找不到进程: {self._process_name}")
            return False

        self._handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_ALL_ACCESS, False, self._pid
        )
        if not self._handle:
            print(f"[MemoryReader] 无法打开进程 (PID={self._pid})，请用管理员运行")
            return False

        self._base_addr = self._get_module_base(self._pid, self._process_name)
        if self._base_addr == 0:
            print(f"[MemoryReader] 无法获取模块基址")
            return False

        # AOB 扫描定位 WorldChrMan / GameDataMan / LockTgtMan
        self._world_chr_man_ptr = self._aob_scan(AOB_SIGNATURES["WorldChrMan"])
        self._game_data_man_ptr = self._aob_scan(AOB_SIGNATURES["GameDataMan"])
        self._lock_tgt_man_ptr = self._aob_scan(AOB_SIGNATURES["LockTgtMan"], rip_offset=0xE)

        if self._world_chr_man_ptr == 0:
            print("[MemoryReader] 警告: 未找到 WorldChrMan AOB，偏移可能需更新")
        if self._game_data_man_ptr == 0:
            print("[MemoryReader] 警告: 未找到 GameDataMan AOB，卢恩读取不可用")
        if self._lock_tgt_man_ptr == 0:
            print("[MemoryReader] 警告: 未找到 LockTgtMan AOB，Boss动画读取不可用")

        print(f"[MemoryReader] 已连接 (PID={self._pid}, base=0x{self._base_addr:X})")
        print(f"[MemoryReader] WorldChrMan=0x{self._world_chr_man_ptr:X} "
              f"GameDataMan=0x{self._game_data_man_ptr:X} "
              f"LockTgtMan=0x{self._lock_tgt_man_ptr:X}")
        return True

    def close(self):
        if self._handle:
            ctypes.windll.kernel32.CloseHandle(self._handle)
            self._handle = None

    @property
    def connected(self) -> bool:
        return self._handle is not None and self._handle != 0

    # ============================================================
    # 主接口
    # ============================================================

    def read(self) -> dict | None:
        """读取所有游戏状态。"""
        if not self.connected or self._world_chr_man_ptr == 0:
            return None

        # ── 玩家属性链 ──
        # [[[[WorldChrMan] + 0x174E8] + 0x1B8] + 0x0] → statsBase
        world_chr_man = self._read_ptr(self._world_chr_man_ptr)
        if world_chr_man is None or world_chr_man == 0:
            return None

        ptr1 = self._read_ptr(world_chr_man + MAIN_PLAYER_OFFSET)
        if ptr1 is None or ptr1 == 0:
            return None

        chr_asm = self._read_ptr(ptr1 + 0x1B8)
        if chr_asm is None or chr_asm == 0:
            return None

        stats_base = self._read_ptr(chr_asm + 0x0)
        if stats_base is None or stats_base == 0:
            return None

        # ── 坐标链 ──
        # [[[[WorldChrMan] + 0x174E8] + 0x1B8] + 0x68] → posBase
        pos_base = self._read_ptr(chr_asm + 0x68)

        # ── 读取数值 ──
        hp = self._read_i32(stats_base + STAT_HP)
        max_hp = self._read_i32(stats_base + STAT_MAX_HP)
        fp = self._read_i32(stats_base + STAT_FP)
        max_fp = self._read_i32(stats_base + STAT_MAX_FP)
        stamina = self._read_i32(stats_base + STAT_STAMINA)
        max_stamina = self._read_i32(stats_base + STAT_MAX_STAMINA)

        pos_x = self._read_f32(pos_base + POS_X) if pos_base else None
        pos_y = self._read_f32(pos_base + POS_Y) if pos_base else None
        pos_z = self._read_f32(pos_base + POS_Z) if pos_base else None

        # 卢恩: [[GameDataMan] + 0x8] + 0x2F4
        runes = None
        if self._game_data_man_ptr != 0:
            gdm = self._read_ptr(self._game_data_man_ptr)
            if gdm and gdm != 0:
                pgd = self._read_ptr(gdm + 0x8)
                if pgd and pgd != 0:
                    runes = self._read_i32(pgd + RUNES_OFFSET)

        return {
            "hp":           hp,
            "max_hp":       max_hp,
            "hp_pct":       hp / max_hp if hp is not None and max_hp and max_hp > 0 else -1.0,
            "fp":           fp,
            "max_fp":       max_fp,
            "fp_pct":       fp / max_fp if fp is not None and max_fp and max_fp > 0 else -1.0,
            "stamina":      stamina,
            "max_stamina":  max_stamina,
            "stamina_pct":  stamina / max_stamina if stamina is not None and max_stamina and max_stamina > 0 else -1.0,
            "pos_x":        pos_x,
            "pos_y":        pos_y,
            "pos_z":        pos_z,
            "runes":        runes,
        }

    # ============================================================
    # 动画
    # ============================================================

    def read_anim_id(self) -> int | None:
        """读取玩家当前动画ID。"""
        if not self.connected or self._world_chr_man_ptr == 0:
            return None

        world_chr_man = self._read_ptr(self._world_chr_man_ptr)
        if world_chr_man is None or world_chr_man == 0:
            return None

        ptr1 = self._read_ptr(world_chr_man + MAIN_PLAYER_OFFSET)
        if ptr1 is None or ptr1 == 0:
            return None

        chr_asm = self._read_ptr(ptr1 + 0x1B8)
        if chr_asm is None or chr_asm == 0:
            return None

        anim_ptr = self._read_ptr(chr_asm + ANIM_PTR_OFFSET)  # +0x80
        if anim_ptr is None or anim_ptr == 0:
            return None

        return self._read_i32(anim_ptr + ANIM_ID_OFFSET)  # +0x98

    def is_dead_anim(self, anim_id: int) -> bool:
        """判断动画ID是否为死亡动画。"""
        if anim_id in DEATH_ANIMS:
            return True
        for lo, hi in DEATH_ANIM_RANGES:
            if lo <= anim_id <= hi:
                return True
        return False

    def read_boss_anim_id(self) -> int | None:
        """读取锁定目标（Boss）的当前动画ID。

        ⚠️ 偏移链需要在游戏上验证（CHRINS_TO_CHRASM 可能不是 0x190）。
        """
        if not self.connected or self._lock_tgt_man_ptr == 0:
            return None

        # [[LockTgtMan] + 0x8] → TargetHandle
        target_handle = self._read_ptr(self._lock_tgt_man_ptr + LOCKTGTOFF_TARGET_HANDLE)
        if target_handle is None or target_handle == 0:
            return None

        # [TargetHandle + 0x190] → ChrIns（待验证）
        chr_ins = self._read_ptr(target_handle + TARGET_TO_CHRINS)
        if chr_ins is None or chr_ins == 0:
            return None

        # [ChrIns + 0x190] → ChrAsm（待验证）
        chr_asm = self._read_ptr(chr_ins + CHRINS_TO_CHRASM)
        if chr_asm is None or chr_asm == 0:
            return None

        # ChrAsm + 0x80 → anim_ptr → +0x98 → anim_id
        anim_ptr = self._read_ptr(chr_asm + ANIM_PTR_OFFSET)
        if anim_ptr is None or anim_ptr == 0:
            return None

        return self._read_i32(anim_ptr + ANIM_ID_OFFSET)

    # ============================================================
    # 写内存（复活 / 恢复用）
    # ============================================================

    def _write_bytes(self, address: int, data: bytes) -> bool:
        """写入字节到进程内存。"""
        if not self._handle or address == 0:
            return False
        buf = ctypes.create_string_buffer(data, len(data))
        bytes_written = ctypes.c_size_t(0)
        ok = ctypes.windll.kernel32.WriteProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            buf,
            len(data),
            ctypes.byref(bytes_written),
        )
        return ok != 0 and bytes_written.value == len(data)

    def write_i32(self, address: int, value: int) -> bool:
        return self._write_bytes(address, struct.pack("<i", value))

    def write_f32(self, address: int, value: float) -> bool:
        return self._write_bytes(address, struct.pack("<f", value))

    def _write_anim_id(self, anim_id: int) -> bool:
        """强制播放指定动画（用于复活时切到站立动画）。"""
        if not self.connected or self._world_chr_man_ptr == 0:
            return False

        world_chr_man = self._read_ptr(self._world_chr_man_ptr)
        if world_chr_man is None or world_chr_man == 0:
            return False

        ptr1 = self._read_ptr(world_chr_man + MAIN_PLAYER_OFFSET)
        if ptr1 is None or ptr1 == 0:
            return False

        chr_asm = self._read_ptr(ptr1 + 0x1B8)
        if chr_asm is None or chr_asm == 0:
            return False

        ctrl_ptr = self._read_ptr(chr_asm + ANIM_CTRL_OFFSET)  # +0x58
        if ctrl_ptr is None or ctrl_ptr == 0:
            return False

        return self.write_i32(ctrl_ptr + ANIM_PLAY_OFFSET, anim_id)  # +0x18

    def revive(self) -> bool:
        """检测死亡状态并复活：回满HP/FP/耐力 + 切站立动画。

        返回 True 表示执行了复活操作。
        用法：训练循环中每帧调用，仅在死亡时触发复活。
        """
        anim = self.read_anim_id()
        if anim is None or not self.is_dead_anim(anim):
            return False

        # 读完整状态确认 HP <= 0
        state = self.read()
        if state is None:
            return False

        hp = state["hp"]
        if hp is not None and hp <= 0:
            # 找到 stats_base 地址来回血
            # 复用指针链
            wcm = self._read_ptr(self._world_chr_man_ptr)
            if wcm:
                p1 = self._read_ptr(wcm + MAIN_PLAYER_OFFSET)
                if p1:
                    chr_asm = self._read_ptr(p1 + 0x1B8)
                    if chr_asm:
                        stats_base = self._read_ptr(chr_asm + 0x0)
                        if stats_base and state["max_hp"]:
                            self.write_i32(stats_base + STAT_HP, state["max_hp"])
                            if state["max_fp"]:
                                self.write_i32(stats_base + STAT_FP, state["max_fp"])
                            if state["max_stamina"]:
                                self.write_i32(stats_base + STAT_STAMINA, state["max_stamina"])
            # 切站立动画
            self._write_anim_id(0)
            return True

        return False

    # ============================================================
    # AOB 扫描
    # ============================================================

    def _aob_scan(self, pattern: bytes, wildcard_offset: int = 3, rip_offset: int = 0) -> int:
        """在进程内存中搜索 AOB 模式，返回 RIP-relative 目标地址。

        pattern: 字节模式，wildcard 位置用 0x00 占位。
        wildcard_offset: 默认 3（指令 opcode 3 字节后的 disp）。
        rip_offset: 要解析的 RIP-relative 指令距匹配开头的偏移（默认 0=第一条指令）。

        返回：RIP-relative 解析后的目标地址（即 CE 中的符号值）。
        """
        if not self._handle or self._base_addr == 0:
            return 0

        # 获取模块大小
        module_size = self._get_module_size(self._pid, self._process_name)
        if module_size == 0:
            return 0

        # 读取整个模块（分块扫描，每块 64MB）
        chunk_size = 64 * 1024 * 1024
        mask = bytes([0xFF if b != 0x00 else 0x00 for b in pattern])

        for offset in range(0, module_size, chunk_size):
            size = min(chunk_size, module_size - offset)
            data = self._read_bytes(self._base_addr + offset, size)
            if data is None:
                continue

            pos = self._find_pattern(data, pattern, mask)
            if pos != -1:
                match_addr = self._base_addr + offset + pos
                # 从 rip_offset 位置读取 displacement
                disp_addr = match_addr + rip_offset + 3  # opcode 后 3 字节
                disp = self._read_i32(disp_addr)
                if disp is None:
                    continue
                # x64 RIP-relative: target = instruction_addr + 7 + displacement
                return match_addr + rip_offset + 7 + disp

        return 0

    @staticmethod
    def _find_pattern(data: bytes, pattern: bytes, mask: bytes) -> int:
        """Boyer-Moore 风格搜索。"""
        plen = len(pattern)
        for i in range(len(data) - plen + 1):
            match = True
            for j in range(plen):
                if mask[j] == 0x00:
                    continue  # wildcard
                if data[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                return i
        return -1

    # ============================================================
    # 内存读写
    # ============================================================

    def _read_i32(self, address: int) -> int | None:
        buf = self._read_bytes(address, 4)
        if buf is None:
            return None
        return struct.unpack("<i", buf)[0]

    def _read_f32(self, address: int) -> float | None:
        buf = self._read_bytes(address, 4)
        if buf is None:
            return None
        return round(struct.unpack("<f", buf)[0], 2)

    def _read_ptr(self, address: int) -> int | None:
        buf = self._read_bytes(address, 8)
        if buf is None:
            return None
        return struct.unpack("<Q", buf)[0]

    def _read_bytes(self, address: int, size: int) -> bytes | None:
        if not self._handle or address == 0:
            return None
        buf = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        ok = ctypes.windll.kernel32.ReadProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            buf,
            size,
            ctypes.byref(bytes_read),
        )
        if not ok or bytes_read.value != size:
            return None
        return buf.raw

    # ============================================================
    # 进程 / 模块信息
    # ============================================================

    @staticmethod
    def _find_pid(name: str) -> int | None:
        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == -1:
            return None

        class PROCESSENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("cntUsage", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.POINTER(wintypes.ULARGE_INTEGER)),
                ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD),
                ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", wintypes.LONG),
                ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_char * 260),
            ]

        entry = PROCESSENTRY32()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if kernel32.Process32First(snapshot, ctypes.byref(entry)):
            while True:
                exe_name = entry.szExeFile.decode("utf-8", errors="ignore").lower()
                if exe_name == name.lower():
                    kernel32.CloseHandle(snapshot)
                    return entry.th32ProcessID
                if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                    break
        kernel32.CloseHandle(snapshot)
        return None

    @staticmethod
    def _get_module_base(pid: int, module_name: str) -> int:
        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000008, pid)
        if snapshot == -1:
            return 0

        class MODULEENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("th32ModuleID", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("GlblcntUsage", wintypes.DWORD),
                ("ProccntUsage", wintypes.DWORD),
                ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
                ("modBaseSize", wintypes.DWORD),
                ("hModule", wintypes.HMODULE),
                ("szModule", ctypes.c_char * 256),
                ("szExePath", ctypes.c_char * 260),
            ]

        entry = MODULEENTRY32()
        entry.dwSize = ctypes.sizeof(MODULEENTRY32)
        if kernel32.Module32First(snapshot, ctypes.byref(entry)):
            while True:
                mod = entry.szModule.decode("utf-8", errors="ignore").lower()
                if mod == module_name.lower():
                    kernel32.CloseHandle(snapshot)
                    return ctypes.cast(entry.modBaseAddr, ctypes.c_void_p).value or 0
                if not kernel32.Module32Next(snapshot, ctypes.byref(entry)):
                    break
        kernel32.CloseHandle(snapshot)
        return 0

    @staticmethod
    def _get_module_size(pid: int, module_name: str) -> int:
        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000008, pid)
        if snapshot == -1:
            return 0

        class MODULEENTRY32(ctypes.Structure):
            _fields_ = [
                ("dwSize", wintypes.DWORD),
                ("th32ModuleID", wintypes.DWORD),
                ("th32ProcessID", wintypes.DWORD),
                ("GlblcntUsage", wintypes.DWORD),
                ("ProccntUsage", wintypes.DWORD),
                ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
                ("modBaseSize", wintypes.DWORD),
                ("hModule", wintypes.HMODULE),
                ("szModule", ctypes.c_char * 256),
                ("szExePath", ctypes.c_char * 260),
            ]

        entry = MODULEENTRY32()
        entry.dwSize = ctypes.sizeof(MODULEENTRY32)
        if kernel32.Module32First(snapshot, ctypes.byref(entry)):
            while True:
                mod = entry.szModule.decode("utf-8", errors="ignore").lower()
                if mod == module_name.lower():
                    kernel32.CloseHandle(snapshot)
                    return entry.modBaseSize
                if not kernel32.Module32Next(snapshot, ctypes.byref(entry)):
                    break
        kernel32.CloseHandle(snapshot)
        return 0


# ============================================================
# 自测
# ============================================================

def main():
    import time
    mr = MemoryReader()

    if not mr.open():
        print("\n请确认：")
        print("  1. 游戏已启动（离线模式绕过 EAC）")
        print("  2. 以管理员权限运行此脚本")
        return

    print("\n持续读取中，Ctrl+C 停止...\n")
    try:
        while True:
            state = mr.read()
            anim_id = mr.read_anim_id()
            boss_anim = mr.read_boss_anim_id()

            if state:
                parts = [
                    f"HP={state['hp']}/{state['max_hp']} ({state['hp_pct']:.0%})",
                    f"FP={state['fp']}/{state['max_fp']} ({state['fp_pct']:.0%})",
                    f"SP={state['stamina']}/{state['max_stamina']} ({state['stamina_pct']:.0%})",
                ]
                if anim_id is not None:
                    dead = "☠" if mr.is_dead_anim(anim_id) else ""
                    parts.append(f"Anim={anim_id}{dead}")
                if boss_anim is not None:
                    parts.append(f"BossAnim={boss_anim}")
                if state["pos_x"] is not None:
                    parts.append(
                        f"Pos=({state['pos_x']:.1f}, {state['pos_y']:.1f}, {state['pos_z']:.1f})"
                    )
                if state["runes"] is not None:
                    parts.append(f"Runes={state['runes']}")
                print("\r" + " | ".join(parts), end="", flush=True)
            else:
                print("\r[读取失败]", end="", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n")
    finally:
        mr.close()


if __name__ == "__main__":
    main()
