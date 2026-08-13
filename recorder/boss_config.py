from dataclasses import dataclass
import json
import os


@dataclass(frozen=True)
class BossConfig:
    boss_id: str = "unknown_boss"
    boss_type: str = "standard"
    difficulty: str = "normal"
    weapon: str = "unknown_weapon"
    required_action: str | None = None
    control: str = "keyboard_mouse"
    label_source: str = "manual"

    def to_dict(self) -> dict:
        return {
            "boss_id": self.boss_id,
            "boss_type": self.boss_type,
            "difficulty": self.difficulty,
            "weapon": self.weapon,
            "required_action": self.required_action,
            "control": self.control,
            "label_source": self.label_source,
        }


def load_boss_config(path: str | None = None, env: dict | None = None) -> BossConfig:
    values = BossConfig().to_dict()

    if path:
        with open(path, "r", encoding="utf-8") as f:
            file_values = json.load(f)
        for key in values:
            if key in file_values:
                values[key] = file_values[key]

    source_env = os.environ if env is None else env
    env_map = {
        "boss_id": "NIGHTREIGN_BOSS_ID",
        "boss_type": "NIGHTREIGN_BOSS_TYPE",
        "difficulty": "NIGHTREIGN_DIFFICULTY",
        "weapon": "NIGHTREIGN_WEAPON",
        "required_action": "NIGHTREIGN_REQUIRED_ACTION",
        "control": "NIGHTREIGN_CONTROL",
    }
    for key, env_name in env_map.items():
        if source_env.get(env_name):
            values[key] = source_env[env_name]

    return BossConfig(**values)
