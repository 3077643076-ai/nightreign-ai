import json

from recorder.boss_config import load_boss_config


def test_default_boss_config_is_manual_unknown():
    config = load_boss_config(env={})
    assert config.to_dict() == {
        "boss_id": "unknown_boss",
        "boss_type": "standard",
        "difficulty": "normal",
        "weapon": "unknown_weapon",
        "required_action": None,
        "control": "keyboard_mouse",
        "label_source": "manual",
    }


def test_environment_overrides_boss_config():
    config = load_boss_config(env={
        "NIGHTREIGN_BOSS_ID": "tree_boss",
        "NIGHTREIGN_BOSS_TYPE": "mechanic",
        "NIGHTREIGN_DIFFICULTY": "normal",
        "NIGHTREIGN_WEAPON": "storm_weapon",
        "NIGHTREIGN_REQUIRED_ACTION": "weapon_art",
    })
    assert config.boss_id == "tree_boss"
    assert config.boss_type == "mechanic"
    assert config.required_action == "weapon_art"


def test_json_file_overrides_defaults(tmp_path):
    path = tmp_path / "boss.json"
    path.write_text(json.dumps({
        "boss_id": "grafted_scion",
        "boss_type": "standard",
        "difficulty": "normal",
        "weapon": "greatsword",
        "required_action": None,
    }), encoding="utf-8")

    config = load_boss_config(path=str(path), env={})
    assert config.boss_id == "grafted_scion"
    assert config.weapon == "greatsword"
