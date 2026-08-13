from recorder.keyboard_mouse_reader import KeyboardMouseReader


def test_key_press_and_release_updates_snapshot():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_key_down("w")
    state = reader.get_state()
    assert state["keys"]["w"] == 1

    reader.record_key_up("w")
    state = reader.get_state()
    assert state["keys"]["w"] == 0


def test_mouse_delta_is_accumulated_then_reset_after_snapshot():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_mouse_move(5, -3)
    reader.record_mouse_move(2, 4)

    first = reader.get_state()
    second = reader.get_state()

    assert first["mouse_delta"] == {"dx": 7, "dy": 1, "wheel": 0}
    assert second["mouse_delta"] == {"dx": 0, "dy": 0, "wheel": 0}


def test_mouse_button_state_is_recorded():
    reader = KeyboardMouseReader(start_listeners=False)

    reader.record_mouse_button("left", True)
    state = reader.get_state()
    assert state["mouse_buttons"]["left"] == 1

    reader.record_mouse_button("left", False)
    state = reader.get_state()
    assert state["mouse_buttons"]["left"] == 0
