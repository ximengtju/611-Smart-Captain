import holoocean
import numpy as np
from pynput import keyboard

name = "agent"
config = {
        "name": "test",
        "world": "Dam",
        "package_name": "Ocean",
        "main_agent": name,
        "agents": [
            {
                "agent_name": name,
                "agent_type": "BlueROV2",
                "sensors": [
                ],
                "control_scheme": 0,
                "location": [0, 0, -12.0],
                "rotation": [0, 0, 180]
            }
        ]
    }

def parse_keys(keys, val):
    command = np.zeros(8)

    if 'i' in keys: # ascend
        command[0:4] += val
    if 'k' in keys: # descend
        command[0:4] -= val
    if 'j' in keys: # yaw left
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys: # yaw right
        command[[4,7]] -= val
        command[[5,6]] += val
    if 'w' in keys: # forward
        command[4:8] += 10*val
    if 's' in keys: # backward
        command[4:8] -= 10*val
    if 'a' in keys: # strafe left
        command[[4,6]] += 10*val
        command[[5,7]] -= 10*val
    if 'd' in keys: # strafe right
        command[[4,6]] -= 10*val
        command[[5,7]] += 10*val
    return command

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

pressed_keys = list()
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

with holoocean.make(scenario_cfg=config) as env:
    force = 25
    while True:
        if 'q' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)
        env.act(name, command)
        state = env.tick()