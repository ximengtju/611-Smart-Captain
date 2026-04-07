import holoocean
import numpy as np
from pynput import keyboard
from env.env_config import BASE_CONFIG

pressed_keys = list()
force = 25

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)
    
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release
)
listener.start()

def parse_keys(keys, val):
    
    command = np.zeros(8)
    # 上升（空格）
    if 'i' in keys:
        command[0:4] += val
    # 下降（Ctrl）
    if 'j' in keys:
        command[0:4] -= val
    # 视角左移（q）
    if 'q' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    # 视角右移（e）
    if 'e' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val
    # 前进（w）
    if 'w' in keys:
        command[4:8] += val
    # 后退（s）
    if 's' in keys:
        command[4:8] -= val
    # 向左平移（a）
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    # 向右平移（d）
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val
        print("Command:", command)
    return command

scenario_cfg = BASE_CONFIG["auv_config"]
with holoocean.make(scenario_cfg=scenario_cfg) as env:
    while True:
        if 'r' in pressed_keys:
            break
        command = parse_keys(pressed_keys, force)
        env.act("auv0", command)
        state = env.tick()

        
