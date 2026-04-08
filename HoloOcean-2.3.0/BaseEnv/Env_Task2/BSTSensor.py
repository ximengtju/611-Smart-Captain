import holoocean
import numpy as np
from pynput import keyboard
from multiprocessing import Process, Queue
import time
from holoocean.environments import HoloOceanEnvironment
from queue import Empty
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Env_Task2.env.env_config import BASE_CONFIG
scenario = copy.deepcopy(BASE_CONFIG["auv_config"])

def temp_fx(location):
    # NOTE: It's required that we name this parameter "location"
    x, y, z = location
    base_temp = 4
    alpha = -.2
    beta = .05
    return base_temp + alpha * z + beta * y


pressed_keys = list()
force = 200

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
    on_release=on_release)
listener.start()

def parse_keys(keys, val, visualizer=None, location=None):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += .1 * val
        command[[5,6]] -= .1 * val
    if 'l' in keys:
        command[[4,7]] -= .1 * val
        command[[5,6]] += .1 * val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    if 'm' in keys:
        if visualizer is not None and location is not None:
            visualizer.update_position(location)


    return command

def plot_process(location_queue, scenario, custom_temp):
    print("Process started")
#    visualizer = HoloOceanEnvironment.initialize_bst_graphs(config=scenario, temperature_func=custom_temp)
    visualizer = HoloOceanEnvironment.initialize_bst_graphs(config=scenario)
    while True:
        latest_location = None
        while True:
            try:
                latest_location = location_queue.get_nowait()
            except Empty:
                break

        if latest_location is not None:
            visualizer.update_position(latest_location)

        time.sleep(.01)

if __name__ == "__main__":
    time = 0
    with holoocean.make(scenario_cfg=scenario, start_world=True) as env:
        # Here we pass in the custom function into the live graph instance to get accurate figures
        queue = Queue()
        queue.put(scenario["agents"][0]["location"])
        p = Process(target=plot_process, args=(queue, scenario, temp_fx))
        p.start()
        env.set_temperature_function("auv0", temp_fx)
        while True:
            # We need to pass the same custom function into the BST sensor to get accurate readings in the terminal
            sensor_data = env.tick()
            location = sensor_data["LocationSensor"]
            queue.put(location)
            bst_data = sensor_data["BSTSensor"]
            # print(bst_data)
            time += 1

            if '-' in pressed_keys:
                break

            # Your control logic
            command = parse_keys(pressed_keys, force)
            env.act("auv0", command)

        # Clean up
        p.join()