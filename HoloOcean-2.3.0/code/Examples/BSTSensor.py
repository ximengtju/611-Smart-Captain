import holoocean
import numpy as np
from pynput import keyboard
from multiprocessing import Process, Queue
import time
from holoocean.environments import HoloOceanEnvironment
from queue import Empty

def temp_fx(location):
    x, y, z = location
    base_temp = 4
    alpha = -0.2
    beta = 0.05
    return base_temp + alpha * z + beta * y

scenario = {
    "name": "bst_demo",
    "world": "PierHarbor",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 60,
    "frames_per_sec": 60,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "BSTSensor",
                    "socket": "COM",
                    "configuration": {
                        "biomass_clusters": [
                            {"position": [10, 10, -10], "strength": 2, "falloff": 10}
                        ]
                    }
                },
                {
                    "sensor_type": "LocationSensor",
                    "socket": "COM"
                }
            ],
            "control_scheme": 0,
            "location": [-20, 10, -10]
        }
    ]
}

pressed_keys = []
force = 200

def on_press(key):
    global pressed_keys
    if hasattr(key, "char") and key.char is not None:
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, "char") and key.char in pressed_keys:
        pressed_keys.remove(key.char)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4, 7]] += 0.1 * val
        command[[5, 6]] -= 0.1 * val
    if 'l' in keys:
        command[[4, 7]] -= 0.1 * val
        command[[5, 6]] += 0.1 * val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if 'd' in keys:
        command[[4, 6]] -= val
        command[[5, 7]] += val

    return command

def plot_process(location_queue, scenario, custom_temp):
    print("BST 可视化进程已启动")
    visualizer = HoloOceanEnvironment.initialize_bst_graphs(
        config=scenario,
#        temperature_func=custom_temp
    )

    while True:
        latest_location = None
        while True:
            try:
                latest_location = location_queue.get_nowait()
            except Empty:
                break

        if latest_location is not None:
            visualizer.update_position(latest_location)

        time.sleep(0.01)

if __name__ == "__main__":
    queue = Queue()
    queue.put(scenario["agents"][0]["location"])
    p = Process(target=plot_process, args=(queue, scenario, temp_fx))
    p.start()

    try:
        with holoocean.make(
            scenario_cfg=scenario,
            start_world=True,
            show_viewport=True
        ) as env:
            env.set_temperature_function("auv0", temp_fx)

            while True:
                sensor_data = env.tick()

                # 2.3.0 文档定义下，tick() 返回 agent -> state 的字典
#                agent_state = sensor_data["auv0"]

                location = sensor_data["LocationSensor"]
                bst_data = sensor_data["BSTSensor"]

                queue.put(location)

                biomass, salinity, temperature = bst_data
                print(
                    f"loc={location}, "
                    f"biomass={biomass:.3f}, "
                    f"salinity={salinity:.3f}, "
                    f"temperature={temperature:.3f}"
                )

                if "-" in pressed_keys:
                    break

                command = parse_keys(pressed_keys, force)
                env.act("auv0", command)

    finally:
        p.terminate()
        p.join()