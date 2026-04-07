import holoocean
import numpy as np
from pynput import keyboard

print(holoocean.__version__)
scenario = {
    "name": "test_currents",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 30,
    "frames_per_sec": 90,
    "current": {
        "vehicle_debugging": True
    },
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "LocationSensor",
                    "socket": "COM",
                    "configuration": {
                        "Sigma": 0
                    }
                }
            ],
            "control_scheme": 0,
            "location": [5, -5, -15]
        },
        {
            "agent_name": "auv1",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "LocationSensor",
                    "socket": "COM",
                    "configuration": {
                        "Sigma": 0
                    }
                }
            ],
            "control_scheme": 0,
            "location": [-5, -5, -15]
        }
    ]
}

pressed_keys = list()
force = 10
flashlight_on = False   # 新增：记录灯是否开启

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char') and key.char is not None:
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char') and key.char in pressed_keys:
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += .25 * val
        command[[5,6]] -= .25 * val
    if 'l' in keys:
        command[[4,7]] -= .25 * val
        command[[5,6]] += .25 * val

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

    return command

def vortex_field(location):
    x, y, z = location
    if z > 0:
        return [0, 0, 0]
    strength = 5.0
    r_squared = x**2 + y**2 + 1e-5
    dx = -y / r_squared * strength
    dy = x / r_squared * strength
    dz = 0.2 * np.cos(0.1 * r_squared)
    return [3*dx, 3*dy, 3*dz]

vehicles = ['auv0', 'auv1']
map_dimensions = [100, 100, 35]

with holoocean.make(scenario_cfg=scenario, start_world=True) as env:

    # 启动就开灯
    env.turn_on_flashlight(
        "flashlight1",
        intensity=50000,
        beam_width=20,
        angle_pitch=-10,
        angle_yaw=0,
        color_R=1,
        color_G=1,
        color_B=1
    )
    flashlight_on = True

    env.draw_debug_vector_field(
        vortex_field,
        location=[0, 0, -15],
        vector_field_dimensions=map_dimensions,
        spacing=4,
        arrow_size=0.5,
        arrow_thickness=7,
        lifetime=0
    )

    while True:
        tick_info = env.tick()

        # 新增：按 o 开关灯
        if 'o' in pressed_keys:
            if flashlight_on:
                env.turn_off_flashlight("flashlight1")
                flashlight_on = False
                print("flashlight1 已关闭")
            else:
                env.turn_on_flashlight(
                    "flashlight1",
                    intensity=50000,
                    beam_width=20,
                    angle_pitch=-10,
                    angle_yaw=0,
                    color_R=1,
                    color_G=1,
                    color_B=1
                )
                flashlight_on = True
                print("flashlight1 已打开")

            # 防止一直按住 o 导致疯狂切换
            pressed_keys.remove('o')

        command = parse_keys(pressed_keys, force)
        env.act("auv0", command)

        for vehicle in vehicles:
            location = tick_info[vehicle]["LocationSensor"]
            current_velocity = vortex_field(location)
            env.set_ocean_currents(vehicle, current_velocity)

            speed = np.linalg.norm(current_velocity)
            print(f"{vehicle}: vel={current_velocity}, |v|={speed:.3f}")