import cv2
import numpy as np
import torch
import time
import json
import os
import keyboard
import pyautogui
import mss
from ultralytics import YOLO
import ctypes

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT),
    ]

def send_mouse_input(dx, dy):
    input_struct = INPUT()
    input_struct.type = INPUT_MOUSE
    input_struct.mi.dx = dx
    input_struct.mi.dy = dy
    input_struct.mi.mouseData = 0
    input_struct.mi.dwFlags = MOUSEEVENTF_MOVE
    input_struct.mi.time = 0
    input_struct.mi.dwExtraInfo = None
    ctypes.windll.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(input_struct))

MODEL_PATH = "yolov8n.pt"
CONFIG_FILE = "detection_classes.json"
AIM_HOTKEY = "alt"
CONFIDENCE_THRESHOLD = 0.4
DEFAULT_FOV_RADIUS = 150

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        json.dump(["person"], f, indent=4)

with open(CONFIG_FILE, "r") as f:
    target_classes = set(json.load(f))

try:
    fov_radius = int(input("Enter FOV radius (e.g. 150 for 300x300 box): "))
except:
    fov_radius = DEFAULT_FOV_RADIUS

def get_center_and_area(box):
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    area = (x2 - x1) * (y2 - y1)
    return (cx, cy), area

def get_closest_target(results, frame_shape):
    h, w, _ = frame_shape
    screen_center = (w // 2, h // 2)
    closest_target = None
    min_distance = float("inf")

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            class_name = model.names[int(cls)]
            if class_name in target_classes and conf > CONFIDENCE_THRESHOLD:
                box_np = box.cpu().numpy()
                (cx, cy), _ = get_center_and_area(box_np)
                chest_y = int((box_np[1] + box_np[3]) / 2)
                dist = np.hypot(cx - screen_center[0], chest_y - screen_center[1])
                if dist < min_distance:
                    min_distance = dist
                    closest_target = (cx, chest_y)

    return closest_target

def draw_detections(frame, results):
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            class_name = model.names[int(cls)]
            if class_name in target_classes and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                chest_y = int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (cx, chest_y), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

def move_mouse_direct(target, screen_center, monitor):
    if target:
        smoothing = 0.3  # Adjust this (0.2–0.5) to tweak smoothness
        dx = int((target[0] - screen_center[0]) * smoothing)
        dy = int((target[1] - screen_center[1]) * smoothing)
        send_mouse_input(dx, dy)

def run_aimbot():
    sct = mss.mss()
    screen_w, screen_h = pyautogui.size()

    last_time = time.time()
    fps = 0

    while True:
        top = screen_h // 2 - fov_radius
        left = screen_w // 2 - fov_radius
        monitor = {"top": top, "left": left, "width": fov_radius * 2, "height": fov_radius * 2}
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = model(frame, verbose=False, device=device)
        draw_detections(frame, results)

        if keyboard.is_pressed(AIM_HOTKEY):
            target = get_closest_target(results, frame.shape)
            if target:
                move_mouse_direct(target, (fov_radius, fov_radius), monitor)

        # FPS calculation
        current_time = time.time()
        fps = 1.0 / (current_time - last_time)
        last_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("YoloV8 Aimbot", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cv2.destroyAllWindows()

print("Starting YOLOv8 Aimbot...")
print("Hold ALT to aim. Press Q in the capture window to quit.")
run_aimbot()
