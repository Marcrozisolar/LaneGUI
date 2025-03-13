import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk, Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import json
import os


# ---------------- Persistent Accounts ----------------
ACCOUNTS_FILE = "accounts.json"


def load_accounts():
   if os.path.exists(ACCOUNTS_FILE):
       with open(ACCOUNTS_FILE, "r") as file:
           return json.load(file)
   else:
       # Default account if no file exists
       return {"admin": "password123"}


def save_accounts(accounts):
   with open(ACCOUNTS_FILE, "w") as file:
       json.dump(accounts, file)


accounts = load_accounts()


# ---------------- Global Lane Detection Variables ----------------
prev_left_line = None
prev_right_line = None


# ---------------- Lane Detection Functions ----------------
left_kalman = cv2.KalmanFilter(8, 4)
right_kalman = cv2.KalmanFilter(8, 4)
video_path = "C:/Users/Mark/Videos/lane.mp4"
image_path = "D:/Mark/Pictures/arrow-up-glyph-black-icon-vector.jpg"


def init_kalman_filter(kalman):
    dt = 1.0  # time step (you could update this based on fps)
    kalman.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0, 0, 0],
        [0, 1, 0, 0, 0, dt, 0, 0],
        [0, 0, 1, 0, 0, 0, dt, 0],
        [0, 0, 0, 1, 0, 0, 0, dt],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], np.float32)

    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0]
    ], np.float32)

    kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-4
    kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kalman.errorCovPost = np.eye(8, dtype=np.float32)


init_kalman_filter(left_kalman)
init_kalman_filter(right_kalman)


def load_video(video_path):
    return cv2.VideoCapture(video_path)


def load_overlay(image_path, scale_factor=0.2):
    image = Image.open(image_path).convert("RGBA")
    image_array = np.array(image)
    image_bgra = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGRA)

    new_size = (int(image_bgra.shape[1] * scale_factor), int(image_bgra.shape[0] * scale_factor))
    resized_image = cv2.resize(image_bgra, new_size)

    if resized_image.shape[2] == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2BGRA)

    return resized_image


def blend_overlay(frame, overlay, position):
    x, y = position
    overlay_bgr, alpha = cv2.split(overlay)[:3], cv2.split(overlay)[3].astype(float) / 255.0
    overlay_bgr = cv2.merge(overlay_bgr)

    h, w, _ = overlay_bgr.shape
    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        print("Overlay exceeds frame boundaries. Adjust position or scale down the overlay.")
        return frame

    roi = frame[y:y + h, x:x + w].astype(float)
    overlay_bgr = overlay_bgr.astype(float)
    blended = (1 - alpha[..., None]) * roi + (alpha[..., None]) * overlay_bgr
    frame[y:y + h, x:x + w] = blended.astype(np.uint8)
    return frame


def undistort_frame(frame):
    h, w = frame.shape[:2]
    camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1), dtype=np.float32)
    return cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(img, mask)


def detect_lanes(frame):
    height, width = frame.shape[:2]
    bottom_offset = int(0.1 * height)

    roi_vertices = np.array([[
        (int(0.15 * width), height - bottom_offset),  # Narrow left side
        (int(0.35 * width), int(0.75 * height)),  # Narrow upper side and lower the height
        (int(0.65 * width), int(0.75 * height)),  # Narrow upper side and lower the height
        (int(0.85 * width), height - bottom_offset)  # Narrow right side
    ]], dtype=np.int32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 80)
    cropped_edges = region_of_interest(edges, roi_vertices)

    lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, threshold=30, minLineLength=50, maxLineGap=150)
    left_lines = []
    right_lines = []

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0.5:
                    right_lines.append([x1, y1, x2, y2])

    if not (left_lines or right_lines):
        return None, None, roi_vertices, False

    def average_line(lines):
        if len(lines) == 0:
            return None
        x_coords, y_coords = [], []
        for line in lines:
            x_coords.extend([line[0], line[2]])
            y_coords.extend([line[1], line[3]])
        if not x_coords:
            return None
        poly = np.polyfit(y_coords, x_coords, 1)
        slope, intercept = poly
        y_bottom = height - bottom_offset
        y_top = int(0.75 * height)
        x_bottom = int(slope * y_bottom + intercept)
        x_top = int(slope * y_top + intercept)
        return [x_bottom, y_bottom, x_top, y_top]

    left_avg = average_line(left_lines)
    right_avg = average_line(right_lines)
    return left_avg, right_avg, roi_vertices, True


def apply_kalman_filter(kalman, detected_line):
    """
    Smooth the detected lane line using Kalman filter.
    detected_line: [x_bottom, y_bottom, x_top, y_top]
    """
    if detected_line is not None:
        measurement = np.array([
            [np.float32(detected_line[0])],
            [np.float32(detected_line[1])],
            [np.float32(detected_line[2])],
            [np.float32(detected_line[3])]
        ])
        kalman.correct(measurement)
    prediction = kalman.predict()
    # Extract the first 4 state values and convert to int
    smoothed_line = prediction[:4].flatten()
    return [int(x) for x in smoothed_line]

def detect_center_line(left_line, right_line):
    if left_line and right_line:
        x_bottom = (left_line[0] + right_line[0]) // 2
        y_bottom = left_line[1]  # Same y-coordinate as lanes
        x_top = (left_line[2] + right_line[2]) // 2
        y_top = left_line[3]  # Same y-coordinate as lanes
        return [x_bottom, y_bottom, x_top, y_top]
    return None

def rotate_overlay(overlay, angle=90):
    """Rotate the overlay image by the given angle (default 90 degrees to the left)."""
    h, w = overlay.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_overlay = cv2.warpAffine(overlay, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated_overlay


def overlay_lanes(frame, left_line, right_line, roi_vertices, center_line):
    overlay = frame.copy()
    cv2.polylines(overlay, [roi_vertices], isClosed=True, color=(0, 255, 255), thickness=3)

    if left_line is not None:
        cv2.line(overlay, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
    if right_line is not None:
        cv2.line(overlay, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)
    if center_line is not None:
        cv2.line(overlay, (center_line[0], center_line[1]), (center_line[2], center_line[3]), (255, 0, 0), 5)

    return overlay


# ----------------==== MAIN APPLICATION (Tkinter UI) ----------------====
def start_main_app():
   global prev_left, prev_right, prev_center


   root = tk.Tk()
   root.title("Robot Control")
   root.configure(bg="#e0f7fa")
   root.geometry("800x600")
   root.grid_columnconfigure((0, 1), weight=1)
   root.grid_rowconfigure((0, 1), weight=1)


   # Top Left Quadrant: Raw Video
   top_left = tk.Frame(root, bg="#000", width=300, height=300)
   top_left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
   video_label_top_left = tk.Label(top_left, bg="#000")
   video_label_top_left.pack(fill="both", expand=True)


   # Bottom Left Quadrant: Processed Video with Lane & Arrow Overlay
   bottom_left = tk.Frame(root, bg="#b0bec5", width=300, height=300)
   bottom_left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
   video_label_bottom_left = tk.Label(bottom_left, bg="#000")
   video_label_bottom_left.pack(fill="both", expand=True)


   # Top Right Quadrant: Controls
   control_frame = tk.Frame(root, bg="#009688", width=300, height=300)
   control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
   button_styles = {
       "↑": ("FORWARD", 0, 1, "white", "#1976D2"),
       "←": ("LEFT", 1, 0, "white", "#8E24AA"),
       "STOP": ("STOP", 1, 1, "white", "#D32F2F"),
       "▶": ("PLAY", 1, 2, "black", "#FBC02D"),
       "→": ("RIGHT", 1, 3, "white", "#388E3C"),
       "↓": ("BACKWARD", 2, 1, "white", "#F57C00"),
   }


   def move_robot(direction):
       command_output.insert(tk.END, f"Robot moving: {direction}\n")
       command_output.see(tk.END)


   def stop_robot():
       command_output.insert(tk.END, "Robot STOPPED\n")
       command_output.see(tk.END)


   def start_video():
       update_video()  # Kick off video playback


   for text, (command, row, col, fg_color, bg_color) in button_styles.items():
       if text == "▶":
           btn = tk.Button(control_frame, text=text, bg=bg_color, fg=fg_color,
                           width=6, height=2, font=("Arial", 14, "bold"),
                           command=start_video)
       else:
           btn = tk.Button(control_frame, text=text, bg=bg_color, fg=fg_color,
                           width=6, height=2, font=("Arial", 14, "bold"),
                           command=lambda c=command: stop_robot() if c == "STOP" else move_robot(c))
       btn.grid(row=row, column=col, padx=5, pady=5)


   # Bottom Right Quadrant: Command Log
   command_output = scrolledtext.ScrolledText(root, height=10, width=50,
                                                bg="#333", fg="white", font=("Arial", 10))
   command_output.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
   command_output.insert(tk.END, "Command output will appear here...\n")


   # Load video capture
   video_path = "C:/Users/Mark/Videos/lane.mp4"  # Update path if needed
   cap = cv2.VideoCapture(video_path)


   # Load and prepare the PNG overlay image (arrow)
   arrow_img_path = "D:/Mark/Pictures/arrow-up-glyph-black-icon-vector.jpg"  # Update path if needed
   arrow_pil = Image.open(arrow_img_path).convert("RGBA")
   arrow_np = np.array(arrow_pil)
   arrow_np = cv2.cvtColor(arrow_np, cv2.COLOR_RGBA2BGRA)
   scale_factor = 0.35  # Adjust scale as desired
   new_width = int(arrow_np.shape[1] * scale_factor)
   new_height = int(arrow_np.shape[0] * scale_factor)
   arrow_np = cv2.resize(arrow_np, (new_width, new_height))
   if arrow_np.shape[2] == 3:
       arrow_np = cv2.cvtColor(arrow_np, cv2.COLOR_BGR2BGRA)
   b, g, r, a = cv2.split(arrow_np)


   def update_video():
       global prev_left_line, prev_right_line, prev_center_line, rotate_variable1, rotate_variable2

       ret, frame = cap.read()
       if ret:
           # Raw Video Display (Top Left)
           frame_top = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           img_top = Image.fromarray(frame_top).resize((300, 300))
           imgtk_top = ImageTk.PhotoImage(image=img_top)
           video_label_top_left.imgtk = imgtk_top
           video_label_top_left.config(image=imgtk_top)

           # Lane detection
           left_line, right_line, roi_vertices, valid = detect_lanes(frame)

           # Apply Kalman filter if lane detected, or fallback to previous detection
           if left_line is not None:
               left_line = apply_kalman_filter(left_kalman, left_line)
               prev_left_line = left_line
               rotate_variable1 = 0
           else:
               left_line = prev_left_line
               rotate_variable1 = 1

           if right_line is not None:
               right_line = apply_kalman_filter(right_kalman, right_line)
               prev_right_line = right_line
               rotate_variable2 = 0
           else:
               right_line = prev_right_line
               rotate_variable2 = 1

           center_line = detect_center_line(left_line, right_line)

           # Overlay lanes and the image
           output_frame = overlay_lanes(frame, left_line, right_line, roi_vertices, center_line)

           # Rotate overlay if no lanes detected
           # Bottom Video Display
           frame_bottom_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
           img_bottom = Image.fromarray(frame_bottom_rgb).resize((300, 300))
           imgtk_bottom = ImageTk.PhotoImage(image=img_bottom)
           video_label_bottom_left.imgtk = imgtk_bottom
           video_label_bottom_left.config(image=imgtk_bottom)

           video_label_top_left.after(30, update_video)  # Call update_video again after 30ms
       else:
           cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
           update_video()

   root.mainloop()


# ----------------==== LOGIN SYSTEM ----------------====
login_window = tk.Tk()
login_window.title("Login - Let's Get It Started!")
login_window.geometry("300x250")
login_window.configure(bg="#2C2F33")


tk.Label(login_window, text="Username:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(20, 5))
username_entry = tk.Entry(login_window, bg="#99AAB5", fg="black", font=("Arial", 12))
username_entry.pack(pady=5)


tk.Label(login_window, text="Password:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(10, 5))
password_entry = tk.Entry(login_window, show="*", bg="#99AAB5", fg="black", font=("Arial", 12))
password_entry.pack(pady=5)


login_message = tk.Label(login_window, text="", fg="red", bg="#2C2F33", font=("Arial", 10))
login_message.pack(pady=(10, 5))


def attempt_login():
   username = username_entry.get()
   password = password_entry.get()
   if username in accounts and accounts[username] == password:
       login_message.config(text="Login successful! Lit!", fg="green")
       login_window.destroy()
       start_main_app()
   else:
       login_message.config(text="Invalid creds, try again!", fg="red")


tk.Button(login_window, text="Login", command=attempt_login, bg="#7289DA", fg="white", font=("Arial", 12)).pack(pady=(10, 5))


def open_registration():
   reg_window = tk.Toplevel(login_window)
   reg_window.title("Create Account")
   reg_window.geometry("300x250")
   reg_window.configure(bg="#2C2F33")


   tk.Label(reg_window, text="New Username:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(20, 5))
   reg_username_entry = tk.Entry(reg_window, bg="#99AAB5", fg="black", font=("Arial", 12))
   reg_username_entry.pack(pady=5)


   tk.Label(reg_window, text="New Password:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(10, 5))
   reg_password_entry = tk.Entry(reg_window, show="*", bg="#99AAB5", fg="black", font=("Arial", 12))
   reg_password_entry.pack(pady=5)


   reg_message = tk.Label(reg_window, text="", fg="green", bg="#2C2F33", font=("Arial", 10))
   reg_message.pack(pady=(10, 5))


   def register():
       new_username = reg_username_entry.get()
       new_password = reg_password_entry.get()
       if new_username in accounts:
           reg_message.config(text="Username already exists!", fg="red")
       else:
           accounts[new_username] = new_password
           save_accounts(accounts)
           reg_message.config(text="Account created! Go login!", fg="green")
           reg_window.after(1000, reg_window.destroy)


   tk.Button(reg_window, text="Register", command=register, bg="#7289DA", fg="white", font=("Arial", 12)).pack(pady=(10, 5))


tk.Button(login_window, text="Create Account", command=open_registration, bg="#99AAB5", fg="black", font=("Arial", 12)).pack(pady=(10, 5))


login_window.mainloop()
