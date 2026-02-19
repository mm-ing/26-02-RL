# ...existing code...
import tkinter as tk
from datetime import datetime
import os, random, threading, time, json
try:
    import requests
except Exception:
    requests = None
try:
    from PIL import Image, ImageTk, ImageDraw
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --- Configuration / changes per requirements ---
user_name = "Johannes"
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
IMAGE_DISPLAY_WIDTH = 500
WEATHER_URL = ("https://api.open-meteo.com/v1/forecast"
               "?latitude=51.0504&longitude=13.7373&current_weather=true"
               "&temperature_unit=celsius&windspeed_unit=kmh")
WEATHER_REFRESH_MS = 10 * 60 * 1000  # 10 minutes
TIME_REFRESH_MS = 1000  # 1 second

# Motivational sentences (>=10)
MOTIVATIONS = [
    "Find a reason to celebrate today.",
    "Small progress is still progress.",
    "You are capable of amazing things.",
    "Every step forward counts.",
    "Make today ridiculously amazing.",
    "Your effort matters more than perfection.",
    "Take one step, then another.",
    "Create something that makes you proud.",
    "Be kind to yourself and keep going.",
    "Embrace challenges—they help you grow."
]

# --- Helper: generate simple nature images if not enough exist ---
def ensure_images(count=10):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    existing = [f for f in os.listdir(IMAGES_DIR)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(existing) >= count:
        return
    if not PIL_AVAILABLE:
        return  # cannot generate without PIL
    # create simple gradient "nature" images
    for i in range(len(existing), count):
        img = Image.new("RGB", (1200, 720))
        draw = ImageDraw.Draw(img)
        # sky gradient
        for y in range(img.height):
            r = int(135 + (120 * y / img.height))
            g = int(206 + (20 * y / img.height))
            b = int(235 + (20 * y / img.height))
            draw.line([(0, y), (img.width, y)], fill=(r, g, b))
        # sun
        draw.ellipse((900, 80, 1040, 220), fill=(255, 245, 125))
        # distant hills
        draw.polygon([(0,480),(250,340),(500,460),(800,360),(1200,480),(1200,720),(0,720)], fill=(34,139,34))
        # water / foreground
        draw.rectangle((0,500,1200,720), fill=(25,25,112))
        fname = os.path.join(IMAGES_DIR, f"nature_{i+1:02d}.jpg")
        img.save(fname, quality=85)

# --- Image loading and selection ---
def load_random_image():
    files = []
    if os.path.isdir(IMAGES_DIR):
        files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not files:
        # fallback to screenshot.png in same dir if present
        fallback = os.path.join(os.path.dirname(__file__), "screenshot.png")
        if os.path.exists(fallback):
            files = [fallback]
    if not files:
        return None  # no image available
    choice = random.choice(files)
    try:
        if PIL_AVAILABLE:
            img = Image.open(choice)
            w, h = img.size
            if w > IMAGE_DISPLAY_WIDTH:
                new_h = int(h * IMAGE_DISPLAY_WIDTH / w)
                img = img.resize((IMAGE_DISPLAY_WIDTH, new_h), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        else:
            # fallback: tkinter.PhotoImage supports PNG; try only if PNG
            if choice.lower().endswith(".png"):
                return tk.PhotoImage(file=choice)
            return None
    except Exception:
        return None

# --- Weather fetching (uses requests if available, else urllib) ---
def fetch_weather():
    try:
        if requests:
            r = requests.get(WEATHER_URL, timeout=8)
            r.raise_for_status()
            data = r.json()
        else:
            from urllib.request import urlopen
            with urlopen(WEATHER_URL, timeout=8) as u:
                data = json.load(u)
        cw = data.get("current_weather", {})
        temp = cw.get("temperature")
        wind = cw.get("windspeed")
        return (f"{temp:.1f}°C" if temp is not None else "N/A",
                f"{wind:.1f} km/h" if wind is not None else "N/A")
    except Exception:
        return ("N/A", "N/A")

# --- GUI setup ---
root = tk.Tk()
root.title(f"Welcome, {user_name}!")
root.configure(bg="#e6f2ff")
root.geometry("760x560")

# Ensure images exist (may be slow - run quickly)
ensure_images(10)

# Image label
img_tk = load_random_image()
image_label = tk.Label(root, bg="#e6f2ff")
if img_tk:
    image_label.config(image=img_tk)
    image_label.image = img_tk
image_label.pack(pady=(20, 10))

# Greeting and motivational sentence
salutation = tk.Label(root, text=f"Hello {user_name}!", font=("Arial", 20, "bold"), bg="#e6f2ff")
salutation.pack(pady=(5, 2))

motivation = tk.Label(root, text=random.choice(MOTIVATIONS),
                      font=("Arial", 16), fg="#006400", bg="#e6f2ff")
motivation.pack(pady=(2, 12))

# Time and weather labels
time_label = tk.Label(root, font=("Arial", 12), bg="#e6f2ff")
time_label.pack(pady=2)

temp_label = tk.Label(root, text="Current temperature in Dresden: N/A", font=("Arial", 12), bg="#e6f2ff")
temp_label.pack(pady=2)

wind_label = tk.Label(root, text="Wind speed: N/A", font=("Arial", 12), bg="#e6f2ff")
wind_label.pack(pady=(2, 12))

# --- Updaters ---
def update_time():
    now = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    time_label.config(text=now)
    root.after(TIME_REFRESH_MS, update_time)

def update_weather():
    def worker():
        temp, wind = fetch_weather()
        # update labels in main thread
        def apply():
            temp_label.config(text=f"Current temperature in Dresden: {temp}")
            wind_label.config(text=f"Wind speed: {wind}")
        root.after(0, apply)
    threading.Thread(target=worker, daemon=True).start()
    root.after(WEATHER_REFRESH_MS, update_weather)

# Start periodic updates
update_time()
update_weather()

# Run GUI
root.mainloop()
# ...existing code...