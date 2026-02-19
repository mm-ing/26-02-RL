# ...existing code...
import tkinter as tk
from datetime import datetime
import random

# Optionale Pillow-Nutzung für in-memory Bilder
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

city = "Reutlingen"
user_name = "Tobias"

greeting_text = "Have a wonderful day ahead!"
greeting_text1 = "Take care today!"
greeting_text2 = "Enjoy the SUN today!"
greeting_text3 = "Make today count!"
greeting_text4 = "Stay positive and strong!"

greetings = [greeting_text, greeting_text1, greeting_text2, greeting_text3, greeting_text4]

def update_time():
    now = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    time_label.config(text=now)
    root.after(1000, update_time)

root = tk.Tk()
root.title(f"Welcome, {user_name}! — {city}")
root.geometry("600x300")
bg_color = "#e6f2ff"
root.configure(bg=bg_color)

# Erzeuge fünf einfache Bilder (farbige Platzhalter)
colors = ["#ff9999", "#99ff99", "#9999ff", "#ffd699", "#c299ff"]
image_widgets = []
selected_index = random.randrange(len(colors))

if PIL_AVAILABLE:
    images_tk = []
    for c in colors:
        img = Image.new("RGB", (140, 90), c)
        images_tk.append(ImageTk.PhotoImage(img))
    image_label = tk.Label(root, image=images_tk[selected_index], bg=bg_color)
    # Referenz halten, damit GC die PhotoImage nicht entfernt
    image_label.image = images_tk[selected_index]
else:
    # Fallback: Canvas mit farbigem Rechteck
    canvas = tk.Canvas(root, width=140, height=90, bg=bg_color, highlightthickness=0)
    canvas.create_rectangle(0, 0, 140, 90, fill=colors[selected_index], outline=colors[selected_index])
    image_label = canvas

# Packe Bild über dem Namen
image_label.pack(pady=(12, 6))

salutation = tk.Label(root, text=f"Hello {user_name}!", font=("Arial", 16, "bold"), bg=bg_color)
salutation.pack(pady=4)

# Wähle zufällig einen der fünf Begrüßungstexte aus und zeige ihn an
selected_greeting = random.choice(greetings)
greeting = tk.Label(root, text=selected_greeting, font=("Arial", 14, "bold"), bg=bg_color, fg="#006400")
greeting.pack(pady=6)

time_label = tk.Label(root, font=("Arial", 12), bg=bg_color)
time_label.pack(pady=8)
update_time()

root.mainloop()
# ...existing code...