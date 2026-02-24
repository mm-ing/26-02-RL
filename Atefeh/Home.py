# python -m venv .venv

import tkinter as tk
from datetime import datetime

city = "Aachen"
user_name = "Atefeh"
greeting_text = "Have a wonderful day ahead!"

def update_time():
    now = datetime.now().strftime("%A, %d %B %Y %H:%M:%S")
    time_label.config(text=now)
    root.after(1000, update_time)

root = tk.Tk()
root.title(f"Welcome, {user_name}!")
root.geometry("600x250")
root.configure(bg="#e6f2ff")


salutation = tk.Label(root, text=f"Hello {user_name}!", font=("Arial", 16, "bold"), bg="#e6f2ff")
salutation.pack(pady=10)
greeting = tk.Label(root, text=greeting_text, font=("Arial", 16, "bold"), bg="#e6f2ff", fg="#006400")
greeting.pack(pady=10)

time_label = tk.Label(root, font=("Arial", 12), bg="#e6f2ff")
time_label.pack(pady=5)
update_time()

root.mainloop()