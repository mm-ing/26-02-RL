Task: Update Home.py to match the user's requirements and the layout in screenshot.png.

Requirements:
- Change greeted name from "Manfred" to "Johannes" (window title and greeting).
- Generate 10 images showing nature in a local folder "images/" in the same directory. Display a random image from this folder above the greeting on each run. Support common formats (png, jpg). Use Pillow (PIL) if available, fallback to tkinter.PhotoImage for PNG.
- Show a randomly selected motivating sentence under the greeting on each run. Provide at least 10 English sentences in a list and pick one each start.
- Under the date/time label, show current temperature in °C for Dresden and under that the current wind speed in km/h for Dresden. Use Open-Meteo API (no API key) with coordinates latitude=51.0504, longitude=13.7373 and current_weather=true, temperature_unit=celsius, windspeed_unit=kmh. Refresh weather every 10 minutes; update time every second.
- Output language remains English.
- Match the visual order and styling from screenshot.png: image at top centered, "Hello Johannes!" bold, motivational sentence in green and slightly larger, date/time centered, then temperature line then wind speed line. Keep window background light blue.
- Add minimal dependencies and clear error handling for network failures (show "N/A" or a message if weather cannot be fetched).
- Keep changes contained to Home.py.

Implementation notes for the code:
- Use tkinter for GUI, datetime for time, requests for API calls (or urllib if requests not available).
- Use root.after to schedule update_time (1s) and update_weather (600000 ms).
- Load images from "./images" and random.choice; resize proportionally to fit width ~500px.
- Keep code concise and comment key changes.

Return: Only the modified Home.py content ready to place at the filepath above.