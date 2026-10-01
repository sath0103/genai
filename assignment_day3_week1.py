import pyautogui
import time
import re
import pyperclip
from pathlib import Path
import subprocess
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 1
pyautogui.hotkey("win", "r")  # open the Run dialog
time.sleep(1)
pyautogui.write("chrome")  # type "notepad"
pyautogui.press("enter")
pyautogui.hotkey("ctrl", "t")  # open a new tab
time.sleep(1)
pyautogui.write("https://www.google.com/search?q=petrol+diesel+price+chart+last+3+days+in+india&rlz=1C1AJCO_enIN1192IN1192&oq=petrol+disel+price+chat+last+3+d&gs_lcrp=EgZjaHJvbWUqCQgCECEYChigATIGCAAQRRg5MgkIARAhGAoYoAEyCQgCECEYChigATIJCAMQIRgKGKABMgcIBBAhGI8CMgcIBRAhGI8CMgcIBhAhGI8C0gEJMjMyOTZqMGo3qAIIsAIB8QVZ_ajH87CK7PEFWf2ox_Owiuw&sourceid=chrome&source=chrome.ob&ie=UTF-8&zx=1790856326394")
pyautogui.press("enter")  # type the URL
time.sleep(8)  # wait for results
pyautogui.hotkey("ctrl", "a")
pyautogui.hotkey("ctrl", "c")
page_text = pyperclip.paste()  # get the copied text from the clipboard
city_pattern = re.compile(
    r"\b(Delhi|Mumbai|Kolkata|Chennai)\b",
    re.IGNORECASE,
)
price_pattern = re.compile(r"₹\s*([\d,]+(?:\.\d{1,2})?)")

city_matches = list(city_pattern.finditer(page_text))
rows = []
seen_cities = set()

for index, city_match in enumerate(city_matches):
    city = city_match.group(1).title()
    if city in seen_cities:
        continue

    end = (
        city_matches[index + 1].start()
        if index + 1 < len(city_matches)
        else len(page_text)
    )
    city_text = page_text[city_match.end():end]
    prices = price_pattern.findall(city_text)

    if len(prices) >= 2:
        rows.append(
            f"{city}\t₹{prices[0]}\t₹{prices[1]}"
        )
        seen_cities.add(city)

lines = ["City\tPetrol (₹/L)\tDiesel (₹/L)", *rows]
output_path = Path(r"F:\GenAI\metro_fuel_prices.txt")
output_path.write_text("\n".join(lines), encoding="utf-8")

print(f"Saved {len(rows)} city rows to {output_path}")
if not rows:
    print("Clipboard text preview:", repr(page_text[:2000]))