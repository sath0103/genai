import pyautogui
import subprocess
import time

# Mouse control
pyautogui.moveTo(100, 100, duration=1)
pyautogui.click()
pyautogui.rightClick()
pyautogui.sleep(1)

pyautogui.hotkey("ctrl", "a")  # select the entire file
pyautogui.hotkey("ctrl", "c")  # copy it
time.sleep(0.5)

pyautogui.hotkey("ctrl", "n")  # open a blank Notepad tab
time.sleep(1)
pyautogui.hotkey("ctrl", "s")
pyautogui.hotkey("ctrl", "v")  # paste the copied file contents
time.sleep(1)
pyautogui.write(r"F:\GenAI\copied_text.txt", interval=0.05)
pyautogui.press("enter")


# Keyboard control
# Screenshot
# Window control