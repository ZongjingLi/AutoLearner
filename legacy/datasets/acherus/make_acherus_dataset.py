import time
#time.sleep(2) # this is the wait time of the actual config

import pyautogui # with mouse keys on
# mouse keys is an accessibility feature on mac that controls the mouse with the keyboard
print("[ALERT]: Make sure mouse keys is on! (press option 5 times if shortcut is enabled)")

from pynput.keyboard import Key, Controller

keyboard = Controller()

# Press and release space

def look_up(second = 2):
    keyboard.press("8")
    time.sleep(second)
    keyboard.release("8")

def look_down(second = 2):
    keyboard.press("2")
    time.sleep(second)
    keyboard.release("2")

def look_left(second = 2):
    keyboard.press("4")
    time.sleep(2)
    keyboard.release("4")

def look_right(second = 2):
    keyboard.press("6")
    time.sleep(2)
    keyboard.release("6")

def open_backpack():pyautogui.press("e")

def switch_to(slot):pyautogui.press(str(slot))

def close_packpack():pyautogui.press("esc")

def left_click():pyautogui.leftClick()

def right_click():pyautogui.rightClick()

import numpy as np
from PIL import ImageGrab


import matplotlib.pyplot as plt


if __name__ == "__main__":
    for i in range(200):
        off_set = 60
        curr = time.time()
        #pyautogui.screenshot("{}.jpg".format(i))
        screen = np.array(ImageGrab.grab(bbox=(20,10+off_set,844,484+off_set)))
        #plt.imshow(screen)
        plt.imsave("/Users/melkor/Documents/datasets/acherus/train/{}.jpg".format(200 + i),screen)
        now  = time.time()
        print(i,now - curr)
        #pyautogui.press("w",100)