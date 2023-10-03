import time
import pyautogui 
import pydirectinput
import cv2 as cv
import random
import numpy as np

haystack_img = cv.imread('Tempmem/farm.PNG', cv.IMREAD_UNCHANGED)
needle_img = cv.imread('memories/Tree/Tree1.PNG', cv.IMREAD_UNCHANGED)

result = cv.matchTemplate(haystack_img, needle_img, cv.TM_CCOEFF_NORMED)
#Debug code
#cv.imshow('Result', result)
#cv.waitKey()

#min max of brightest and darkest pixel, get best match position
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

print('%s' %str(max_loc))
print('%s' %str(max_val))