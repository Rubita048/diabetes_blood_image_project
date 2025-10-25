import cv2
import numpy as np
from skimage.color import rgb2gray

def preprocess_image(img):
    gray = rgb2gray(img)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    _, mask = cv2.threshold((gray*255).astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    return gray, mask
