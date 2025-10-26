import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray

def texture_features(image):
    gray = (rgb2gray(image) * 255).astype('uint8')
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return contrast, energy
