import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from .preprocessing import preprocess_image
from skimage.filters import sobel, prewitt, roberts
from skimage.feature import local_binary_pattern

def extract_features(img):
    gray, mask = preprocess_image(img)
    lbl = label(mask)
    regions = regionprops(lbl)

    if len(regions) == 0:
        return None

    r = regions[0]
    if r.area < 10:
        return None

    # Temel morfolojik özellikler
    solidity = r.solidity
    extent = r.extent
    aspect_ratio = r.major_axis_length / (r.minor_axis_length + 1e-6)

    # Renk özellikleri
    mean_r = img[:,:,0].mean()
    mean_g = img[:,:,1].mean()
    mean_b = img[:,:,2].mean()

    # Tekstür özellikleri
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix((gray*255).astype('uint8'), [1], angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()

    # Kenar tabanlı özellik (örneğin hücre zar belirginliği)
    edges = sobel(gray)
    edge_density = edges.mean()

    # LBP (Local Binary Pattern) – doku özellikleri
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()

    return {
        "area": r.area,
        "perimeter": r.perimeter,
        "solidity": solidity,
        "extent": extent,
        "aspect_ratio": aspect_ratio,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy
    }
