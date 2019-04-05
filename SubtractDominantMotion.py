import numpy as np
import cv2
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_opening, binary_closing

from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.ones(image1.shape, dtype=bool)

    region = np.zeros(image1.shape, dtype=np.float32)
    region[10:150, 60:220] = 1.0

    threshold = 0.15
    M = LucasKanadeAffine(image1, image2)
    # M = InverseCompositionAffine(image1, image2)

    warped_img = cv2.warpAffine(image1, M, (image1.shape[1], image1.shape[0]))
    diff = np.absolute(warped_img - image1) * region
    mask = diff > threshold
    mask = binary_dilation(mask, np.ones((11, 11)))

    return mask
