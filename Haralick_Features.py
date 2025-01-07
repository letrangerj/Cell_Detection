import mahotas as mh
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from pathlib import Path
import cv2

'''
This script is used to extract Haralick features from the cell region of an image.
The Haralick features are texture features that describe the distribution of pixel intensities within an image.
These features are used to quantify the texture of the cell region and can be used to classify cells based on their texture.
'''




def create_mask_from_contour(contour, image_shape):
    """
    Create binary mask from contour
    
    Parameters:
    -----------
    contour : numpy.ndarray
        OpenCV contour array
    image_shape : tuple
        Shape of the original image (height, width)
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where cell region is 1 and background is 0
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], 0, 1, -1)
    return mask



def analyze_cell_texture(image, contour):
    """
    Analyze Haralick features within a cell contour
    
    Parameters:
    -----------
    image : numpy.ndarray
        Original grayscale image
    contour : numpy.ndarray
        OpenCV contour array
        
    Returns:
    --------
    dict
        Haralick features for the cell region
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask from contour
    mask = create_mask_from_contour(contour, image.shape)
    
    # Normalize to uint8
    image_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply mask to image
    masked_image = image_normalized.copy()
    masked_image[~mask.astype(bool)] = 0
    
    # Get bounding box of contour for efficient processing
    x, y, w, h = cv2.boundingRect(contour)
    roi_image = masked_image[y:y+h, x:x+w]
    
    # Compute Haralick features
    haralick_features = mh.features.haralick(roi_image, return_mean=True, 
                                           ignore_zeros=False) 
    # set ignore_zeros to False to include the background in the calculation (otherwise error might occur)
    
    # Feature names
    feature_names = [
        'angular_second_moment',
        'contrast',
        'correlation',
        'sum_of_squares',
        'inverse_diff_moment',
        'sum_avg',
        'sum_var',
        'sum_entropy',
        'entropy',
        'difference_var',
        'difference_entropy',
        'info_measure_corr_1',
        'info_measure_corr_2'
    ]
    
    return dict(zip(feature_names, haralick_features)), mask

