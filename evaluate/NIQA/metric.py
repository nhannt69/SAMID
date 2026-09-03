import numpy as np
from skimage.metrics import structural_similarity as ssim

def calculate_image_quality_score(noisy_image, denoised_image):
    """
    Calculate the image quality score e between a noisy image and a denoised image.
    
    Args:
        noisy_image (numpy.ndarray): The noisy image (I)
        denoised_image (numpy.ndarray): The denoised image (Î_h)
        
    Returns:
        float: The image quality score e
    """
    # Ensure images are numpy arrays and have the same shape
    noisy_image = np.array(noisy_image, dtype=np.float32)
    denoised_image = np.array(denoised_image, dtype=np.float32)
    
    if noisy_image.shape != denoised_image.shape:
        raise ValueError("Noisy and denoised images must have the same shape")

    # Step 1: Compute MNI (Mean Noise Image)
    M_h = noisy_image - denoised_image
    
    
    # Step 2: Compute structure similarity map N between noisy image and MNI using SSIM (Eq. 3)
    # Use full=True to get the map, not just the score
    _, N = ssim(noisy_image, M_h, full=True, channel_axis=-1 if noisy_image.ndim == 3 else None, data_range=1.0)

    
    # Step 3: Compute structure similarity map P between noisy image and denoised image using SSIM (Eq. 4)
    _, P = ssim(noisy_image, denoised_image, full=True, channel_axis=-1 if noisy_image.ndim == 3 else None, data_range=1.0)

    
    # Step 4: Compute image quality score e as the linear correlation coefficient of N and P
    # Flatten the maps to 1D arrays for correlation
    N_flat = N.flatten()
    P_flat = P.flatten()
    
    # Debug: Print variances

    
    # Check for variance to avoid undefined correlation
    if np.var(N_flat) == 0 or np.var(P_flat) == 0:
        print("Warning: One of the SSIM maps has zero variance, correlation is undefined. Returning 0.")
        return 0.0
    
    # Compute Pearson correlation coefficient
    e = np.corrcoef(N_flat, P_flat)[0, 1]
    
    # If correlation is nan (due to numerical issues), return 0
    if np.isnan(e):
        print("Warning: Correlation resulted in NaN. Returning 0.")
        return 0.0
    
    return np.abs(e)

import cv2
if __name__ == "__main__":
    # Load images in grayscale
    img1 = cv2.imread('images/unsupervised/Bot_3_layer_6.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/visualization_v5/SAMTest18/Bot_3_layer_6.png', cv2.IMREAD_GRAYSCALE)
    

    score = calculate_image_quality_score(img1, img2)
    print(f"Image quality score: {score}")