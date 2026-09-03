import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

def add_gaussian_noise(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Add Gaussian noise to the image in monochrome.

    Args:
        image: Input image array
        mean: Mean of the Gaussian distribution
        stddev: Standard deviation of the Gaussian distribution

    Returns:
        Noisy image array
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gaussian_noise = np.random.normal(0, noise_ratio*255, image.shape)
    noisy_image = image + gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_speckle_noise(image: np.ndarray, noise_ratio: float) -> np.ndarray:
    """
    Add Speckle noise to the image in monochrome.
    
    Args:
        image: Input image array
        noise_ratio: Ratio of noise to add
        
    Returns:
        Noisy image array
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    noise = np.random.normal(0, noise_ratio, image.shape)
    noisy_image = image * (1 + noise)
    return np.clip(noisy_image, 0, 255)

def get_noise_ratios(dataset_type: str) -> List[float]:
    """
    Get appropriate noise ratios based on dataset type.
    
    Args:
        dataset_type: Type of dataset ('train' or 'test')
        
    Returns:
        List of noise ratios
    """
    if dataset_type.lower() == 'speckle':
        return [0.05, 0.1, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8, 0.9]  # More aggressive noise for training
    elif dataset_type.lower() == 'gaussian':
        return [0.05, 0.1, 0.125, 0.135, 0.15, 0.165, 0.175, 0.185, 0.195, 0.2, 0.225, 0.25, 0.3, 0.35, 0.375]  # More aggressive noise for training
    else:
        raise ValueError("Dataset type must be either 'train' or 'test'")

def process_images(input_folder: str, output_folder: str, dataset_type: str) -> None:
    """
    Process all images in the input folder and save degraded versions.
    
    Args:
        input_folder: Path to input images
        output_folder: Path to save processed images
        dataset_type: Type of dataset ('train' or 'test')
    """
    os.makedirs(output_folder, exist_ok=True)
    noise_ratios = get_noise_ratios(dataset_type)
    
    input_path = Path(input_folder)
    image_files = list(input_path.glob('*.png'))
    
    if not image_files:
        print(f"No PNG files found in {input_folder}")
        return
        
    pbar_images = tqdm(image_files, desc="Processing images")
    
    for img_path in pbar_images:
        pbar_images.set_description(f"Processing {img_path.name}")
        
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not read image: {img_path}")
            continue
            
        base_name = img_path.stem

        if dataset_type == "speckle":        
            for ratio in tqdm(noise_ratios, desc="Speckle noise", leave=False):
                noisy_image = add_speckle_noise(image, ratio)
                output_name = f"{base_name}_speckle_{ratio:.3f}.png"
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, noisy_image)
        elif dataset_type == "gaussian":
            for ratio in tqdm(noise_ratios, desc="Gaussian noise", leave=False):
                noisy_image = add_gaussian_noise(image, ratio)
                output_name = f"{base_name}_gaussian_{ratio:.3f}.png"
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, noisy_image)
        else:
            pass

def parse_arguments() -> Tuple[str, str, str]:
    """
    Parse command line arguments.
    
    Returns:
        Tuple of (input_folder, output_folder, dataset_type)
    """
    parser = argparse.ArgumentParser(description='Process images with speckle noise')
    parser.add_argument('--input', type=str, required=True, help='Input folder containing images')
    parser.add_argument('--output', type=str, required=True, help='Output folder for processed images')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset type (train or test)')
    
    args = parser.parse_args()
    return args.input, args.output, args.dataset

if __name__ == "__main__":
    input_folder, output_folder, dataset_type = parse_arguments()
    process_images(input_folder, output_folder, dataset_type)