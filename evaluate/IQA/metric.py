#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-08-13 21:37:58

'''
Calculate LPIPS, PSNR, SSIM and other metrics.
'''

import os, sys, math
import lpips
import pyiqa
import pickle
import argparse
import numpy as np
from scipy import linalg
from pathlib import Path
from loguru import logger as base_logger
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
import torch
import torch.nn as nn
from pytorch_fid import fid_score
import tempfile
import shutil
from skimage.metrics import structural_similarity as ssim


# Fix OpenMP duplicate library issue
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_im_tensor(im_path):
    """
    Load image and normalize to [0, 1]
    """
    img = Image.open(im_path).convert('RGB')
    img = transforms.ToTensor()(img).unsqueeze(0).cuda()
    return img

def calculate_psnr(img1, img2):
    # Ensure input range is [0,1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Calculate SSIM (Structural Similarity Index) between two images.
    """
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Convert to grayscale for SSIM calculation
    img1_gray = np.dot(img1[..., :3], [0.299, 0.587, 0.114])
    img2_gray = np.dot(img2[..., :3], [0.299, 0.587, 0.114])
    
    return ssim(img1_gray, img2_gray, data_range=1.0)


def prepare_images_for_fid(src_dir, dest_dir):
    """Prepare images for FID calculation by resizing to 299x299"""
    os.makedirs(dest_dir, exist_ok=True)
    paths = list(Path(src_dir).glob('*.png'))
    transform = transforms.Compose([
        transforms.Resize((299, 299), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])
    
    for path in paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(dest_dir, path.name))

def calculate_single_image_fid(gt_path, sr_path):
    """Calculate FID score for a single image pair"""
    with tempfile.TemporaryDirectory() as gt_temp_dir, tempfile.TemporaryDirectory() as sr_temp_dir:
        # Create temporary subdirectories for single images
        gt_img_dir = os.path.join(gt_temp_dir, 'img')
        sr_img_dir = os.path.join(sr_temp_dir, 'img')
        os.makedirs(gt_img_dir)
        os.makedirs(sr_img_dir)
        
        # Copy and resize single images
        transform = transforms.Compose([
            transforms.Resize((299, 299), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        
        for src_path, dst_dir in [(gt_path, gt_img_dir), (sr_path, sr_img_dir)]:
            img = Image.open(src_path).convert('RGB')
            img = transform(img)
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(dst_dir, src_path.name))
            
            # Create copies to have enough samples for FID calculation
            for i in range(9):  # Create 9 copies to have 10 samples total
                img.save(os.path.join(dst_dir, f'copy_{i}_{src_path.name}'))
        
        try:
            fid = fid_score.calculate_fid_given_paths(
                [gt_img_dir, sr_img_dir],
                batch_size=1,
                device='cuda',
                dims=2048,
                num_workers=4
            )
        except Exception as e:
            print(f"Error calculating single image FID: {str(e)}")
            fid = float('nan')
            
    return fid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="", help="Path to save the HQ images")
    parser.add_argument("--sr_dir", type=str, default="", help="Path to save the SR images")
    args = parser.parse_args()

    # setting logger
    log_path = str(Path(args.sr_dir).parent / 'metrics.log')
    logger = base_logger
    logger.remove()
    logger.add(log_path, format="{time:YYYY-MM-DD(HH:mm:ss)}: {message}", mode='w', level='INFO')
    logger.add(sys.stderr, format="{message}", level='INFO')

    for key in vars(args):
        value = getattr(args, key)
        logger.info(f'{key}: {value}')

    # Initialize metrics
    lpips_metric_vgg = lpips.LPIPS(net='vgg').cuda()
    lpips_metric_alex = lpips.LPIPS(net='alex').cuda()
    clipiqa_metric = pyiqa.create_metric('clipiqa', device=torch.device('cuda'))
    musiq_metric = pyiqa.create_metric('musiq', device=torch.device('cuda'))

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Get all image paths
    sr_paths = list(Path(args.sr_dir).glob('*.png'))
    logger.info(f"Total images: {len(sr_paths)}")

    # Initialize metrics storage
    total_lpips_vgg = 0
    total_lpips_alex = 0
    total_clipiqa = 0
    total_musiq = 0
    total_psnr = 0
    total_fid = 0
    total_ssim = 0

    # Create list to store results
    results = []

    # Calculate overall FID score
    with tempfile.TemporaryDirectory() as gt_temp_dir, tempfile.TemporaryDirectory() as sr_temp_dir:
        # Prepare images for FID calculation
        prepare_images_for_fid(args.gt_dir, gt_temp_dir)
        prepare_images_for_fid(args.sr_dir, sr_temp_dir)
        
        # Calculate FID score
        try:
            overall_fid = fid_score.calculate_fid_given_paths(
                [gt_temp_dir, sr_temp_dir],
                batch_size=1,
                device='cuda',
                dims=2048,
                num_workers=4
            )
        except Exception as e:
            logger.error(f"Error calculating overall FID: {str(e)}")
            overall_fid = float('nan')

    # Log header for per-image metrics
    logger.info("\nPer-image metrics:")
    logger.info("Image Name | PSNR | SSIM | LPIPS-VGG | LPIPS-Alex | CLIPIQA | MUSIQ | FID")
    logger.info("-" * 90)

    for sr_path in sr_paths:
        gt_path = Path(args.gt_dir) / sr_path.name
        
        # Load and preprocess images
        sr_img = Image.open(sr_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        sr_tensor = preprocess(sr_img).unsqueeze(0).cuda()
        gt_tensor = preprocess(gt_img).unsqueeze(0).cuda()
        
        # For CLIPIQA and MUSIQ
        sr_tensor_norm = transforms.ToTensor()(sr_img).unsqueeze(0).cuda()
        gt_tensor_norm = transforms.ToTensor()(gt_img).unsqueeze(0).cuda()

        with torch.no_grad():
            # Calculate metrics
            lpips_vgg = float(lpips_metric_vgg(gt_tensor, sr_tensor))
            lpips_alex = float(lpips_metric_alex(gt_tensor, sr_tensor))
            clipiqa = float(clipiqa_metric(sr_tensor_norm))
            musiq = float(musiq_metric(sr_tensor_norm))
            psnr = float(calculate_psnr(gt_tensor_norm, sr_tensor_norm))
            ssim = float(calculate_ssim(gt_tensor_norm, sr_tensor_norm))


            
            # Calculate single image FID
            fid = calculate_single_image_fid(gt_path, sr_path)

            # Log metrics for this image
            logger.info(f"{sr_path.name} | {psnr:6.2f} | {ssim:6.2f} | {lpips_vgg:6.4f} | {lpips_alex:6.4f} | {clipiqa:6.4f} | {musiq:5.2f} | {fid:6.2f}")

            # Add results to list
            results.append({
                'Image Name': sr_path.name,
                'PSNR': psnr,
                'SSIM': ssim,
                'LPIPS-VGG': lpips_vgg,
                'LPIPS-Alex': lpips_alex,
                'CLIPIQA': clipiqa,
                'MUSIQ': musiq,
                'FID': fid
            })

            # Add to totals
            total_lpips_vgg += lpips_vgg
            total_lpips_alex += lpips_alex
            total_clipiqa += clipiqa
            total_musiq += musiq
            total_psnr += psnr
            total_fid += fid
            total_ssim += ssim

    num_images = len(sr_paths)
    # Calculate and log averages
    logger.info("\nAverage metrics:")
    logger.info(f"Overall FID: {overall_fid:6.2f}")
    logger.info(f"Average per-image FID: {total_fid/num_images:6.2f}")
    logger.info(f"PSNR: {total_psnr/num_images:6.2f}")
    logger.info(f"LPIPS-VGG: {total_lpips_vgg/num_images:6.4f}")
    logger.info(f"LPIPS-Alex: {total_lpips_alex/num_images:6.4f}")
    logger.info(f"CLIPIQA: {total_clipiqa/num_images:6.4f}")
    logger.info(f"MUSIQ: {total_musiq/num_images:5.2f}")
    logger.info(f"SSIM: {total_ssim/num_images:5.2f}")

    # Add average row
    results.append({
        'Image Name': 'Average',
        'FID': overall_fid,
        'PSNR': total_psnr/num_images,
        'SSIM': total_ssim/num_images,
        'LPIPS-VGG': total_lpips_vgg/num_images,
        'LPIPS-Alex': total_lpips_alex/num_images,
        'CLIPIQA': total_clipiqa/num_images,
        'MUSIQ': total_musiq/num_images
    })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results)
    excel_path = str(Path(args.sr_dir).parent / f'{Path(args.sr_dir).name}_iqa.xlsx')
    df.to_excel(excel_path, index=False)
    logger.info(f"\nMetrics saved to: {excel_path}")

if __name__ == "__main__":
    main()