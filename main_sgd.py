"""
Stochastic Gradient Descent (SGD) optimization for Computer Generated Holography.
This script implements the SGD method for phase retrieval.
"""

import os
import torch
import numpy as np
import imageio.v3 as iio
from torch.utils.data import DataLoader

from tools import utils
from tools.cgh_methods import SGD
from datasets.div2k import DIV2KDataset

def init_physical_params():
    """Initialize physical parameters for hologram generation."""
    # Unit conversions
    cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
    
    # Basic parameters
    channel = 1
    prop_dist = [-20 * cm, -20 * cm, -20 * cm][channel]
    wavelength = (638 * nm, 520 * nm, 450 * nm)[channel]
    feature_size = (6.4 * um, 6.4 * um)  # SLM pitch
    
    # Algorithm parameters
    scale_output = 1.0
    num_iter = 500
    
    # Resolution parameters
    homography_res = (1072, 1920)
    roi_res = (880, 1600)
    
    return {
        'channel': channel,
        'prop_dist': prop_dist,
        'wavelength': wavelength,
        'feature_size': feature_size,
        'scale_output': scale_output,
        'num_iter': num_iter,
        'homography_res': homography_res,
        'roi_res': roi_res
    }

def init_data_loader(params):
    """Initialize data loader for test images."""
    gt_dir = 'data/DIV2K_valid_HR/rgb'
    return DataLoader(
        DIV2KDataset(
            gt_dir, 
            channel=params['channel'] if params['channel'] < 3 else None,
            split='val',
            roi_res=params['roi_res'],
            homography_res=params['homography_res']
        ),
        shuffle=False
    )

def save_results(final_amp, final_phase, index, save_dir='sgd_green'):
    """Save generated amplitude and phase images."""
    # Create output directories
    dir = os.path.join('results', save_dir)
    rgb_path = os.path.join(dir, 'rgb')
    phs_path = os.path.join(dir, 'phs')
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(phs_path, exist_ok=True)
    
    # Save amplitude image
    amp_image = (torch.clamp(final_amp[0, 0, ...], min=0, max=1) * 255).round()
    amp_image = amp_image.detach().cpu().numpy().astype(np.uint8)
    iio.imwrite(os.path.join(rgb_path, f'{index}.png'), amp_image)
    
    # Save phase image
    iio.imwrite(
        os.path.join(phs_path, f'{index}.png'),
        utils.phasemap_8bit(final_phase, inverted=True)
    )

def main():
    """Main function for SGD-based phase retrieval."""
    # Initialize parameters
    params = init_physical_params()
    device = torch.device('cuda')
    
    # Initialize data loader
    gt_loader = init_data_loader(params)
    
    # Initialize SGD algorithm
    phase_only_algorithm = SGD(
        prop_dist=params['prop_dist'],
        wavelength=params['wavelength'],
        feature_size=params['feature_size'],
        num_iters=params['num_iter'],
        roi_res=params['roi_res'],
        s0=params['scale_output']
    )
    
    # Process each image
    for k, target_amp in enumerate(gt_loader):
        # Move data to device
        target_amp = target_amp.to(device)
        
        # Initialize random phase
        init_phase = (-0.5 + 1.0 * torch.rand(1, 1, *params['homography_res'])).to(device)
        
        # Generate hologram
        final_phase, final_amp = phase_only_algorithm(target_amp, init_phase)
        
        # Save results
        save_results(final_amp, final_phase, k)
        print(f'Processed image {k}')

if __name__ == '__main__':
    main()