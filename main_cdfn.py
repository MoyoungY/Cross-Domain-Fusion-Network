"""
Complex-valued Deep Fourier Network (CDFN) for Computer Generated Holography
This is the main training and testing script for the CDFN model.
"""

import os
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import tools.utils as utils
from tools.propagation.ASM import propagation_ASM
from models.archs.cdfn import CDFN
from datasets.div2k import DIV2KDataset
from tools.loss import PerceptualLoss

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description='CDFN Training Script')
    p.add_argument('--channel', type=int, default=1, 
                  help='Channel selection: red:0, green:1, blue:2, rgb:3')
    p.add_argument('--run_id', type=str, default='', help='Experiment name')
    p.add_argument('--proptype', type=str, default='ASM', help='Ideal propagation model')
    p.add_argument('--data_root', type=str, default='./data/DIV2K_train_HR', help='data root')
    p.add_argument('--train_root', type=str, default='./data/DIV2K_train_HR/rgb', help='data root')
    p.add_argument('--valid_root', type=str, default='./data/DIV2K_valid_HR/rgb', help='data root')
    p.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    p.add_argument('--batch_size', type=int, default=1, help='Size of minibatch')
    p.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    p.add_argument('--scale_output', type=float, default=0.95,
                  help='Scale of output applied to reconstructed intensity from SLM')
    p.add_argument('--log_dir', type=str, default='tb_logs', help='tensorboard log directory')
    p.add_argument('--exp_name', type=str, default='exp', help='experiment name')
    p.add_argument('--ipm', type=utils.str2bool, default=True, help='Use IPM')
    p.add_argument('--ds', type=utils.str2bool, default=False, help='Use DS')
    p.add_argument('--test', type=utils.str2bool, default=False, help='Test mode')
    p.add_argument('--ckpt_path', type=str, help='checkpoint path')
    return p.parse_args()

class HoloSystem(LightningModule):
    """
    PyTorch Lightning module for holographic reconstruction system using CDFN.
    
    Attributes:
        batch_size (int): Size of each training batch
        args (argparse.Namespace): Command line arguments
        channel (int): Selected color channel
        prop_dist (float): Propagation distance
        wavelength (float): Light wavelength
        feature_size (tuple): SLM pitch dimensions
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self._init_system_params(args)
        self._init_propagation_kernels()
        self._init_model_components()
        self._init_datasets()
        self._init_loss_functions()

    def _init_system_params(self, args: argparse.Namespace) -> None:
        """Initialize system parameters."""
        self.batch_size = args.batch_size
        self.args = args
        self.channel = args.channel
        self.data_root = args.data_root
        self.is_IPM = args.ipm
        self.is_DS = args.ds
        
        # Physical parameters
        cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
        self.prop_dist = [-20 * cm, -20 * cm, -20 * cm][self.channel]
        self.wavelength = (638 * nm, 520 * nm, 450 * nm)[self.channel]
        self.feature_size = (6.4 * um, 6.4 * um)
        self.pitch = 6.4 * um
        self.pad = False
        self.scale_output = args.scale_output
        
        # Resolution parameters
        self.homography_res = (1072, 1920)
        self.roi_res = (880, 1600)

    def _init_propagation_kernels(self) -> None:
        """Initialize forward and backward propagation kernels."""
        n, m = self.homography_res
        self.Hbackward = propagation_ASM(
            torch.empty(self.batch_size, 1, n, m), 
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength, 
            z=-self.prop_dist, 
            linear_conv=self.pad, 
            return_H=True
        )
        self.Hforward = propagation_ASM(
            torch.empty(self.batch_size, 1, n, m), 
            feature_size=[self.pitch, self.pitch],
            wavelength=self.wavelength, 
            z=self.prop_dist, 
            linear_conv=self.pad, 
            return_H=True
        )

    def _init_model_components(self) -> None:
        """Initialize neural network components.
        
        Sets up:
            - CDFN phase generator with IPM and DS options
        """
        self.phase_generator = CDFN(is_IPM=self.is_IPM, is_DS=self.is_DS)

    def _init_datasets(self) -> None:
        """Initialize training and validation datasets."""
        self.train_set = DIV2KDataset(
            self.args.train_root, 
            channel=self.channel, 
            split='train',
            roi_res=self.roi_res,
            homography_res=self.homography_res
        )
        self.val_set = DIV2KDataset(
            self.args.valid_root, 
            channel=self.channel, 
            split='val',
            roi_res=self.roi_res,
            homography_res=self.homography_res
        )

    def _init_loss_functions(self) -> None:
        """Initialize loss functions.
        
        Sets up:
            - MSE Loss for amplitude reconstruction
            - Perceptual Loss for visual quality
        """
        # Initialize MSE loss for amplitude reconstruction
        self.mse_loss = nn.MSELoss()
        
        # Initialize perceptual loss with specified feature weight
        self.pep_loss = PerceptualLoss(lambda_feat=0.025)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Execute one training step."""
        target_amp = batch
        target_res = self.roi_res

        # Generate phase with CDFN
        holo_phase = self.phase_generator(target_amp, self.Hbackward)

        # Process based on whether using Deep Supervision (DS)
        if self.is_DS:
            # Process multiple outputs from different stages
            recon_amps_crp = []
            weights = [0.25, 0.5, 0.75, 1.0]  # Weights for different stages
            
            for i in range(4):
                # Convert phase to complex field
                slm_complex = torch.exp(1j * holo_phase[i])
                # Propagate to image plane
                recon_complex = propagation_ASM(u_in=slm_complex, precomped_H=self.Hforward)
                # Calculate amplitude
                recon_amp = torch.pow((recon_complex.abs() ** 2) * self.scale_output, 0.5)
                # Crop to target size
                recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)
                recon_amps_crp.append(recon_amp_crp)
        else:
            # Process single output
            slm_complex = torch.exp(1j * holo_phase)
            recon_complex = propagation_ASM(u_in=slm_complex, precomped_H=self.Hforward)
            recon_amp = torch.pow((recon_complex.abs() ** 2) * self.scale_output, 0.5)
            recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)
            recon_amps_crp = [recon_amp_crp]
            weights = [1.0]

        # Crop target amplitude
        target_amp_crp = utils.crop_image(target_amp, target_res, stacked_complex=False)

        # Calculate losses
        loss_main = 0
        for amp_crp, weight in zip(recon_amps_crp, weights):
            # MSE loss
            loss_main += weight * self.mse_loss(amp_crp, target_amp_crp)
            # Perceptual loss
            loss_main += weight * self.pep_loss(
                amp_crp.repeat(1, 3, 1, 1), 
                target_amp_crp.repeat(1, 3, 1, 1)
            )

        # Log metrics
        log = {'train/loss': loss_main}
        self.log_dict(log)
        return {'loss': loss_main, 'log': log}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, Any]:
        """Execute one validation step."""
        target_amp = batch
        target_res = self.roi_res

        # Generate phase with CDFN
        holo_phase = self.phase_generator(target_amp, self.Hbackward)

        # Process based on whether using Deep Supervision (DS)
        if self.is_DS:
            # Use final stage output for validation
            slm_complex = torch.exp(1j * holo_phase[3])
            phase_vis = holo_phase[3]
        else:
            slm_complex = torch.exp(1j * holo_phase)
            phase_vis = holo_phase

        # Propagate to image plane
        recon_complex = propagation_ASM(u_in=slm_complex, precomped_H=self.Hforward)
        recon_amp = torch.pow((recon_complex.abs() ** 2) * self.scale_output, 0.5)
        
        # Crop images
        recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)
        target_amp_crp = utils.crop_image(target_amp, target_res, stacked_complex=False)

        # Calculate losses
        loss_main = self.mse_loss(recon_amp_crp, target_amp_crp)
        loss_main += self.pep_loss(
            recon_amp_crp.repeat(1, 3, 1, 1),
            target_amp_crp.repeat(1, 3, 1, 1)
        )

        # Log metrics
        log = {'val/loss': loss_main}
        self.log_dict(log, sync_dist=True)

        # Log images periodically
        if batch_idx == 22:
            self.logger.experiment.add_image(
                'phases', 
                (phase_vis[0, ...] + torch.pi) / (2 * torch.pi), 
                self.current_epoch
            )
            self.logger.experiment.add_image(
                'images', 
                recon_amp_crp[0, ...], 
                self.current_epoch
            )
            self.logger.experiment.add_image(
                'target_img_crp', 
                target_amp_crp[0, ...], 
                self.current_epoch
            )

        return {'loss': loss_main, 'log': log}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Execute one test step."""
        target_amp = batch
        target_res = self.roi_res

        # Generate phase with CDFN
        holo_phase = self.phase_generator(target_amp, self.Hbackward)

        if self.is_DS:
            # Save results for all stages during testing
            for i in range(4):
                slm_complex = torch.exp(1j * holo_phase[i])
                recon_complex = propagation_ASM(u_in=slm_complex, precomped_H=self.Hforward)
                recon_amp = torch.pow((recon_complex.abs() ** 2) * self.scale_output, 0.5)
                recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)
                
                # Save reconstructed amplitude
                iio.imwrite(
                    os.path.join('results/rgb', f'{i+1}.png'),
                    (torch.clamp(recon_amp_crp[0, 0, ...], min=0, max=1) * 255)
                    .round()
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                
                # Save phase map
                iio.imwrite(
                    os.path.join('results/phs', f'phs{i+1}.png'),
                    utils.phasemap_8bit(slm_complex.angle(), inverted=True)
                )
        else:
            # Process and save single output
            slm_complex = torch.exp(1j * holo_phase)
            recon_complex = propagation_ASM(u_in=slm_complex, precomped_H=self.Hforward)
            recon_amp = torch.pow((recon_complex.abs() ** 2) * self.scale_output, 0.5)
            recon_amp_crp = utils.crop_image(recon_amp, target_res, stacked_complex=False)

            # Create output directories
            dir = os.path.join('results', self.args.exp_name)
            rgb_path = os.path.join(dir, 'rgb')
            phs_path = os.path.join(dir, 'phs')
            os.makedirs(rgb_path, exist_ok=True)
            os.makedirs(phs_path, exist_ok=True)

            # Save results
            iio.imwrite(
                os.path.join(rgb_path, f'{batch_idx}.png'),
                (torch.clamp(recon_amp_crp[0, 0, ...], min=0, max=1) * 255)
                .round()
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
            iio.imwrite(
                os.path.join(phs_path, f'{batch_idx}.png'),
                utils.phasemap_8bit(holo_phase, inverted=True)
            )

    def train_dataloader(self) -> DataLoader:
        """Configure training data loader.
        
        Returns:
            DataLoader: Training data loader with specified batch size and settings
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Configure validation data loader.
        
        Returns:
            DataLoader: Validation data loader with specified batch size and settings
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """Configure test data loader.
        
        Returns:
            DataLoader: Test data loader with batch size 1 for evaluation
        """
        return DataLoader(
            self.val_set,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure model optimizer.
        
        Returns:
            Optimizer: Adam optimizer with specified learning rate
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr
        )
        
        return optimizer

def main():
    """Main training/testing function."""
    args = parse_args()

    if args.test:
        # Test mode
        checkpoint_path = args.ckpt_path
        system = HoloSystem.load_from_checkpoint(checkpoint_path, args=args)
        trainer = Trainer()
        trainer.test(system)
    else:
        # Training mode
        checkpoint_callback = ModelCheckpoint(
            monitor='val/loss',
            save_top_k=1,
            mode='min',
            save_last=True
        )
        
        system = HoloSystem(args)
        logger = TensorBoardLogger(save_dir=args.log_dir, name=args.exp_name)
        
        trainer = Trainer(
            max_epochs=args.num_epochs,
            logger=logger,
            strategy='ddp_find_unused_parameters_true',
            callbacks=checkpoint_callback
        )
        
        trainer.fit(system)

if __name__ == '__main__':
    main()
