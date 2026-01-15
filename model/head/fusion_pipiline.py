# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from model.diffusers.schedulers.scheduling_ddim import DDIMScheduler
from model.diffusers.schedulers.scheduling_deis_multistep import DEISMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from model.diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from model.diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from model.diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from model.diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from model.diffusers.schedulers.scheduling_heun_discrete import HeunDiscreteScheduler
from model.diffusers.schedulers.scheduling_pndm import PNDMScheduler
from typing import Union, Dict, Tuple, Optional
from model.loss import loss
from model.head.fusion_head import Fusion_head
from model.head.DFT_arch import DFT


class Fusion_Pipiline(nn.Module):
    def __init__(
            self,
            LDM_Enc,
            LDM_Dec,
            sample_selected,
            model_selected,
            inference_steps=5,
            num_train_timesteps=1000,
            mode='Max',
    ):
        super().__init__()
        self.mode = mode
        self.LDM_Enc = LDM_Enc  # LDM encoder
        self.LDM_Dec = LDM_Dec  # LDM decoder
        self.sample_selected = sample_selected  # Selected sampling method
        self.model_selected = model_selected  # Selected model
        self.diffusion_inference_steps = inference_steps  # Number of diffusion inference steps

        if self.model_selected == 'DFT':
            self.model = DFT(
                inp_channels=16,
                out_channels=16,
            )

        self.fusion = Fusion_head(
            out_channels=8,
            dim=128,
            bias=False,
        )

        # Initialize scheduler based on selected sampling method
        if self.sample_selected == 'DDIM':
            self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False)
        if self.sample_selected == 'ddp-solver':
            self.scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'ddp-solver++':
            self.scheduler = DPMSolverMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Deis':
            self.scheduler = DEISMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Unipc':
            self.scheduler = UniPCMultistepScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'LMS':
            self.scheduler = LMSDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Heun':
            self.scheduler = HeunDiscreteScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'PNDM':
            self.scheduler = PNDMScheduler(num_train_timesteps=num_train_timesteps)
        if self.sample_selected == 'Euler':
            self.scheduler = EulerDiscreteScheduler(num_train_timesteps=num_train_timesteps)

        self.pipeline = DenoisePipiline(self.model, self.scheduler, self.sample_selected)

    def set_loss(self, device):
        """Set loss functions"""
        self.dif_loss = nn.MSELoss().to(device)  # Diffusion loss (MSE)
        self.fusion_loss = loss.Fusion_loss(lambda1=10, lambda2=40, lambda3=40).to(device)  # Fusion loss

    def test_Fusion(self, x_in, device):
        """Fusion process for testing phase"""
        x_vis = x_in[:, :1]  # Visible light image
        x_ir = x_in[:, 1:]  # Infrared image

        with torch.no_grad():
            batch_size = x_vis.shape[0]
            device = x_vis.device
            dtype = x_vis.dtype

            # Encode visible light and infrared images
            x1, x2, h = self.LDM_Enc(x_vis, x_ir)

            # Concatenate encoded features
            latent = torch.cat((x1, x2), dim=1)

            # Perform denoising process
            latent_result, _, _, middle_feat = self.pipeline(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                image=latent,
                num_inference_steps=self.diffusion_inference_steps,  # Diffusion steps
                return_dict=False
            )

            # Fusion of intermediate features and latent results
            Fusion_result = self.fusion(middle_feat, latent_result)

            # Decode fusion result
            Fusion_result = self.LDM_Dec(Fusion_result, h)

        return Fusion_result

    def forward(self, image_vis, image_ir):
        """Forward propagation"""
        batch_size = image_vis.shape[0]  # Batch size
        device = image_vis.device  # Device
        dtype = image_vis.dtype  # Data type

        # Encode visible light and infrared images
        x1, x2, h = self.LDM_Enc(image_vis, image_ir)

        # Concatenate encoded features
        latent = torch.cat((x1, x2), dim=1)

        # Perform denoising pipeline
        latent_result, latent_noise, pre_noise, middle_feat = self.pipeline(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            image=latent,
            num_inference_steps=self.diffusion_inference_steps,
            return_dict=False
        )

        # Calculate diffusion loss
        dif_loss = self.dif_loss(pre_noise, latent_noise)

        # Fusion process
        Fusion_result = self.fusion(middle_feat, latent_result)
        Fusion_result = self.LDM_Dec(Fusion_result, h)

        # Calculate fusion loss and its components
        fusion_loss, loss_gradient, loss_l1, loss_SSIM = self.fusion_loss(image_vis, image_ir, Fusion_result)

        # Total loss = fusion loss + diffusion loss
        loss = fusion_loss + dif_loss

        # Return output dictionary
        output = {
            'Fusion': Fusion_result,  # Fusion result
            'loss': loss,  # Total loss
            'loss_gradient': loss_gradient,  # Gradient loss
            'loss_l1': loss_l1,  # L1 loss
            'loss_SSIM': loss_SSIM,  # SSIM loss
            'dif_loss': dif_loss  # Diffusion loss
        }
        return output


class DenoisePipiline:
    def __init__(self, model, scheduler, sample_selected):
        super().__init__()
        self.model = model  # Denoising model
        self.scheduler = scheduler  # Diffusion scheduler
        self.sample_selected = sample_selected  # Selected sampling method

    def __call__(
            self,
            batch_size,  # Batch size
            device,  # Device
            dtype,  # Data type
            image,  # Input image
            generator: Optional[torch.Generator] = None,  # Random number generator
            eta: float = 0.0,
            num_inference_steps: int = 50,
            return_dict: bool = True,
    ) -> Union[Dict, Tuple]:
        # Check if generator device matches
        if generator is not None and generator.device.type != device.type and device.type != "mps":
            message = (
                f"The `generator` device is `{generator.device}` and does not match the pipeline "
                f"device `{device}`, so the `generator` will be ignored. "
                f'Please use `generator=torch.Generator(device="{device}")` instead.'
            )
            raise RuntimeError("generator.device == 'cpu'", "0.11.0", message)

        # Set sampling timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # Generate random noise
        noise = torch.randn(image.shape).to(device)

        # DDIM sampling method
        if self.sample_selected == 'DDIM':
            # Randomly generate timesteps
            timesteps = torch.randint(0, 1000, (batch_size,)).long().to(device)
            # Add noise to image
            image = self.scheduler.add_noise(image, noise, timesteps).to(device)
            noise_image = image

            # Iterative denoising process
            for t in self.scheduler.timesteps:
                # Predict model noise and retain intermediate features
                model_output, middle_feat = self.model(image, t, device)
                # Perform one denoising step: D_t->D_t-1
                image = self.scheduler.step(
                    model_output, t, image, eta=eta,
                    use_clipped_model_output=True, generator=generator
                )['prev_sample']

        # DDP-Solver, DDP-Solver++, Deis, Unipc, PNDM sampling methods
        if self.sample_selected in ['ddp-solver', 'ddp-solver++', 'Deis', 'Unipc', 'PNDM']:
            # Get timestep sequence
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            # Add noise to image
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)

            # Iterative denoising process
            for t in self.scheduler.timesteps:
                # Predict model noise and retain intermediate features
                model_output, middle_feat = self.model(image, t, device)
                # Perform one denoising step
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # Heun sampling method
        if self.sample_selected == 'Heun':
            # Get timestep sequence
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            # Add noise to image
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image

            # Iterative denoising process
            for t in self.scheduler.timesteps:
                # Scale model input
                image = self.scheduler.scale_model_input(image, t)
                # Predict model noise and retain intermediate features
                model_output, middle_feat = self.model(image, t, device)
                # Perform one denoising step
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # LMS sampling method
        if self.sample_selected == 'LMS':
            # Get timestep sequence
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            # Add noise to image
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image

            # Iterative denoising process
            for t in self.scheduler.timesteps:
                # Scale model input
                image = self.scheduler.scale_model_input(image, t)
                # Predict model noise and retain intermediate features
                model_output, middle_feat = self.model(image, t, device)
                # Perform one denoising step
                image = self.scheduler.step(model_output, t, image)['prev_sample']

        # Euler sampling method
        if self.sample_selected == 'Euler':
            # Get timestep sequence
            timesteps = self.scheduler.timesteps[self.scheduler.order:].to(device)
            # Add noise to image
            image = self.scheduler.add_noise(image, noise, timesteps[:1]).to(device)
            noise_image = image
            # Set random seed
            generator = torch.manual_seed(0)

            # Iterative denoising process
            for t in self.scheduler.timesteps:
                image = self.scheduler.scale_model_input(image, t)
                model_output, middle_feat = self.model(image, t, device)
                image = self.scheduler.step(model_output, t, image, generator=generator)['prev_sample']

        # Determine return format based on return_dict
        if not return_dict:
            return (image, noise, model_output, middle_feat)

        return {'images': image, 'noise': noise}