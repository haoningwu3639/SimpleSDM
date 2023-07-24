import os
import random
from typing import Optional
from torchvision import transforms

import numpy as np
import torch
import torch.utils.data
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from utils.util import ddim_inversion
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline import StableDiffusionPipeline
from einops import rearrange
import argparse

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-v1-5/', type=str)
    parser.add_argument('--prompt', default="A black cat is running in the rain.", type=str)    
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    mixed_precision: Optional[str] = "no"   # "fp16"
):
    
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    ddim_inv_scheduler.set_timesteps(num_inference_steps)
    
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM")

    vae.eval()
    text_encoder.eval()
    unet.eval()
    
    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    shape = (1, 4, 64, 64) # Init latents
    noise_latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=weight_dtype).to(accelerator.device)
    output = pipeline(
        prompt = prompt,
        height = 512,
        width = 512,
        latents = noise_latents,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
    )
    
    # visualize noise and image
    b = noise_latents.shape[0]
    noise_latents = noise_latents / 0.18215 
    noise = vae.decode(noise_latents).sample
    noise = noise.clamp(-1, 1)
    noise = (noise / 2 + 0.5).clamp(0, 1)
    noise = rearrange(noise, "b c h w -> b h w c", b=b)
    noise = noise.squeeze(0).detach().cpu().float().numpy()
    noise = transforms.ToPILImage()((noise * 255).astype(np.uint8))
    noise.save(os.path.join(logdir, "sample_noise.png"))

    output_image = output.images[0][0] # PIL Image here
    output_image.save(os.path.join(logdir, "generated.png"))

    # VAE encode image
    image = transforms.ToTensor()(output_image)
    image = torch.from_numpy(np.ascontiguousarray(image)).float()
    image = image * 2. - 1.
    image = image.unsqueeze(0)
    image = image.to(accelerator.device)
    latents = vae.encode(image).latent_dist.sample()
    latents = latents * 0.18215
    
    # DDIM inversion
    noisy_latents = ddim_inversion(pipeline, ddim_inv_scheduler, video_latent=latents, guidance_scale=guidance_scale, num_inv_steps=num_inference_steps, prompt=prompt)[-1].to(weight_dtype)
    output = pipeline(
        prompt = prompt,
        height = 512,
        width = 512,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        latents = noisy_latents,
    )

    # visualize DDIM inversion noise and regenerated image
    output_image = output.images[0][0] # PIL Image here
    output_image.save(os.path.join(logdir, "output.png"))

    b = noisy_latents.shape[0]
    noisy_latents = 1 / 0.18215 * noisy_latents
    noise = vae.decode(noisy_latents).sample
    noise = (noise / 2 + 0.5).clamp(0, 1)
    noise = rearrange(noise, "b c h w -> b h w c", b=b)
    noise = noise.squeeze(0).detach().cpu().float().numpy()
    noise = transforms.ToPILImage()((noise * 255).astype(np.uint8))
    noise.save(os.path.join(logdir, "DDIM_noise.png"))



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    pretrained_model_path = args.ckpt
    logdir = args.logdir
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    mixed_precision = "no" # "fp16",
    test(pretrained_model_path, logdir, prompt, num_inference_steps, guidance_scale, mixed_precision)

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py