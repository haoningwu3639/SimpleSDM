import numpy as np
import torch
import sys
import copy
import inspect
import datetime
from typing import Dict, Union
from os.path import *
from tqdm import tqdm

def get_time_string() -> str:
    x = datetime.datetime.now()
    
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device)).last_hidden_state
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device)).last_hidden_state
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    return context

def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    # model_output: epsilon, next_original_sample: pred_xstart;
    timestep, next_timestep = min(timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
    next_sample_direction = torch.sqrt(1 - alpha_prod_t_next) * model_output
    next_sample = torch.sqrt(alpha_prod_t_next) * next_original_sample + next_sample_direction
    
    return next_sample

def get_noise_pred_single(latents, t, guidance_scale, context, uncond_context, unet):
    noise_cond = unet(latents, t, encoder_hidden_states=context)["sample"]
    noise_uncond = unet(latents, t, encoder_hidden_states=uncond_context)["sample"]
    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
    noise_pred = noise_cond
    
    return noise_pred

@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, guidance_scale, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, guidance_scale, cond_embeddings, uncond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    # Note: Here, we have not added noise to t=999, that is, totally Gaussian.
    return all_latent

@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, latent, guidance_scale, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, latent, guidance_scale, num_inv_steps, prompt)
    return ddim_latents