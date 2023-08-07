import gc
import io
import os
import time

import numpy as np
import logging

# Keep the import below for registering all model definitions
from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from RectifiedFlow.utils import save_checkpoint, restore_checkpoint
import RectifiedFlow.datasets as datasets

from RectifiedFlow.models.utils import get_model_fn
from RectifiedFlow.models import utils as mutils

from .flowgrad_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj, flowgrad_optimization 

FLAGS = flags.FLAGS

def flowgrad_edit(config, text_prompt, alpha, model_path, image_path, output_folder="output"):
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(model=score_model, ema=ema, step=0)

  state = restore_checkpoint(model_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  model_fn = mutils.get_model_fn(score_model, train=False)

  # Load the image to edit
  original_img = get_img(image_path)  
  
  log_folder = os.path.join(output_folder, 'figs')
  print('Images will be saved to:', log_folder)
  if not os.path.exists(log_folder): os.makedirs(log_folder)
  save_img(original_img, path=os.path.join(log_folder, 'original.png'))

  # Get latent code of the image and save reconstruction
  original_img = original_img.to(config.device)
  clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  

  t_s = time.time()
  latent = embed_to_latent(model_fn, scaler(original_img))
  traj = generate_traj(model_fn, latent, N=100)
  save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'reconstruct.png'))
  print('Finished getting latent code and reconstruction; image saved.')
  
  # Edit according to text prompt
  u_ind = [i for i in range(100)]
  opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=10, straightness_threshold=5e-3, lr=10.0) 

  traj = generate_traj(model_fn, latent, u=opt_u, N=100)
   
  print('Total time:', time.time() - t_s)
  save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'optimized.png'))
  print('Finished Editting; images saved.')

