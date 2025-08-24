from typing import Dict, List

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)

from core.controllers.base_controller import BaseController
from core.controllers.quadrotor_clf_cbf_qp import QuadrotorCLFCBFController
from core.networks.conditional_unet1d import ConditionalUnet1D
from utils.normalizers import BaseNormalizer


def build_networks_from_config(config: Dict):
    action_dim = config["controller"]["networks"]["action_dim"]
    obs_dim = config["controller"]["networks"]["obs_dim"]
    obs_horizon = config["obs_horizon"]
    obstacle_encode_dim = config["controller"]["networks"]["obstacle_encode_dim"]
    return ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim * obs_horizon + obstacle_encode_dim)


def build_noise_scheduler_from_config(config: Dict):
    type_noise_scheduler = config["controller"]["noise_scheduler"]["type"]
    if type_noise_scheduler.lower() == "ddpm":
        return DDPMScheduler(
            num_train_timesteps=config["controller"]["noise_scheduler"]["ddpm"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["ddpm"]["beta_schedule"],
            clip_sample=config["controller"]["noise_scheduler"]["ddpm"]["clip_sample"],
            prediction_type=config["controller"]["noise_scheduler"]["ddpm"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "ddim":
        return DDIMScheduler(
            num_train_timesteps=config["controller"]["noise_scheduler"]["ddim"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["ddim"]["beta_schedule"],
            clip_sample=config["controller"]["noise_scheduler"]["ddim"]["clip_sample"],
            prediction_type=config["controller"]["noise_scheduler"]["ddim"]["prediction_type"],
        )
    elif type_noise_scheduler.lower() == "dpmsolver":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=config["controller"]["noise_scheduler"]["dpmsolver"]["num_train_timesteps"],
            beta_schedule=config["controller"]["noise_scheduler"]["dpmsolver"]["beta_schedule"],
            prediction_type=config["controller"]["noise_scheduler"]["dpmsolver"]["prediction_type"],
            use_karras_sigmas=config["controller"]["noise_scheduler"]["dpmsolver"]["use_karras_sigmas"],
        )
    else:
        raise NotImplementedError


class QuadrotorDiffusionPolicy(BaseController):
    def __init__(
        self,
        model: ConditionalUnet1D,
        noise_scheduler: DDPMScheduler,
        normalizer: BaseNormalizer,
        clf_cbf_controller: QuadrotorCLFCBFController, #new
        config: Dict,
        device: str = "cuda",
    ):
        self.device = device
        self.net = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = normalizer

        self.set_config(config)
        self.net.to(self.device)

        self.clf_cbf_controller = clf_cbf_controller
        self.use_clf_cbf_guidance = False if clf_cbf_controller is None else True

    def predict_action(self, obs_dict: Dict[str, List]) -> np.ndarray:
        # stack the observations
        obs_seq = np.stack(obs_dict["state"])
        # normalize observation and make it 1D
        nobs = self.normalizer.normalize_data(obs_seq, stats=self.norm_stats["obs"])
        nobs = nobs.flatten()
        # concat obstacle information to observations
        nobs = np.concatenate([nobs] + obs_dict["obs_encode"], axis=0)
        # device transfer
        nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (1, obs_horizon*obs_dim+obstacle_encode_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn((1, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            # init scheduler
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)

            # denoise
            denoise_timesteps = (
                self.noise_scheduler.timesteps[:1] if self.use_single_step_inference else self.noise_scheduler.timesteps
            )
            for k in denoise_timesteps:
                # predict noise
                noise_pred = self.net(sample=naction, timestep=k, global_cond=obs_cond)
                # inverse diffusion step (remove noise)
                if self.use_single_step_inference:
                    naction = noisy_action - noise_pred
                else:
                    naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        # unnormalize action
        naction = naction.detach().to("cpu").numpy()
        # (1, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = self.normalizer.unnormalize_data(naction, stats=self.norm_stats["act"])

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]  # (action_horizon, action_dim)

        return action

    def load_weights(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def set_config(self, config: Dict):
        self.obs_horizon = config["obs_horizon"]
        self.action_horizon = config["action_horizon"]
        self.pred_horizon = config["pred_horizon"]
        self.action_dim = config["controller"]["networks"]["action_dim"]
        self.sampling_time = config["controller"]["common"]["sampling_time"]
        self.norm_stats = {
            "act": config["normalizer"]["action"],
            "obs": config["normalizer"]["observation"],
        }
        self.quadrotor_params = config["simulator"]
        self.use_single_step_inference = config.get("controller").get("common").get("use_single_step_inference", False)

