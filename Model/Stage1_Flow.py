"""Stage I flow matching objective and sampling behavior.

The velocity network is shared with Stage II. Stage I retains the original
noisy training path and clipped sampling so existing training runs remain
compatible after consolidating the model implementations.
"""

import torch
from einops import reduce

from Model.Flow_MMIF import FlowMatching, ModelPrediction


class Stage1FlowMatching(FlowMatching):
    def __init__(self, model, *, sigma_min=1e-4, sigma_max=1.0, **kwargs):
        super().__init__(model, sigma_min=sigma_min, sigma_max=sigma_max, **kwargs)

    def get_velocity_field(self, x_start, x_input, t):
        t_expanded = t.float().view(-1, 1, 1, 1) / self.num_timesteps
        return x_start - x_input + t_expanded * (x_start - x_input)

    def sample_flow_path(self, x_start, x_input, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t_expanded = t.float().view(-1, 1, 1, 1) / self.num_timesteps
        sigma_t = self.sigmas.gather(-1, t).reshape(-1, 1, 1, 1)
        return (1 - t_expanded) * x_input + t_expanded * x_start + sigma_t * noise

    def model_predictions(self, x_input, x, t, clip_denoised=True):
        if self.condition:
            x_in = torch.cat((x, x_input[0], x_input[1]), dim=1)
        else:
            x_in = x
        velocity = self.model(x_in, t.float() / self.num_timesteps)
        if clip_denoised:
            velocity = velocity.clamp(-1.0, 1.0)
        x_start = self.predict_start_from_velocity(x, t, velocity)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        return ModelPrediction(velocity, x_start)

    @torch.no_grad()
    def sample_euler(self, x_input, shape, last=True):
        # The original training validation expects [source, fused] for last=True.
        from Model.Flow_MMIF import unnormalize_to_zero_to_one
        from tqdm.auto import tqdm

        if self.condition:
            img = (x_input[0] + x_input[1]) / 2
            batch, device = img.shape[0], img.device
            input_start = img.clone()
        else:
            batch, device = shape[0], next(self.model.parameters()).device
            img = torch.randn(shape, device=device)
        dt = 1.0 / self.sampling_timesteps
        times = torch.linspace(0, self.num_timesteps - 1, self.sampling_timesteps, device=device)
        img_list = []
        for t in tqdm(times, desc='Flow matching sampling'):
            t_batch = torch.full((batch,), int(t), device=device, dtype=torch.long)
            img, _ = self.euler_sample_step(x_input, img, t_batch, dt)
            if not last:
                img_list.append(img.clone())
        if self.condition:
            img_list = [input_start] + (img_list if not last else [img])
        elif last:
            img_list = [img]
        return unnormalize_to_zero_to_one(img_list)

    def p_losses(self, imgs, t, noise=None):
        x_ir, x_vi, x_start = imgs
        if noise is None:
            noise = torch.randn_like(x_start)
        x_input = (x_ir + x_vi) / 2
        x_t = self.sample_flow_path(x_start, x_input, t, noise=noise)
        true_velocity = self.get_velocity_field(x_start, x_input, t)
        x_in = torch.cat((x_t, x_ir, x_vi), dim=1) if self.condition else x_t
        pred_velocity = self.model(x_in, t.float() / self.num_timesteps)
        loss = self.loss_fn(pred_velocity, true_velocity, reduction='none')
        return [reduce(loss, 'b ... -> b (...)', 'mean').mean()]
