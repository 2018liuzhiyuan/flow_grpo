# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # 解决 bf16 精度下计算 prev_sample_mean 时可能溢出的问题，强制转换为 fp32
    model_output = model_output.float()  # 将模型输出转为 FP32 类型
    sample = sample.float()              # 将当前样本转为 FP32 类型
    if prev_sample is not None:
        prev_sample = prev_sample.float()  # 若存在前一步样本，也转为 FP32

    # 根据时间步获取调度器中的索引（用于获取预计算的 sigma 值）
    step_index = [self.index_for_timestep(t) for t in timestep]  # 当前时间步对应的调度器索引列表
    prev_step_index = [step+1 for step in step_index]            # 前一步时间步对应的调度器索引列表

    # 获取当前时间步和前一步的 sigma 值（噪声水平），并调整形状以匹配样本维度
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))  # 当前 sigma（形状：[batch_size, 1, 1, ...]）
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))  # 前一步 sigma

    sigma_max = self.sigmas[1].item()  # 获取调度器中第二个 sigma 值（初始噪声水平）
    dt = sigma_prev - sigma  # 计算时间步长（前一步 sigma 与当前 sigma 的差值）

    # 计算当前时间步的标准差（考虑噪声水平参数）
    # torch.where 处理 sigma=1 的情况（避免除零），使用 sigma_max 替代
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * noise_level

    # 根据自定义 SDE 公式计算前一步样本的均值（核心扩散步骤） 
    prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt

    # 若未显式提供前一步样本，则通过加噪声生成
    if prev_sample is None:
        # 生成与模型输出同形状的随机噪声（用于扩散过程的随机性）
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        # 前一步样本 = 均值 + 标准差 * 时间步长平方根 * 噪声
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1 * dt) * variance_noise

    # 计算当前步骤的对数概率（基于高斯分布假设）
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt))**2))  # 高斯分布指数项
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))  # 标准差的对数项
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))  # 高斯分布的归一化常数项
    )

    # 对非批次维度求平均（得到每个样本的平均对数概率）
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    
    # 返回前一步样本、对数概率、均值、标准差
    return prev_sample, log_prob, prev_sample_mean, std_dev_t