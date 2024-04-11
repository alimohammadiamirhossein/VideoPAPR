import torch
import requests
from PIL import Image
from diffusers.utils import load_image
from diffusers import (
    Zero123PlusPipeline,
    StableVideoDiffusionPipeline,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from util.weight_transfer import transfer_unets


def load_image(path):
    image = Image.open(path)
    return image


def load_pipeline_zero123(use_video=True):
    pipe_zero123 = Zero123PlusPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", torch_dtype=torch.float16
    )

    pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe_svd.enable_model_cpu_offload()
    pipe_zero123, pipe_svd = transfer_unets(pipe_zero123, pipe_svd)
    if use_video:
        pipe_zero123.unet = pipe_svd.unet
    pipe_zero123.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe_zero123.scheduler.config, timestep_spacing='trailing'
    )
    pipe_zero123.to("cuda:0")

    return pipe_zero123, pipe_svd


use_video = True
pipe_zero123, pipe_svd = load_pipeline_zero123()
cond = load_image("/localhome/aaa324/Generative Models/VideoPAPR/data/apple.png")
result = pipe_zero123(cond, num_inference_steps=75, pipe_svd=pipe_svd, use_video=use_video).images[0]
if isinstance(result, list):
    result = result[0]
result.show()


