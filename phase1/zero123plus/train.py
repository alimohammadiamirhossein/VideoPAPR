import torch
import requests
from PIL import Image
from diffusers.utils import load_image
from util.weight_transfer import transfer_unets
from diffusers import (
    Zero123PlusPipeline,
    StableVideoDiffusionPipeline,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
)
from torch.utils.data import DataLoader
from util.dataset import Zero123Dataset


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

data_dir = './dataset'
train_dataset = Zero123Dataset(root=data_dir, transform=None)

batch_size = 1
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

training_method = [
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    ][0]

cond = load_image("/localhome/aaa324/Generative Models/VideoPAPR/data/apple.png")
pipe_zero123.training_step(cond, num_inference_steps=75, pipe_svd=pipe_svd, use_video=use_video,
                            training_method=training_method, dataloader=train_dataloader, train_lr=1e-4)



