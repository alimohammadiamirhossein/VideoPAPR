import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableVideoDiffusionPipeline
from diffusers_support.pipeline import Zero123PlusPipeline

# Load the pipeline
pipeline = Zero123PlusPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", torch_dtype=torch.float16
)
pipe_video = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipeline.unet = pipe_video.unet
del pipe_video
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')
# Run the pipeline
# cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
cond = Image.open("/localhome/aaa324/Generative Models/VideoPAPR/data/apple.png")
# cond = Image.open("/localhome/aaa324/Generative Models/VideoPAPR/data/Laptop.jpg")
result = pipeline(cond, num_inference_steps=75).images[0]
# result.show()
result.save("output.png")
