import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
# from diffusers import StableVideoDiffusionPipeline
from diffusers_support.pipeline_zero import Zero123PlusPipeline
from diffusers_support.pipeline_svd import StableVideoDiffusionPipeline

# Load the pipeline
use_video = True
pipeline = Zero123PlusPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", torch_dtype=torch.float16
)
pipe_video = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe_video.enable_model_cpu_offload()

pipeline.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.to_k = \
    pipe_video.unet.up_blocks[1].attentions[0].transformer_blocks[0].attn1.to_k


if use_video:
    pipeline.unet = pipe_video.unet
# print(pipeline.do_classifier_free_guidance, 1)
# del pipe_video
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')
# Run the pipeline
# cond = Image.open(requests.get("https://d.skis.ltd/nrp/sample-data/lysol.png", stream=True).raw)
cond = Image.open("/localhome/aaa324/Generative Models/VideoPAPR/data/apple.png")
# cond = Image.open("/localhome/aaa324/Generative Models/VideoPAPR/data/Laptop.jpg")
result = pipeline(cond, num_inference_steps=75, pipe_svd=pipe_video, use_video=use_video).images[0]
if isinstance(result, list):
    result = result[0]
result.show()
# result.save("output.png")


# ## video generation ##
# image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
# frames = pipe_video(image.resize((1024, 576)), decode_chunk_size=8, generator=torch.manual_seed(42)).frames[0]
