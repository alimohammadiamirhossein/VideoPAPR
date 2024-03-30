import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from diffusers_support.pipeline import Zero123PlusPipeline

# Load the pipeline
# pipeline = DiffusionPipeline.from_pretrained(
#     "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
#     torch_dtype=torch.float16
# )
pipeline = Zero123PlusPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", torch_dtype=torch.float16
)
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
