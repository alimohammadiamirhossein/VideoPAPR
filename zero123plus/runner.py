import torch
import requests
from PIL import Image
from diffusers import Zero123PlusPipeline

pipeline = Zero123PlusPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", torch_dtype=torch.float16
)



