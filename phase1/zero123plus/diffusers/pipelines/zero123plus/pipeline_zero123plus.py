from typing import Any, Dict, Optional
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers

import numpy
import torch
import inspect
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import torch.optim as optim
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Callable, Dict, List, Optional, Union
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from PIL import Image

from ... import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput,
    ControlNetModel,
    StableDiffusionPipeline,

)
from ...image_processor import VaeImageProcessor
from ...models.attention_processor import Attention, AttnProcessor, XFormersAttnProcessor, AttnProcessor2_0
from ...utils.import_utils import is_xformers_available
from ...utils.torch_utils import is_compiled_module, randn_tensor
from ...utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import random


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = numpy.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(
            self,
            chained_proc,
            enabled=False,
            name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
            self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
            mode="w", ref_dict: dict = None, is_cfg_guidance = False
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask)
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(self, unet: UNet2DConditionModel, train_sched: DDPMScheduler, val_sched: EulerAncestralDiscreteScheduler) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= '2.0':
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(self, noisy_cond_lat, timestep, encoder_hidden_states, class_labels, ref_dict, is_cfg_guidance, **kwargs):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat, timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs
        )

    def forward(
            self, sample, timestep, encoder_hidden_states, class_labels=None,
            *args, cross_attention_kwargs,
            down_block_res_samples=None, mid_block_res_sample=None,
            **kwargs
    ):
        cond_lat = cross_attention_kwargs['cond_lat']
        is_cfg_guidance = cross_attention_kwargs.get('is_cfg_guidance', False)
        noise = torch.randn_like(cond_lat)
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(noisy_cond_lat, timestep)
        else:
            noisy_cond_lat = self.val_sched.add_noise(cond_lat, noise, timestep.reshape(-1))
            noisy_cond_lat = self.val_sched.scale_model_input(noisy_cond_lat, timestep.reshape(-1))
        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat, timestep,
            encoder_hidden_states, class_labels,
            ref_dict, is_cfg_guidance, **kwargs
        )
        weight_dtype = self.unet.dtype
        return self.unet(
            sample, timestep,
            encoder_hidden_states, *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance),
            down_block_additional_residuals=[
                sample.to(dtype=weight_dtype) for sample in down_block_res_samples
            ] if down_block_res_samples is not None else None,
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None else None
            ),
            **kwargs
        )


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


class DepthControlUNet(torch.nn.Module):
    def __init__(self, unet: RefOnlyNoisedUNet, controlnet: Optional[ControlNetModel] = None, conditioning_scale=1.0) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(self, sample, timestep, encoder_hidden_states, class_labels=None, *args, cross_attention_kwargs: dict, **kwargs):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop('control_depth')
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split('.')[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusPipeline(StableDiffusionPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            vision_encoder: transformers.CLIPVisionModelWithProjection,
            feature_extractor_clip: CLIPImageProcessor,
            feature_extractor_vae: CLIPImageProcessor,
            ramping_coefficients: Optional[list] = None,
            safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            unet=unet, scheduler=scheduler, safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def add_controlnet(self, controlnet: Optional[ControlNetModel] = None, conditioning_scale=1.0):
        self.prepare()
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([('controlnet', self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    def get_optimizer(self, name: str):
        name = name.lower()
        if name.startswith("dadapt"):
            import dadaptation

            if name == "dadaptadam":
                return dadaptation.DAdaptAdam
            elif name == "dadaptlion":
                return dadaptation.DAdaptLion
            else:
                raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

        elif name.endswith("8bit"):  # 検証してない
            import bitsandbytes as bnb

            if name == "adam8bit":
                return bnb.optim.Adam8bit
            elif name == "lion8bit":
                return bnb.optim.Lion8bit
            else:
                raise ValueError("8bit optimizer must be adam8bit or lion8bit")

        else:
            if name == "adam":
                return torch.optim.Adam
            elif name == "adamw":
                return torch.optim.AdamW

                return Lion
            elif name == "prodigy":
                import prodigyopt

                return prodigyopt.Prodigy
            else:
                raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")

    def configure_optimizers(self, network, train_lr=0.1, optimizer_name="adamw", **optimizer_kwargs):
        optimizer_module = self.get_optimizer(optimizer_name)
        self.optimizer = optimizer_module(network.prepare_optimizer_params(), lr=train_lr, **optimizer_kwargs)


    def _get_add_time_ids(
            self,
            fps: int,
            motion_bucket_id: int,
            noise_aug_strength: float,
            dtype: torch.dtype,
            batch_size: int,
            num_videos_per_prompt: int,
            do_classifier_free_guidance: bool,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    def prepare_latents_video(
            self,
            batch_size: int,
            num_frames: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: torch.dtype,
            device: Union[str, torch.device],
            generator: torch.Generator,
            latents: Optional[torch.FloatTensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_vae_image(
                self,
                image: torch.Tensor,
                device: Union[str, torch.device],
                num_videos_per_prompt: int,
                do_classifier_free_guidance: bool,
        ):
            image = image.to(device=device)
            image_latents = self.vae.encode(image).latent_dist.mode()

            if do_classifier_free_guidance:
                negative_image_latents = torch.zeros_like(image_latents)

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                image_latents = torch.cat([negative_image_latents, image_latents])

            # duplicate image_latents for each generation per prompt, using mps friendly method
            image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

            return image_latents

    def decode_video_latents(self, latents: torch.FloatTensor, num_frames: int, decode_chunk_size: int = 14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    @torch.no_grad()
    def __call__(
            self,
            image: Image.Image = None,
            prompt = "",
            *args,
            num_images_per_prompt: Optional[int] = 1,
            guidance_scale=4.0,
            depth_image: Image.Image = None,
            output_type: Optional[str] = "pil",
            width=640,
            height=960,
            num_inference_steps=28,
            return_dict=True,
            **kwargs
    ):
        self.prepare()
        if image is None:
            raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
        assert not isinstance(image, torch.Tensor)
        image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
        image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values

        if depth_image is not None and hasattr(self.unet, "controlnet"):
            depth_image = to_rgb_image(depth_image)
            depth_image = self.depth_transforms_multi(depth_image).to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)
        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt,
                self.device,
                1,
                False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt,
                self.device,
                1,
                False
            )
        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp


        if num_images_per_prompt > 1:
            bs_embed, *lat_shape = cond_lat.shape
            assert len(lat_shape) == 3
            cond_lat = cond_lat.repeat(1, num_images_per_prompt, 1, 1)
            cond_lat = cond_lat.view(bs_embed * num_images_per_prompt, *lat_shape)

        cak = dict(cond_lat=cond_lat)

        if hasattr(self.unet, "controlnet"):
            cak['control_depth'] = depth_image


        use_video = kwargs["use_video"]
        # ## SVD compatible code ##
        if use_video:
            fps = 7
            fps = fps - 1
            num_frames = 5
            do_classifier_free_guidance = True
            noise_aug_strength = 0.02
            generator = torch.manual_seed(42)
            image_3 = self.image_processor.preprocess(image_1,
                                                      height=512, width=512).to(self.unet.device) ## to be checked
            noise = randn_tensor(image_3.shape, generator=generator,
                                 device=self.unet.device, dtype=self.unet.dtype)
            image_3 = image_3 + noise_aug_strength * noise
            image_3 = image_3.to(self.unet.device, self.unet.dtype)
            # image_embeddings = pipe_svd._encode_image(
            #     image_3,
            #     self.unet.device,
            #     1,
            #     do_classifier_free_guidance
            # )
            # print(image_embeddings.shape)
            image_embeddings = global_embeds
            if do_classifier_free_guidance:
                negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

            image_latents = self._encode_vae_image(
                image_3,
                device=self.unet.device,
                num_videos_per_prompt=1,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            image_latents = image_latents.to(self.unet.dtype)
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
            batch_size = 1
            motion_bucket_id = 127
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                self.unet.dtype,
                batch_size,
                1,
                do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(self.unet.device)
            num_channels_latents = self.unet.config.in_channels
            latents = None
            latents = self.prepare_latents_video(
                batch_size * 1,
                num_frames,
                num_channels_latents,
                512,
                512,
                self.unet.dtype,
                self.unet.device,
                generator,
                latents,
                )

            kwargs["latents_video"] = latents
            kwargs["added_time_ids"] = added_time_ids.to(self.unet.device)
            kwargs["image_latents"] = image_latents
            kwargs["image_embeddings"] = image_embeddings
            latents: torch.Tensor = super().__call__(
                None,
                *args,
                cross_attention_kwargs=cak,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=num_inference_steps,
                output_type='latent',
                width=width,
                height=height,
                **kwargs
            )
        # #########################
        else:
            latents: torch.Tensor = super().__call__(
                None,
                *args,
                cross_attention_kwargs=cak,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=num_inference_steps,
                output_type='latent',
                width=width,
                height=height,
                **kwargs
            )

        latents = latents.images
        latents = unscale_latents(latents)

        if not output_type == "latent":
            if len(latents.shape) == 4:
                image = unscale_image(self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0])
            else:
                decode_chunk_size = 1
                image = self.decode_video_latents(latents, latents.shape[1], decode_chunk_size)
        else:
            image = latents
        if len(image.shape) == 4:
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            video = []
            for i in range(image.shape[2]):
                video.append(self.image_processor.postprocess(image[:, :, i, :, :], output_type=output_type))
            image = video
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def predict_noise(
            self,
            timestep: int,
            latents: torch.FloatTensor,
            image_latents: torch.FloatTensor,
            prompt_embeds: torch.FloatTensor,
            added_time_ids: torch.FloatTensor,
            image_embeddings: torch.FloatTensor,
            guidance_scale=7.5,
    ) -> torch.FloatTensor:
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        do_classifier_free_guidance = True if guidance_scale > 1 else False
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)

        # predict the noise residual
        noise_pred = self.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_time_ids=added_time_ids,
            encoder_hidden_states_temporal=image_embeddings,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guided_target = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
        else:
            guided_target = noise_pred

        return guided_target

    def training_step(
            self,
            image: Image.Image = None,
            dataloader: DataLoader = None,
            prompt = "",
            *args,
            num_images_per_prompt: Optional[int] = 1,
            guidance_scale=4.0,
            depth_image: Image.Image = None,
            output_type: Optional[str] = "pil",
            width=640,
            height=960,
            num_inference_steps=28,
            return_dict=True,
            **kwargs
    ):
        training_method = kwargs.get("training_method", "innoxattn")

        network = LoRANetwork(
            self.unet,
            rank=4,
            multiplier=1.0,
            alpha=1.0,
            train_method=training_method,
        ).to(self.unet.device, dtype=self.unet.dtype)
        train_lr = kwargs.get("train_lr", 1e-4)
        self.configure_optimizers(network, train_lr=train_lr)

        self.prepare()

        criterion = nn.MSELoss()

        for batch in dataloader:
            input_image, gt = batch
            gt = gt.to(self.unet.device, dtype=self.unet.dtype)
            inp = input_image[0, 0]
            image = transforms.ToPILImage()(inp)
            gts = []
            for i in range(gt.shape[1]):
                image_latents = self.vae.encode(gt[0:1, i]).latent_dist.mode()
                gts.append(image_latents)
            gts = torch.stack(gts, dim=1)

            if image is None:
                raise ValueError("Inputting embeddings not supported for this pipeline. Please pass an image.")
            assert not isinstance(image, torch.Tensor)
            image = to_rgb_image(image)
            image_1 = self.feature_extractor_vae(images=image, return_tensors="pt").pixel_values
            image_2 = self.feature_extractor_clip(images=image, return_tensors="pt").pixel_values

            if depth_image is not None and hasattr(self.unet, "controlnet"):
                depth_image = to_rgb_image(depth_image)
                depth_image = self.depth_transforms_multi(depth_image).to(
                    device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
                )
            image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
            image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
            cond_lat = self.encode_condition_image(image)
            if guidance_scale > 1:
                negative_lat = self.encode_condition_image(torch.zeros_like(image))
                cond_lat = torch.cat([negative_lat, cond_lat])

            encoded = self.vision_encoder(image_2, output_hidden_states=False)
            global_embeds = encoded.image_embeds
            global_embeds = global_embeds.unsqueeze(-2)

            if hasattr(self, "encode_prompt"):
                encoder_hidden_states = self.encode_prompt(
                    prompt,
                    self.device,
                    1,
                    False
                )[0]
            else:
                encoder_hidden_states = self._encode_prompt(
                    prompt,
                    self.device,
                    1,
                    False
                )
            ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
            encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

            if num_images_per_prompt > 1:
                bs_embed, *lat_shape = cond_lat.shape
                assert len(lat_shape) == 3
                cond_lat = cond_lat.repeat(1, num_images_per_prompt, 1, 1)
                cond_lat = cond_lat.view(bs_embed * num_images_per_prompt, *lat_shape)

            cak = dict(cond_lat=cond_lat)

            if hasattr(self.unet, "controlnet"):
                cak['control_depth'] = depth_image

            use_video = kwargs["use_video"]
            # ## SVD compatible code ##
            if use_video:
                fps = 7
                fps = fps - 1
                num_frames = 5
                do_classifier_free_guidance = True
                noise_aug_strength = 0.02
                generator = torch.manual_seed(42)
                image_3 = self.image_processor.preprocess(image_1,
                                                          height=512, width=512).to(self.unet.device) ## to be checked
                noise = randn_tensor(image_3.shape, generator=generator,
                                     device=self.unet.device, dtype=self.unet.dtype)
                image_3 = image_3 + noise_aug_strength * noise
                image_3 = image_3.to(self.unet.device, self.unet.dtype)
                image_embeddings = global_embeds
                if do_classifier_free_guidance:
                    negative_image_embeddings = torch.zeros_like(image_embeddings)
                image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

                image_latents = self._encode_vae_image(
                    image_3,
                    device=self.unet.device,
                    num_videos_per_prompt=1,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )
                image_latents = image_latents.to(self.unet.dtype)
                image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
                batch_size = 1
                motion_bucket_id = 127
                added_time_ids = self._get_add_time_ids(
                    fps,
                    motion_bucket_id,
                    noise_aug_strength,
                    self.unet.dtype,
                    batch_size,
                    1,
                    do_classifier_free_guidance,
                )
                added_time_ids = added_time_ids.to(self.unet.device)
                num_channels_latents = self.unet.config.in_channels
                latents = None
                latents = self.prepare_latents_video(
                    batch_size * 1,
                    num_frames,
                    num_channels_latents,
                    512,
                    512,
                    self.unet.dtype,
                    self.unet.device,
                    generator,
                    latents,
                    )

                # 3. Encode input prompt
                cross_attention_kwargs = None
                lora_scale = (
                    self.cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
                )
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    None,
                    self.unet.device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    None,
                    prompt_embeds=encoder_hidden_states,
                    negative_prompt_embeds=None,
                    lora_scale=lora_scale,
                )

                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

                t = random.choice(self.scheduler.timesteps)

                # with torch.set_grad_enabled(True):
                with torch.no_grad():
                    self.optimizer.zero_grad()
                    with network:
                        noise_pred = self.predict_noise(
                            t,
                            latents,
                            image_latents,
                            prompt_embeds=prompt_embeds,
                            added_time_ids=added_time_ids,
                            image_embeddings=image_embeddings,
                            guidance_scale=guidance_scale,
                        )
                        latents = self.scheduler.step(noise_pred, t, latents).pred_original_sample
                        loss = criterion(latents, gts)
                        loss.backward()
                        self.optimizer.step()


        network.save_weights(
            f"./lora_weights.pt",
            dtype=torch.float32,
            )
