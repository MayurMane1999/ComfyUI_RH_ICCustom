
import argparse
import os
from datetime import datetime
from typing import Any, Optional

from PIL import Image, ImageOps

import torch

from .ic_custom.pipelines.ic_custom_pipeline import ICCustomPipeline
import comfy.utils
import numpy as np
import folder_paths

def ensure_divisible_by_value(image_pil: Image.Image, value: int = 8, interpolate: Image.Resampling = Image.Resampling.LANCZOS) -> Image.Image:
    """
    Ensure the image dimensions are divisible by value.

    Args:
        image_pil (Image.Image): The image to ensure divisible by value.
        value (int): The value to ensure divisible by.
    """

    w, h = image_pil.size
    w = (w // value) * value
    h = (h // value) * value
    image_pil = image_pil.resize((w, h), interpolate)
    return image_pil

def resize_paired_image(
        reference_image: Image.Image, 
        target_image: Image.Image, 
        mask_target: Image.Image, 
    ) -> tuple[Image.Image, Image.Image, Image.Image]:

    ref_w, ref_h = reference_image.size
    target_w, target_h = target_image.size

    # resize the ref image to the same height as the target image and ensure the ratio remains the same
    if ref_h != target_h:
        scale_ratio = target_h / ref_h
        reference_image = reference_image.resize((int(ref_w * scale_ratio), target_h), interpolate=Image.Resampling.LANCZOS)

    #  Ensure the image dimensions are divisible by 16.
    reference_image = ensure_divisible_by_value(reference_image, value=16, interpolate=Image.Resampling.LANCZOS)
    target_image = ensure_divisible_by_value(target_image, value=16, interpolate=Image.Resampling.LANCZOS)
    mask_target = ensure_divisible_by_value(mask_target, value=16, interpolate=Image.Resampling.NEAREST)

    return reference_image, target_image, mask_target

class RH_ICCustom_Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
            },
        }

    RETURN_TYPES = ("RHICCustomPipeline",)
    RETURN_NAMES = ("ICCustom Pipeline",)
    FUNCTION = "load"

    CATEGORY = "Runninghub/ICCustom"

    def load(self, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_dtype = torch.bfloat16

        clip_path = os.path.join(folder_paths.models_dir, 'clip_vision', 'clip-vit-large-patch14')
        # t5_path = os.path.join(folder_paths.models_dir, 'clip', 't5-v1_1-xxl')
        t5_path = os.path.join(folder_paths.models_dir, 'clip', 'xflux_text_encoders') #kiki: use xflux instead
        siglip_path = os.path.join(folder_paths.models_dir, 'clip', 'siglip-so400m-patch14-384')
        ae_path = os.path.join(folder_paths.models_dir, 'black-forest-labs', 'FLUX.1-Fill-dev', 'ae.safetensors')
        dit_path = os.path.join(folder_paths.models_dir, 'black-forest-labs', 'FLUX.1-Fill-dev', 'flux1-fill-dev.safetensors')
        redux_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'flux1-redux-dev.safetensors')
        lora_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_lora_0x1561.safetensors')
        img_txt_in_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_txt_img_in_0x1561.safetensors')
        boundary_embeddings_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_boundary_embeddings_0x1561.safetensors')
        task_register_embeddings_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_task_register_embeddings_0x1561.safetensors')

        pipeline = ICCustomPipeline(
            clip_path=clip_path,
            t5_path=t5_path,
            siglip_path=siglip_path,
            ae_path=ae_path,
            dit_path=dit_path,
            redux_path=redux_path,
            lora_path=lora_path,
            img_txt_in_path=img_txt_in_path,
            boundary_embeddings_path=boundary_embeddings_path,
            task_register_embeddings_path=task_register_embeddings_path,
            network_alpha=64,
            double_blocks_idx="0,1,2,3,4,5,6,7,8,9",
            single_blocks_idx="0,1,2,3,4,5,6,7,8,9",
            device=device,
            weight_dtype=weight_dtype,
            offload=True,
        )
        pipeline.set_pipeline_offload(True)
        pipeline.set_show_progress(True)

        from optimum.quanto import freeze, qint8, quantize, quantization_map, QuantizedDiffusersModel, requantize
        #quantize(pipeline.t5, qint8)
        #freeze(pipeline.t5)
        quantize(pipeline.model, qint8)
        freeze(pipeline.model)
        return (pipeline, )
        # return (None, )

class RH_ICCustom_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RHICCustomPipeline", ),
                "ref_image": ("IMAGE", ),
                "prompt": ("STRING", {"multiline": True,
                                      'default': ''}),
                # "width": ("INT", {"default": 1024}),
                # "height": ("INT", {"default": 1024}),
                "num_inference_steps": ("INT", {"default": 25}),
                "guidance": ("FLOAT", {"default": 40.0}),
                "true_gs": ("FLOAT", {"default": 3.0}),
                "seed": ("INT", {"default": 20, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "The random seed used for creating the noise."}),
            },
            "optional": {
                "target_image": ("IMAGE", ),
                "target_mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "sample"

    CATEGORY = "Runninghub/ICCustom"

    def tensor_2_pil(self, img_tensor):
        if img_tensor is not None:
            i = 255. * img_tensor.squeeze().cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            return img
        else:
            return None
        
    def prepare_input_images(self, ref_image, target_image, target_mask):
        img_ref = ref_image

        # Initialize img_target as a pure white image and mask_target as a pure black image, same size as reference
        img_target = Image.new("RGB", img_ref.size, (255, 255, 255))
        mask_target = Image.new("RGB", img_ref.size, (0, 0, 0))

        if target_image is not None:
            img_target = target_image
        if target_mask is not None:
            mask_target = target_mask

        img_ref, img_target, mask_target = resize_paired_image(img_ref, img_target, mask_target)

        return img_ref, img_target, mask_target

    def sample(self, **kwargs):
        pipeline = kwargs.get('pipeline')
        # width = kwargs.get('width')
        # height = kwargs.get('height')
        ref_image = self.tensor_2_pil(kwargs.get('ref_image'))
        target_image = self.tensor_2_pil(kwargs.get('target_image'))
        target_mask = self.tensor_2_pil(kwargs.get('target_mask'))

        if target_mask is None: #pos-free
            target_image = None
            mask_type_ids = 0
        else: #pos-precise
            mask_type_ids = 1
            target_mask = ImageOps.invert(target_mask).convert('RGB')

        img_ref, img_target, mask_target = self.prepare_input_images(
            ref_image, target_image, target_mask
        )
        img_ip = img_ref.copy()
        cond_w_regions = [img_ref.size[0]]

        width, height = img_target.size[0] + img_ref.size[0], img_target.size[1]
        prompt = kwargs.get('prompt')
        guidance = kwargs.get('guidance')
        num_steps = kwargs.get('num_inference_steps')
        seed = kwargs.get('seed') % (2 ** 32)
        true_gs = kwargs.get('true_gs')

        #kiki:hardcode hyperparameters
        neg_prompt = 'worst quality, normal quality, low quality, low res, blurry,'
        self.pbar = comfy.utils.ProgressBar(num_steps)

        with torch.no_grad():
            image_gen = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                guidance=guidance,
                num_steps=num_steps,
                seed=seed,
                img_ref=img_ref,
                img_target=img_target,
                mask_target=mask_target,
                img_ip=img_ip,
                cond_w_regions=cond_w_regions,
                mask_type_ids=mask_type_ids,
                use_background_preservation=False,
                use_progressive_background_preservation=False,
                background_blend_threshold=0.0,
                true_gs=true_gs,
                neg_prompt=neg_prompt,
                update_func=self.update,
            )[0]

        image = np.array(image_gen).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

        return (image, )
    
    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub ICCustom Loader": RH_ICCustom_Loader,
    "RunningHub ICCustom Sampler":RH_ICCustom_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RunningHub ICCustom Loader": "RunningHub ICCustom Loader",
    "RunningHub ICCustom Sampler": "RunningHub ICCustom Sampler",
} 
