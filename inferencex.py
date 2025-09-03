import argparse
import os
from datetime import datetime
from typing import Any, Optional

from omegaconf import OmegaConf
from PIL import Image

import torch

from ic_custom.pipelines.ic_custom_pipeline import ICCustomPipeline


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


def prepare_input_images(
    img_ref_path: str,
    img_target_path: Optional[str] = None,
    mask_target_path: Optional[str] = None,
    ) -> tuple[Image.Image, Image.Image, Image.Image]:

    img_ref = Image.open(img_ref_path)

    # Initialize img_target as a pure white image and mask_target as a pure black image, same size as reference
    img_target = Image.new("RGB", img_ref.size, (255, 255, 255))
    mask_target = Image.new("RGB", img_ref.size, (0, 0, 0))

    if img_target_path is not None:
        img_target = Image.open(img_target_path)
    if mask_target_path is not None:
        mask_target = Image.open(mask_target_path)

    img_ref, img_target, mask_target = resize_paired_image(img_ref, img_target, mask_target)

    return img_ref, img_target, mask_target


def get_mask_type_ids(mask_type: str) -> int:
    if mask_type.lower() == "pos-free":
        return 0
    elif mask_type.lower() == "pos-aware-precise":
        return 1
    elif mask_type.lower() == "pos-aware-drawn":
        return 2
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")


def concat_image(
    img_ref: Image.Image,
    img_target: Image.Image,
    output_img: Image.Image,
) -> Image.Image:
    concat_img = Image.new("RGB", (img_ref.width + img_target.width + output_img.width, output_img.height))
    concat_img.paste(img_ref, (0, 0))
    concat_img.paste(img_target, (img_ref.width, 0))
    concat_img.paste(output_img, (img_ref.width + img_target.width, 0))

    return concat_img


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="IC-Custom Inference.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        required=False,
        help="Hugging Face token",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=str,
        required=False,
        default=os.path.expanduser("~/.cache/huggingface/hub"),
        help="Cache directory to save the models, default is ~/.cache/huggingface/hub",
    )
    return parser.parse_args()

try:
    import folder_paths
except:
    print('not run in comfyui')
    from types import SimpleNamespace
    folder_paths = SimpleNamespace()
    folder_paths.models_dir = '/workspace/comfyui/models/'

from contextlib import contextmanager
import time
@contextmanager
def kiki_timer(name=""):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"<--- {name} ---> execution time: {end - start:.6f} sec")

def main() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    clip_path = os.path.join(folder_paths.models_dir, 'clip_vision', 'clip-vit-large-patch14')
    # t5_path = os.path.join(folder_paths.models_dir, 'clip', 't5-v1_1-xxl')
    t5_path = os.path.join(folder_paths.models_dir, 'clip', 'xflux_text_encoders')
    siglip_path = os.path.join(folder_paths.models_dir, 'clip', 'siglip-so400m-patch14-384')
    ae_path = os.path.join(folder_paths.models_dir, 'black-forest-labs', 'FLUX.1-Fill-dev', 'ae.safetensors')
    dit_path = os.path.join(folder_paths.models_dir, 'black-forest-labs', 'FLUX.1-Fill-dev', 'flux1-fill-dev.safetensors')
    redux_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'flux1-redux-dev.safetensors')
    lora_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_lora_0x1561.safetensors')
    img_txt_in_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_txt_img_in_0x1561.safetensors')
    boundary_embeddings_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_boundary_embeddings_0x1561.safetensors')
    task_register_embeddings_path = os.path.join(folder_paths.models_dir, 'IC-Custom', 'dit_task_register_embeddings_0x1561.safetensors')

    with kiki_timer('load pipeline'):
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

    with kiki_timer('quantize'):
        from optimum.quanto import freeze, qint8, quantize, quantization_map, QuantizedDiffusersModel, requantize
        quantize(pipeline.t5, qint8)
        freeze(pipeline.t5)
        quantize(pipeline.model, qint8)
        freeze(pipeline.model)

    # with kiki_timer('to cuda'):
    #     pipeline.model.to(device)

    with kiki_timer('prepare_images'):
        img_ref, img_target, mask_target = prepare_input_images(
            img_ref_path='/workspace/comfyui/github/IC-Custom/assets/inference/002/img_ref.png',
            # img_target_path='/workspace/comfyui/github/IC-Custom/assets/inference/001/img_target.png',
            # mask_target_path='/workspace/comfyui/github/IC-Custom/assets/inference/001/mask_target.png',
        )
        img_ip = img_ref.copy()

        width, height = img_target.size[0] + img_ref.size[0], img_target.size[1]

    cond_w_regions = [img_ref.size[0]]
    # kiki-options: pos-free, pos-aware-precise, pos-aware-drawn
    # mask_type_ids = get_mask_type_ids('pos-aware-precise')
    mask_type_ids = get_mask_type_ids('pos-free')
    prompt = 'masterpiece, best quality, a man wearing the shirt'
    neg_prompt = 'worst quality, normal quality, low quality, low res, blurry,'
    guidance = 40
    num_steps = 20
    seed = 123
    true_gs = 3
    width, height = 512, 1024

    # with kiki_timer('prepare_images'):
    #     img_ref, img_target, mask_target = prepare_input_images(
    #         img_ref_path='/workspace/comfyui/github/IC-Custom/assets/inference/002/img_ref.png',
    #         img_target_path='/workspace/comfyui/github/IC-Custom/assets/inference/002/img_target.png',
    #         mask_target_path='/workspace/comfyui/github/IC-Custom/assets/inference/002/mask_target.png',
    #     )
    #     img_ip = img_ref.copy()

    #     width, height = img_target.size[0] + img_ref.size[0], img_target.size[1]

    # cond_w_regions = [img_ref.size[0]]
    # # kiki-options: pos-free, pos-aware-precise, pos-aware-drawn
    # mask_type_ids = get_mask_type_ids('pos-aware-precise')
    # prompt = 'masterpiece, best quality, A man is wearing a T-shirt.'
    # neg_prompt = 'worst quality, normal quality, low quality, low res, blurry,'
    # guidance = 48
    # num_steps = 30
    # seed = 123
    # true_gs = 1

    with torch.no_grad():
        with kiki_timer('generate'):
            output_img: Image.Image = pipeline(
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
            )[0]

        output_img.save('res.png')

if __name__ == "__main__":
    main()


