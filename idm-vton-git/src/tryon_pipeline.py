# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    FusedAttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline



if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0",
        ...     torch_dtype=torch.float16,
        ...     variant="fp16",
        ...     use_safetensors=True,
        ... )
        >>> pipe.to("cuda")

        >>> img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        >>> mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

        >>> init_image = load_image(img_url).convert("RGB")
        >>> mask_image = load_image(mask_url).convert("RGB")

        >>> prompt = "A majestic tiger sitting on a bench"
        >>> image = pipe(
        ...     prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=50, strength=0.80
        ... ).images[0]
        ```
"""


# # 노이즈를 재스케일하는 함수입니다 -> 노이즈로 인한 과도한 왜곡이나 조정을 방지하는 함수.
# Common Diffusion Noise Schedules and Sample Steps are Flawed논문에 기반해서 함수를 구성함.
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
     # 텍스트 노이즈의 표준 편차를 계산합니다.
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    # CFG 노이즈의 표준 편차를 계산합니다.
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # 가이드의 결과를 재스케일합니다 (과다 노출 방지).
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
     # 재스케일된 노이즈와 원본 가이드 결과를 혼합하여 이미지가 밋밋해지지 않도록 합니다.
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

# PIL 이미지형식인 마스크 이미지를 토치 텐서로 변환하는 함수(mask이미지는 여러장일 수 있는거 같음)
def mask_pil_to_torch(mask, height, width):
    #mask = 변환할 마스크이미지, height,width =입력된 마스크 이미지를 변환해서 바꿀 크기
    # preprocess mask
    if isinstance(mask, (PIL.Image.Image, np.ndarray)):
        mask = [mask] # 마스크가 이미지나 넘파이배열인 경우 리스트로 변환

    if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):#입력된 MASEKD이미지가 PIL형식일경우
        mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
        mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0) #L 모드 = 이미지를 단일채널흑백으로 변환하는거
        mask = mask.astype(np.float32) / 255.0 # 값을 [0, 1] 범위로 정규화
    # 리스트에 포함된 배열을 단일 텐서로 변환합니다.
    elif isinstance(mask, list) and isinstance(mask[0], np.ndarray): #입력된 MASKED이미지가 넘파이배열형식일 경우
        mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

    mask = torch.from_numpy(mask) # 넘파이 배열을 토치 텐서로 변환
    return mask


#기존에 prepare_mask_and_masked_image함수 있었는데 지금도 사용안하고 이후 나올 버전에 없어질거라 ㅈ없앰
#def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    #encoder_output: 인코더의 출력값으로, torch.Tensor 객체
    #generator: 선택적으로 제공되는 난수 생성기 객체(torch.Generator)로, sample_mode가 "sample"일 때 잠재 벡터를 샘플링할 때 사용할 수 있습니다.
    #sample_mode: 잠재 벡터를 가져오는 방식으로 "sample" 또는 "argmax" 값을 가질 수 있습니다. 기본값은 "sample"입니다.

    #여기서 encoder_output에 latent속성이 있다면 인코더아웃풋에서 미리 계산된 잠재벡터가 있다는의미고
    #latent_dist면 인코더의 아웃풋이 존재하고 잠재벡터는 아직 추출안했다는 의미임.

    #여기서 기본적으로 다루고있는거 분포객체임 encoder_output도 분포객체인데
    #배치사이즈10 이고 분포객체의 차원이 5차원에 속해있다면 크기가 [10,5]가 될것임
    #여기서 분포객체는 일반데이터와다르게 평균과 표준편차를 값으로 가지고 있음
    #그래서 무작위로 추출하거나 최빈값추출이 가능한것.

    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample": #encoder_output이 latent_dist 속성을 가지고 있으며 sample_mode가 "sample"인 경우
        return encoder_output.latent_dist.sample(generator) 
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":#encoder_output이 latent_dist 속성을 가지고 있으며 sample_mode가 "argmax"인 경우
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):#encoder_output이 latents 속성을 가지고 있는 경우
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    #num_inference_steps: 샘플 생성에 사용할 확산 단계 수를 나타내는 정수입니다. timesteps가 제공되지 않을 때만 사용됩니다
    #device: 타임스텝이 위치할 장치를 지정합니다. 예를 들어 cuda 또는 cpu 등을 설정할 수 있습니다. None이면 장치 이동 없이 그대로 둡니다.
    #timesteps: 사용자 정의 타임스텝 리스트입니다. num_inference_steps 대신 사용할 수 있으며, 스케줄러의 기본 타임스텝 대신 특정한 간격을 지정하고자 할 때 사용합니다.
    #**kwargs: 스케줄러의 set_timesteps 메서드로 전달할 추가 인자들입니다.
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        #timesteps가 제공된 경우, scheduler.set_timesteps 메서드가 timesteps 인자를 받을 수 있는지 확인
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()) 
        if not accepts_timesteps: #스케줄러가 timesteps 파라미터를 지원하지 않으면 ValueError를 발생시킴
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        #사용자 정의 timesteps가 지정되었다면, scheduler.set_timesteps를 호출해 타임스텝과 장치를 설정
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps #scheduler.timesteps 속성을 통해 최종 설정된 타임스텝 값을 가져와서 timesteps에 할당함
        num_inference_steps = len(timesteps) #num_inference_steps는 최종 타임스텝 리스트의 길이로 설정됩니다.
    else: #timesteps가 None이면 기본 설정으로 돌아가 num_inference_steps를 이용해 타임스텝을 설정
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class StableDiffusionXLInpaintPipeline(
    DiffusionPipeline, #이 클래스는 모든 파이프라인의 기본 기능(다운로드, 저장, 장치 설정 등)을 제공합니다.
    TextualInversionLoaderMixin, #텍스트 인버전(Textual Inversion) 임베딩을 불러오는 기능을 추가합니다.
    StableDiffusionXLLoraLoaderMixin, # LoRA(저비용 어댑터) 가중치를 로드 및 저장하는 기능을 추가합니다.
    FromSingleFileMixin, #.ckpt 파일에서 모델을 로드할 수 있는 기능을 추가합니다.
    IPAdapterMixin, #IP 어댑터를 로드하는 기능을 추가합니다.
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->unet->vae"

    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
        "unet_encoder",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "add_neg_time_ids",
        "mask",
        "masked_image_latents",
    ]

    def __init__(
        self,
        vae: AutoencoderKL, # VAE(변이형 오토인코더) 모델로, 이미지를 잠재 공간에서 인코딩 및 디코딩하여 모델이 압축된 표현을 사용해 이미지 변환을 수행할 수 있게 합니다.
        text_encoder: CLIPTextModel, #CLIP 모델의 텍스트 인코더로, 텍스트를 임베딩 벡터로 변환해 U-Net 모델이 이해할 수 있도록 합니다.
        text_encoder_2: CLIPTextModelWithProjection, #두 번째 CLIP 텍스트 인코더로, text_encoder와 함께 텍스트 임베딩을 제공합니다.
        tokenizer: CLIPTokenizer, # 텍스트 데이터를 CLIP 모델의 입력 형식에 맞게 토큰화하는 도구입니다.
        tokenizer_2: CLIPTokenizer, #두 번째 텍스트 인코더와 호환되는 토크나이저입니다.
        unet: UNet2DConditionModel, #이미지의 노이즈를 제거하는 U-Net 모델로, 인코딩된 이미지 잠재 벡터를 사용해 디노이징을 수행합
        unet_encoder: UNet2DConditionModel,#CLIP 기반의 이미지 인코더로, 입력 이미지를 잠재 공간에 맞게 변환하여 파이프라인에서 처리할 수 있는 형태로 만듭니다.
        scheduler: KarrasDiffusionSchedulers,#U-Net과 함께 노이즈 제거 단계(추론 단계)를 관리하는 스케줄러입니다. 다양한 스케줄러 유형(예: DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler)을 사용할 수 있습니다.
        image_encoder: CLIPVisionModelWithProjection = None,#CLIP 기반의 이미지 인코더로, 입력 이미지를 잠재 공간에 맞게 변환하여 파이프라인에서 처리할 수 있는 형태로 만듭니다.
        feature_extractor: CLIPImageProcessor = None,#CLIP 모델의 이미지 처리 요구사항에 맞게 이미지를 전처리하여 image_encoder에 전달할 수 있도록 도와줍니다
        requires_aesthetics_score: bool = False,#aesthetic_score 조건이 필요한지 설정하는 옵션입니다. stabilityai/stable-diffusion-xl-refiner-1-0 구성에서 사용됩니다.
        force_zeros_for_empty_prompt: bool = True,#빈 프롬프트에 대해 항상 0으로 설정된 임베딩을 사용할지 여부를 설정하는 옵션입니다.
    ):
        super().__init__()

        self.register_modules(
            #CLIPTextModel vs CLIPTextModelwithProjection
            #CLIPTextModel은 텍스트를 CLIP 모델이 이해할 수 있는 잠재 벡터로 인코딩하는 역할을 하는거
            #텍스트로 이미지를 diffusion하는 모델에서 텍스트와 이미지를 비교하기 위해서 텍스트와 이미지를 비교하기 위해서 이미지와 텍스트를 비교가능한 차원으로 옮겨주어야함
            #그래서 CLIPTextModelwithProjection는 선형변환을 이용해서 텍스트를 이미지와 같은 공간으로 투영시키는 모델
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            image_encoder=image_encoder,
            unet_encoder=unet_encoder,
            feature_extractor=feature_extractor,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )


    #슬라이싱 -> 특정한 차원을 기준으로 데이터를 나눠서처리 -> 배치크기가 클 때 적합함.
    #타일링 -> 이미지가 1024*1024이면 이걸 256*256이렇게 해서 4번처리하는것 -> 고해상도일때 적합
    #두 방법 모두 gpu메모리를 절약하기위한 방법

    # vae의 슬라이싱 활성화
    def enable_vae_slicing(self):

        self.vae.enable_slicing()

    # vae의 슬라이싱 비활성화
    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    # vae의 타일링 활성화
    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    #vae의 타일링 비활성화
    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        # 이미지 인코더의 데이터 타입을 가져옵니다.
        dtype = next(self.image_encoder.parameters()).dtype
        # print(image.shape)
        # 이미지가 torch.Tensor가 아닌 경우, feature_extractor(clip모델의 feature_extractor를 사용)를 사용하여 텐서로 변환
        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
        ## 이미지를 지정된 장치와 데이터 타입으로 변환하고 gpu 메모리로 올림
        image = image.to(device=device, dtype=dtype)

        #이미지 인코더도 clip모델의 전처리 함수를 그대로 가져와서 사용(라이브러리 가져오는것)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else: 
            ## 이미지 임베딩을 생성하고
            image_embeds = self.image_encoder(image).image_embeds
            #num_images_per_prompt크기만큼 이미지를 복사(idm-vton에서는 한개씩 처리해서 이거 1로 고정되어있음)
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds #image_embeds는 입력된 이미지를 인코딩한 결과이고 uncond_는 값이 모두 0인데 크기는 입력이이미지와 같은 텐서 

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_ip_adapter_image_embeds
    def prepare_ip_adapter_image_embeds(self, ip_adapter_image, device, num_images_per_prompt):
        # if not isinstance(ip_adapter_image, list):
        #     ip_adapter_image = [ip_adapter_image]

        # if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
        #     raise ValueError(
        #         f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
        #     )

        # UNet의 encoder_hid_proj가 ImageProjection인지 확인하여 output_hidden_state를 결정
        output_hidden_state = not isinstance(self.unet.encoder_hid_proj, ImageProjection)
        # print(output_hidden_state)
        #인코더함수이용해서 인코딩된 이미지와, 0으로 채워진 같은크기 텐서를 image_embeds와 negative_image_embeds로 받음(negative embeds는 값이 다 0으로 채워져있음)
        image_embeds, negative_image_embeds = self.encode_image(
            ip_adapter_image, device, 1, output_hidden_state
        )
        # print(single_image_embeds.shape)
        # single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
        # single_negative_image_embeds = torch.stack([single_negative_image_embeds] * num_images_per_prompt, dim=0)
        # print(single_image_embeds.shape)

        #Classifier-Free Diffusion Guidance을 위해서 unconditional embed와 conditional embed를 concat해서 image_embed를 반환
        if self.do_classifier_free_guidance:
            image_embeds = torch.cat([negative_image_embeds, image_embeds])
            image_embeds = image_embeds.to(device)


        return image_embeds


    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.encode_prompt
    def encode_prompt( #프롬프트를 인코딩해서 hiddenstate로 변화하는 함수
        self,
        prompt: str, #인코딩할 프롬프트
        prompt_2: Optional[str] = None, #tokenizer_2와 text_encoder_2에 전달될 프롬프트입니다. 정의되지 않은 경우, prompt가 두 텍스트 인코더 모두에 사용됩니다.
        device: Optional[torch.device] = None, 
        num_images_per_prompt: int = 1, #각 프롬프트당 생성할 이미지 수.
        do_classifier_free_guidance: bool = True,#Classifier Free Guidance를 사용할지 여부.
        negative_prompt: Optional[str] = None, #이미지 생성의 지침으로 사용하지 않을 프롬프트입니다. 정의되지 않은 경우, negative_prompt_embeds를 대신 전달해야 합니다. Guidance를 사용하지 않을 때 (guidance_scale이 1 미만일 때) 무시됩니다.
        negative_prompt_2: Optional[str] = None,#tokenizer_2와 text_encoder_2에 전달될, 이미지 생성을 지시하지 않을 프롬프트입니다. 정의되지 않은 경우, negative_prompt가 두 텍스트 인코더 모두에 사용됩니다.
        prompt_embeds: Optional[torch.FloatTensor] = None,#미리 생성된 텍스트 임베딩으로, 프롬프트 가중치 조정 등의 작업에 쉽게 사용할 수 있습니다. 제공되지 않은 경우 prompt 인수로부터 텍스트 임베딩이 생성됩니다
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,#제공되지 않은 경우 negative_prompt 인수로부터 네거티브 프롬프트 임베딩이 생성됩니다.
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        #그냥 텍스트임베딩은 각 단어에 대한 개별적인 임베딩을 제공해서 세부적인 의미를 파악하고 pooled는 maxpooling이나 avgpooling을 적용해서 문장의 전체적인 의미를 파악함.
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,#nagtive_prompt의 pooled_embeds
        lora_scale: Optional[float] = None,#텍스트 인코더의 모든 LoRA 레이어에 적용할 LoRA 스케일 값입니다. LoRA 레이어가 로드된 경우에만 적용됩니다.
        clip_skip: Optional[int] = None,#프롬프트 임베딩을 계산할 때 CLIP에서 건너뛸 레이어 수입니다. 값이 1인 경우, 최종 레이어 직전의 출력을 사용하여 프롬프트 임베딩을 계산합니다.
    ):
        
        #Encodes the prompt into text encoder hidden states.


        device = device or self._execution_device

        #lora_scale을 따로 지정하면 그걸로 set해줌
        #Lora_scale이란 추가된 Lora파라미터들이 기존 모델의 가중치에 적용될때 사용하는 가중치 값으로 새로학습된 파라미터들을 기존의 모델에 얼마나 반영할 것인가를 조절함
        #lora_scale은 LoRA를 통해 추가된 학습 파라미터의 강도를 조절하여, 모델이 특정 조건이나 텍스트 프롬프트에 대해 얼마나 민감하게 반응할지 결정
        #여기서는 프롬프트가 이미지생성에 얼마만큼 크게 영향을 미칠지 그 강도를 조정하는 용도
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            #동적으로 lora_scale을 조정하는 부분
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)
            #텍스트인코더2가 존재할경우 똑같이 lora_scale조정
            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        #prompt가 문자열일경우 리스트로변환함.
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else: #프롬프트가 없으면 프롬프트 embeds들어오니까 크기 이거로 설정
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [self.text_encoder_2]
        )

        #미리 생성된 prompt임베딩이 없을때 프롬프트를 임베딩으로 변환하는 과정.
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2] #일반적으로 신경망의 마지막층은 분류나 회귀같은 결과에 특화되어있어서 일반적인 특성보다는 어떤 목표를 위한 특성정보를 포함하기떄문에
                    #뒤에서 두번째 레이어를 사용함
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # classifier free guidance을 위해서 unconditional embeddings 생성

        #negative prompt혹은 force_zeros_for_empty_prompt가 있으면 zero_out_negative_prompt = True
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt

        #cfg가 true이고 negative_prompt_embed가 없고 negative_prompt가 있으면 prompt이용해서 embed생성(모두0으로된 텐서생성)
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)
            #zero_out_negative_prompt이게 없으면 빈 문자열로 negative_prompt생성
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # negative_prompt와 negative_prompt_2가 문자열이면 batch_size크기만큼 리스트형태로 복제
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )


            uncond_tokens: List[str]
            # # prompt가 None이 아니고, negative_prompt의 타입이 prompt와 다르면 오류를 발생
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
                # # negative_prompt의 길이가 batch_size와 맞지 않으면 오류를 발생
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
                #cfg를 위한 unconditonal_tokens생성
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            #unconditional_embeds를 위한 리스트생성
            negative_prompt_embeds_list = []

            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                #프롬프트를 토큰화
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                #토큰된거를 임베딩화
                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True, #모들 레이어에서의 hidden state를 반환함
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0] #인코더의 첫번째 출력레이어를 pooled로 사용(전체적인 의미파악에 유용함)
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]#인코더의 마지막에서 두번째레이어를 일반임베딩으로 사용(세부적인 의미파악에 유용)

                negative_prompt_embeds_list.append(negative_prompt_embeds)#임베딩의 모든 레이어출력이 리스트로 저장됨.

            negative_prompt_embeds = torch.concat(negative_prompt_embeds_list, dim=-1)#마지막차원에 모든레이어의 출력을 추가해줌



        if self.text_encoder_2 is not None:
            #프롬프트임베딩의 타입을 text_encoder의 input타입으로 맞춰줌
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
        else:
            #text_encoder_2가 없으면 diffusion의 unet의 데이터타입으로 프롬프트임베딩 타입을 변경해줌
            prompt_embeds = prompt_embeds.to(dtype=self.unet.dtype, device=device)

        #bs_embed = 배치사이즈로 프롬프트개수, seq_len = 각 프롬프트가 토큰화 되었을때 그 토큰화된 길이 
        bs_embed, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        #
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1) #각 프롬프트당 생성할이미지수로 idm-vton에서는 1로 고정되어있음(프롬프트하나로 이미지하나생성하니까)
        #그래서 이 repeat코드는 사실상 의미가 없다고 봐도 무방함.
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)#view 메서드는 텐서를 지정된 크기로 다시 구성하하는거

        #dfg를 수행하기 위해서 unconditional 임베딩에 대해서도 위에서와 똑같이 복제하고 크기 재구성
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        #pooled프롬프트 임베딩도 똑같이 생성할 이미지 개수만큼 복제(여기선 1로고정)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )

        #dfg가 true면 negative_pooled_prompt도 생성할 이미지 개수만큼 복제(여기서 1로 고정)
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                #현재 LoRA 레이어에 적용된 스케일을 제거하거나 원래 값으로 되돌리는 부분
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                #현재 LoRA 레이어에 적용된 스케일을 제거하거나 원래 값으로 되돌리는 부분
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    
    def prepare_extra_step_kwargs(self, generator, eta):
        #eta는 DDIM에서만 사용됨 - 샘플링 과정에서 노이즈의 양을 제어하는 변수(0에서 1사이값을 가짐)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        #inspect.signature함수를 이용해서 schduler에 있는 매개변수 모두 가져오고 그중에 eta있는지 확인
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta: #schduler가 eta가 존재하는 ddpm모델이면 extra_step_kwargs에 eta저장
            extra_step_kwargs["eta"] = eta

        #inspect.signature함수를 이용해서 schduler에 있는 매개변수들을 모두 가져오고 그중 generator가져옴
        #generator는 난수 생성기로, 샘플링 과정에서 재현성(reproducibility)을 위해 사용
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:#generator지원하면 저장
            extra_step_kwargs["generator"] = generator

        #return값은 step 메서드에서 필요한 매개변수만 포함하고있음
        return extra_step_kwargs

    def check_inputs( #input이 제대로 들어왔는지 체크하는부분
        self,
        prompt,
        prompt_2,
        image,
        mask_image,
        height,
        width,
        strength,
        callback_steps,
        output_type,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        padding_mask_crop=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
        if padding_mask_crop is not None:
            if not isinstance(image, PIL.Image.Image):
                raise ValueError(
                    f"The image should be a PIL image when inpainting mask crop, but is of type" f" {type(image)}."
                )
            if not isinstance(mask_image, PIL.Image.Image):
                raise ValueError(
                    f"The mask image should be a PIL image when inpainting mask crop, but is of type"
                    f" {type(mask_image)}."
                )
            if output_type != "pil":
                raise ValueError(f"The output type should be PIL when inpainting mask crop, but is" f" {output_type}.")

    def prepare_latents( #이미지를 잠재공간으로 변환하고 노이즈를 더한 임베딩반환하는 함수 
        self,
        batch_size, #생성할 이미지의 개수
        num_channels_latents,#잠재 공간의 채널 수.
        height,#원본 이미지의 높이와 너비
        width,
        dtype,#데이터타입
        device,
        generator,#난수생성기로 노이즈생성할때 필요
        latents=None,#미리 만들어둔 잠재공간에서의 텐서 
        image=None,#원본이미지
        timestep=None,#노이즈 추가할때 필요한 timestep
        is_strength_max=True,#노이즈 강도설정
        add_noise=True,#노이즈 추가할지여부
        return_noise=False,#노이즈 반환할지여부
        return_image_latents=False,#이미지의 잠재텐서를 반환할지 여부
    ):
        #vae_scale_factor는 VAE에서 이미지 크기를 줄이는 비율. 이미지 해상도를 줄여 효율성을 높이기 위해 사용
        #shape = 잠재공간에서의 이미지크기
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)

        #generator와 batchsize크기 비교함, generator는 리스트여야함 아니면 오류발생
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        #노이즈 강도인 is_strength_max가 없거나 image or timestep이 없으면 오류발생
        if (image is None or timestep is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise timestep has not been provided."
            )
        
        #이미지가 RGBA 즉 채널의 크기가 4인경우 이거 잠재공간의 이미지로 직접사용
        if image.shape[1] == 4:
            image_latents = image.to(device=device, dtype=dtype)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        #이미지의 채널크기가 4가 아닌경우 encode__vae_image함수를 사용해서 그 결과를 사용함
        elif return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)
            image_latents = self._encode_vae_image(image=image, generator=generator)
            #배치사이즈만큼 이미지복사 -> 여기서 1이니까 복제안하는거랑 똑같음.
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        #latents가 정의되지 않았을때
        if latents is None and add_noise:
            #shape에 맞는 무작위의 노이즈 텐서를생성함.
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # is_strength_max가 True면 맨처음의 임베딩 벡터를 완전한 노이즈에서 시작하고 그렇지 않으면 이미지+노이즈로 시작함.
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        elif add_noise:#latents가 정의되었을때 스케줄러를 이용해서 초기노이즈를 더해줌.
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = image_latents.to(device)
        #매개변수 조건에 맞게ouput조절
        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    #이미지를 잠재공간으로 인코딩하는 함수 
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        #dtype = input이미지의 데이터타입
        dtype = image.dtype
        #force_upcast가 true이면 float32로 데이터형을 바꿔줌.
        if self.vae.config.force_upcast:
            image = image.float()
            self.vae.to(dtype=torch.float32)
        #generator가 리스트형태면 
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
                #image.shape[0]만큼 for문 즉 배치사이즈(이미지의 개수)만큼 반복문 돌면서
                #retriee_latents함수를 이용해서 각 이미지에 대한 latent를 구함.
            ]
            #torch.cat(..., dim=0)은 첫 번째 차원(배치 차원)으로 연결하여 (batch_size, latent_dim, height, width) 형태의 최종 텐서를 만들어줌.
            image_latents = torch.cat(image_latents, dim=0)
        else:
            #generator가 리스트가 아닌 한개만있을경우 모든 이미지에 대해서동일한 generator를 적용
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)
        
        if self.vae.config.force_upcast:
            self.vae.to(dtype)

        #latent공간의 이미지를 원래 이미지의 데이터타입으로 변경
        image_latents = image_latents.to(dtype)
        #vae에 있는 scaling_factor를 사용해서 잠재공간의 이미지를 스케일링 해줌.
        image_latents = self.vae.config.scaling_factor * image_latents

        #반환값인 image_latents는 (batch_size, latent_dim, height, width)가됨.(스케일링된)
        #이미지의 기본 차원 -> batch_size, channel, height, width인데
        #잠재공간으로 변환하면 이미지의 크기는 줄이고 채널수를 늘리게됨
        #그래서 반환값이  (batch_size, latent_dim, height, width) 이렇게 되는것.
        return image_latents

    #마스킹된 이미지와 마스킹된이미지의 잠재벡터를 반환하는 함수 
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        #mask는 사용하기 위해서 masked_image를 전처리한거고 masked_image는 원본 masking이미지로 여기서 masked_image의
        #잠재벡터를 구하기 위해서 필요함 -> 마스킹된 이미지의 잠재벡터 구하는 이유는 원본이미지의 잠재벡터와 결합하기 위해서임.
        mask = torch.nn.functional.interpolate(#mask이미지의 크기를 변환하고 데이터타입변환(잠재벡터와 결합하기위해서)
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        #프롬프트마다 다른 이미지를 생성하기위해서mask이미지를 복제하는데 여기서는 이미지1개만 생성하니까
        # mask.shape[0]이게 1임 그래서 복제가 안됨
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)

        #dfg면 negative_prompt까지쓰니까 mask크기를 두배로 늘리고 아니면 그냥 mask사용
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask

        #masked_image또한 위에서 mask에 대해서 실행한 것과 동일한 과정을 거쳐서 실행하는부분
        if masked_image is not None and masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = None

        if masked_image is not None:
            if masked_image_latents is None:
                masked_image = masked_image.to(device=device, dtype=dtype)
                masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

            if masked_image_latents.shape[0] < batch_size:
                if not batch_size % masked_image_latents.shape[0] == 0:
                    raise ValueError(
                        "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                        f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                        " Make sure the number of images that you pass is divisible by the total requested batch size."
                    )
                masked_image_latents = masked_image_latents.repeat(
                    batch_size // masked_image_latents.shape[0], 1, 1, 1
                )

            masked_image_latents = (
                torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
            )

            # aligning device to prevent device errors when concating it with the latent model input
            masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)

            #cgf와 이미지배치사이즈를 고려한 mask, masked_image_latents를 반환
        return mask, masked_image_latents

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device, denoising_start=None):
        #매개변수 순서대로 - diffusion step의 수, 강도, denoising의 시작점.

        #denoising시작점이 정의안되어있을경우 강도(strength)와 디노이징 스텝수(num_inference_steps)를 겨로해서 init_time_step생성
        if denoising_start is None:
            #init_timestep은 "추론을 시작할 지점까지의 남은 단계 수
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            #t_start는 실제 추론을 "어디서부터 시작할지"를 결정하는 인덱스
            t_start = max(num_inference_steps - init_timestep, 0)
            #denoising시작점이 주어지면 t_start = 0
        else:
            t_start = 0

        #스케쥴러의 타입스텝 리스트에서 t_start이부터의 값을 슬라이싱하여 timesteps에 할당
        #self.scheduler.order = 각 step을 몇번씩 반복해서 denoising할껀가를 해주는 값으로 
        #각 단계에서의 계산이 더욱 정밀하게 이루어지도록 함
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        #특정 시점에서 추론을 시작할 경우 즉 denoising_start가 none이 아닌경우
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )

            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1

            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps

        #timestels = 모든 시점의 리스트(반복포함 즉 [0,0,1,1,2,2,3,3,])이고
        #num_inference_steps= 총 몇단계로 추론할건지 즉 위의 예로 4단계로 추론하는것.(반복제외)
        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline._get_add_time_ids
    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        aesthetic_score,
        negative_aesthetic_score,
        negative_original_size,
        negative_crops_coords_top_left,
        negative_target_size,
        dtype,
        text_encoder_projection_dim=None,
    ):
        if self.config.requires_aesthetics_score:
            add_time_ids = list(original_size + crops_coords_top_left + (aesthetic_score,))
            add_neg_time_ids = list(
                negative_original_size + negative_crops_coords_top_left + (negative_aesthetic_score,)
            )
        else:
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_neg_time_ids = list(negative_original_size + crops_coords_top_left + negative_target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if (
            expected_add_embed_dim > passed_add_embed_dim
            and (expected_add_embed_dim - passed_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to enable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=True)` to make sure `aesthetic_score` {aesthetic_score} and `negative_aesthetic_score` {negative_aesthetic_score} is correctly used by the model."
            )
        elif (
            expected_add_embed_dim < passed_add_embed_dim
            and (passed_add_embed_dim - expected_add_embed_dim) == self.unet.config.addition_time_embed_dim
        ):
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. Please make sure to disable `requires_aesthetics_score` with `pipe.register_to_config(requires_aesthetics_score=False)` to make sure `target_size` {target_size} is correctly used by the model."
            )
        elif expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_neg_time_ids = torch.tensor([add_neg_time_ids], dtype=dtype)

        return add_time_ids, add_neg_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    # 현재의 data type을 저장한 상태로, VAE 전체를 float로 변경하여 메모리 사용량 최적화하는 함수. Attention 블록에 대한 최적화 함수라고 생각하면 될듯?
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    # Unet에서의 Feature Saturation 문제를 해결하기 위한 FreeU 구조를 활성화시키는 함수
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    # Unet에서의 Feature Saturation 문제를 해결하기 위한 FreeU 구조를 비활성화시키는 함수
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.fuse_qkv_projections
    # Attention Module에서 Query, Key, Value를 하나로 뭉쳐서 사용하게 된다.
    def fuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.
        """
        self.fusing_unet = False
        self.fusing_vae = False

        if unet:
            self.fusing_unet = True
            self.unet.fuse_qkv_projections()
            self.unet.set_attn_processor(FusedAttnProcessor2_0())

        if vae:
            if not isinstance(self.vae, AutoencoderKL):
                raise ValueError("`fuse_qkv_projections()` is only supported for the VAE of type `AutoencoderKL`.")

            self.fusing_vae = True
            self.vae.fuse_qkv_projections()
            self.vae.set_attn_processor(FusedAttnProcessor2_0())

    # Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl.StableDiffusionXLPipeline.unfuse_qkv_projections
    # Attention Module에서 뭉쳐 놓은 Query, Key, Value 쌍에 대해서 다시 각각 사용하기 위해 합친 것을 풀어주는 함수.
    def unfuse_qkv_projections(self, unet: bool = True, vae: bool = True):
        """Disable QKV projection fusion if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        Args:
            unet (`bool`, defaults to `True`): To apply fusion on the UNet.
            vae (`bool`, defaults to `True`): To apply fusion on the VAE.

        """
        if unet:
            if not self.fusing_unet:
                logger.warning("The UNet was not initially fused for QKV projections. Doing nothing.")
            else:
                self.unet.unfuse_qkv_projections()
                self.fusing_unet = False

        if vae:
            if not self.fusing_vae:
                logger.warning("The VAE was not initially fused for QKV projections. Doing nothing.")
            else:
                self.vae.unfuse_qkv_projections()
                self.fusing_vae = False

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    # 임베딩 벡터를 생성하기 위한 함수. Guidance를 생성하기 위해 사용된다.
    # w가 입력 텐서로 사용 / embedding_dim, dtype은 생성되는 임베딩 벡터에 대한 설명이다.
    # 결과는 이를 통해서 생성된 임베딩 벡터. 여기에는 timesteps가 같이 포함되어 반환된다.
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def denoising_start(self):
        return self._denoising_start

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        mask_image: PipelineImageInput = None,
        masked_image_latents: torch.FloatTensor = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 0.9999,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        cloth =None,
        pose_img = None,
        text_embeds_cloth=None,
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        pooled_prompt_embeds_c=None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `image`. White pixels in the mask will be
                repainted, while black pixels will be preserved. If `mask_image` is a PIL image, it will be converted
                to a single channel (luminance) before use. If it's a tensor, it should contain one color channel (L)
                instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            padding_mask_crop (`int`, *optional*, defaults to `None`):
                The size of margin in the crop to be applied to the image and masking. If `None`, no crop is applied to image and mask_image. If
                `padding_mask_crop` is not `None`, it will first find a rectangular region with the same aspect ration of the image and
                contains all masked area, and then expand that area based on `padding_mask_crop`. The image and mask_image will then be cropped based on
                the expanded area before resizing to the original image size for inpainting. This is useful when the masked area is small while the image is large
                and contain information inreleant for inpainging, such as background.
            strength (`float`, *optional*, defaults to 0.9999):
                Conceptually, indicates how much to transform the masked portion of the reference `image`. Must be
                between 0 and 1. `image` will be used as a starting point, adding more noise to it the larger the
                `strength`. The number of denoising steps depends on the amount of noise initially added. When
                `strength` is 1, added noise will be maximum and the denoising process will run for the full number of
                iterations specified in `num_inference_steps`. A value of 1, therefore, essentially ignores the masked
                portion of the reference `image`. Note that in the case of `denoising_start` being declared as an
                integer, the value of `strength` will be ignored.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_start (`float`, *optional*):
                When specified, indicates the fraction (between 0.0 and 1.0) of the total denoising process to be
                bypassed before it is initiated. Consequently, the initial part of the denoising process is skipped and
                it is assumed that the passed `image` is a partly denoised image. Note that when this is specified,
                strength will be ignored. The `denoising_start` parameter is particularly beneficial when this pipeline
                is integrated into a "Mixture of Denoisers" multi-pipeline setup, as detailed in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise (ca. final 20% of timesteps still needed) and should be
                denoised by a successor pipeline that has `denoising_start` set to 0.8 so that it only denoises the
                final 20% of the scheduler. The denoising_end parameter should ideally be utilized when this pipeline
                forms a part of a "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output).
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            aesthetic_score (`float`, *optional*, defaults to 6.0):
                Used to simulate an aesthetic score of the generated image by influencing the positive text condition.
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_aesthetic_score (`float`, *optional*, defaults to 2.5):
                Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). Can be used to
                simulate an aesthetic score of the generated image by influencing the negative text condition.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            mask_image,
            height,
            width,
            strength,
            callback_steps,
            output_type,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            padding_mask_crop,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._denoising_start = denoising_start
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. set timesteps
        def denoising_value_valid(dnv):
            return isinstance(self.denoising_end, float) and 0 < dnv < 1

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4:
            # if images are in latent space, we can't mask it
            masked_image = None
        else:
            masked_image = init_image * (mask < 0.5)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        add_noise = True if self.denoising_start is None else False
        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            add_noise=add_noise,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )
        pose_img = pose_img.to(device=device, dtype=prompt_embeds.dtype)

        pose_img = self.vae.encode(pose_img).latent_dist.sample()
        pose_img = pose_img * self.vae.config.scaling_factor

        # pose_img = self._encode_vae_image(pose_img, generator=generator)

        pose_img = (
                torch.cat([pose_img] * 2) if self.do_classifier_free_guidance else pose_img
        )
        cloth = self._encode_vae_image(cloth, generator=generator)

        # # 8. Check that sizes of mask, masked image and latents match
        # if num_channels_unet == 9:
        #     # default case for runwayml/stable-diffusion-inpainting
        #     num_channels_mask = mask.shape[1]
        #     num_channels_masked_image = masked_image_latents.shape[1]
        #     if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
        #         raise ValueError(
        #             f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
        #             f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
        #             f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
        #             f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
        #             " `pipeline.unet` or your `mask_image` or `image` input."
        #         )
        # elif num_channels_unet != 4:
        #     raise ValueError(
        #         f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
        #     )
        # 8.1 Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 10. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        if ip_adapter_image is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )

            #project outside for loop
            image_embeds = self.unet.encoder_hid_proj(image_embeds).to(prompt_embeds.dtype)


        # 11. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        if (
            self.denoising_end is not None
            and self.denoising_start is not None
            and denoising_value_valid(self.denoising_end)
            and denoising_value_valid(self.denoising_start)
            and self.denoising_start >= self.denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {self.denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {self.denoising_end} when using type float."
            )
        elif self.denoising_end is not None and denoising_value_valid(self.denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 11.1 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)



        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                # bsz = mask.shape[0]
                if num_channels_unet == 13:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents,pose_img], dim=1)

                # if num_channels_unet == 9:
                #     latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                # down,reference_features = self.UNet_Encoder(cloth,t, text_embeds_cloth,added_cond_kwargs= {"text_embeds": pooled_prompt_embeds_c, "time_ids": add_time_ids},return_dict=False)
                down,reference_features = self.unet_encoder(cloth,t, text_embeds_cloth,return_dict=False)
                # print(type(reference_features))
                # print(reference_features)
                reference_features = list(reference_features)
                # print(len(reference_features))
                # for elem in reference_features:
                #     print(elem.shape)
                # exit(1)
                if self.do_classifier_free_guidance:
                    reference_features = [torch.cat([torch.zeros_like(d), d]) for d in reference_features]


                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    garment_features=reference_features,
                )[0]
                # noise_pred = self.unet(latent_model_input, t, 
                #                             prompt_embeds,timestep_cond=timestep_cond,cross_attention_kwargs=self.cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs,down_block_additional_attn=down ).sample


                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask

                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )

                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    add_neg_time_ids = callback_outputs.pop("add_neg_time_ids", add_neg_time_ids)
                    mask = callback_outputs.pop("mask", mask)
                    masked_image_latents = callback_outputs.pop("masked_image_latents", masked_image_latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        # else:
        #     return StableDiffusionXLPipelineOutput(images=latents)


        image = self.image_processor.postprocess(image, output_type=output_type)

        if padding_mask_crop is not None:
            image = [self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]

        # Offload all models
        self.maybe_free_model_hooks()

        # if not return_dict:
        return (image,)

        # return StableDiffusionXLPipelineOutput(images=image)