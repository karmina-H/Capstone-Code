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
        unet_encoder: UNet2DConditionModel,#garmentnet 신경망
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
        #masked_image는 masking이미지로 여기서 masked_image의
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

        if masked_image is not None: #masked_image가 존재하나 latent에 없을때
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

    # 입력값의 크기나 좌표등 추가적인 정보를 시간임베딩으로 변환하는 함수 ( 일단보류)
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
        #freeU를 사용하려면 self가 unet을 가지고 있어야함. unet은 주로 이미지 생성 모델의 백본 역할을 하는 신경망
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

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding

    # 임베딩 벡터를 생성하기 위한 함수. Guidance를 생성하기 위해 사용된다.
    # w가 입력 텐서로 사용 / embedding_dim, dtype은 생성되는 임베딩 벡터에 대한 설명이다.
    # 결과는 이를 통해서 생성된 임베딩 벡터. 여기에는 timesteps가 같이 포함되어 반환된다.
    #특정 특성을 강화하거나 감쇠할 수 있도록, w 값에 따라 조정된 포지셔널 임베딩을 생성하는 역할
    #sin과 cos을 사용한 임베딩 생성 방식은 Transformer와 같은 모델에서 사용하는 방식과 유사하여, 시간이나 단계에 따른 정보를 모델이 효과적으로 학습할 수 있게 돕습니다.
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32): #일단보류
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
        assert len(w.shape) == 1#w는 1차원 텐서여야함,  이 함수가 w의 각 값에 대해 임베딩을 생성하는 방식이기 때문에, 배치 형태의 1차원 입력을 기대
        #w를 1000배로 확장하는 단계입니다. 이 확장은 입력 값의 범위를 조절해 임베딩 벡터의 분포를 넓히는 역할을 합니다. 이를 통해 임베딩 벡터가 더 다양한 특성을 표현할 수 있도록 도움
        w = w * 1000.0

        #embedding_dim의 절반(half_dim)을 계산합니다. 이 절반값을 사용하는 이유는 이후 sin과 cos 함수를 적용할 때, 각 함수를 절반씩 사용하여 임베딩 차원을 채우기 위함
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
        #이 모델 텍스트인코더 두개를 사용중인데 
        #prompt만 매개변수로 주면 이 프롬프트하나가 텍스트인코더 두개에 들어감.
        self,
        prompt: Union[str, List[str]] = None, #이거 정의되있지 않은경우 prompt_embeds를 직접 넣어주어야함.
        prompt_2: Optional[Union[str, List[str]]] = None, #text_encoder_2에 전달될 프롬프트
        image: PipelineImageInput = None, #이미지 인페인팅(inpainting)**을 위한 이미지 = 원본이미지
        mask_image: PipelineImageInput = None, #마스크된이미지(마스크된 부분은 흰색으로 마스크해야됨)
        masked_image_latents: torch.FloatTensor = None,#마스크된 이미지의 latent구할껀데 그거 저장할 변수
        height: Optional[int] = None,#생성할 이미지의 높이로 기본적으로 1024로 설정함 근데 512이하로 설정하면 성능이 저하됨
        width: Optional[int] = None,# 높이와 마찬가지로 보통 1024 픽셀로 설정
        padding_mask_crop: Optional[int] = None,#마스크 영역이 작고, 배경이 포함된 큰 이미지를 사용할 때 유용(우리는 이거 안쓸거 같음)
        strength: float = 0.9999,#diffusion모델의 노이즈 강도
        num_inference_steps: int = 50,#denoising단계수로 기본값은 50, 이게 클수록 품질이 좋아지지만 시간이 더 많이걸림
        timesteps: List[int] = None,# 커스텀 디노이징 타임스텝으로 num_inference_steps에 따라 타입스텝이 설정되지만 사용자가 이거로 직접 정의가능
        denoising_start: Optional[float] = None,#전체 디노이징 프로세스 중 얼마나 많은 부분을 생략할지 설정하는 값. 이미지가 부분적으로 디노이지된 상태로 간주될때 사용됨(Mixture of Denoisers설정에서 유용하게ㅐ 쓰인다고함)
        denoising_end: Optional[float] = None,#denoising을 미리 종료하여 최종 결과물이 약간의 노이즈를 남겨 후속 파이프라인이 이를 제거할 수 있도록 하는 설정
        guidance_scale: float = 7.5,#가이던스 스케일 값으로 값이 클수록 텍스트 프롬프트에 더 강하게 반응함.
        negative_prompt: Optional[Union[str, List[str]]] = None,#생성할 반대의 프롬프트
        negative_prompt_2: Optional[Union[str, List[str]]] = None,#두번째 인코더에 들어갈 생성할 반대의 프롬프트
        num_images_per_prompt: Optional[int] = 1,#프롬프트당 이미지수로 프롬프트한개에 이미지 한개생성하니까 1로고정
        eta: float = 0.0,#DDIM스케줄러에서 사용할 파라미터, 노이즈조절에 사용됨.
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,#SAMPLING할때 쓸 난수생성기라고함
        latents: Optional[torch.FloatTensor] = None,#이미지 생성 과정에서 사용할 미리 생성된 노이즈 텐서로 이게 초기 노이즈상태임 이게 안주어지면 GENERATOR를 이용해서 랜덤한 노이즈텐서를 생성함.
        prompt_embeds: Optional[torch.FloatTensor] = None,#각 기본 텐서들에대한 임베딩버전들.
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,#입력 이미지의 스타일이나 색감을 생성 이미지에 반영할 때 사용됨
        output_type: Optional[str] = "pil",#생성된 이미지의 출력 형식으로, PIL.Image.Image 또는 np.array 중에서 선택(기본값은 pil)
        cloth =None, #옷 사진
        pose_img = None,#사람이미지에 대한 pose이미지
        text_embeds_cloth=None,#옷에대한 설명 프롬프트(임베딩버전)
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,#AttentionProcessor에 전달할 추가 인수
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,#이미지의 원본크기(기본값은 1024*1024)
        crops_coords_top_left: Tuple[int, int] = (0, 0),#이미지의 왼쪽 상단을 크롭위치의 처음으로 설정
        target_size: Tuple[int, int] = None,#생성할 이미지의 최종크기로 기본값 = 1024*1024
        negative_original_size: Optional[Tuple[int, int]] = None,#
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,#미학점수? 제공하는건데 이거 일단 분석보류함
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,#CLIP에서 프롬프트 임베딩을 계산할 때 스킵할 레이어 수
        pooled_prompt_embeds_c=None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,#각 디노이징 단계가 끝날 때 호출되는 함수입니다. 각 단계에서 특정 작업을 수행할 수 있습니다.
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],#callback_on_step_end 함수에 전달할 텐서 입력 목록
        **kwargs,
    ):

        #기존에 callback썼던거같음 지금은 안써서 사용하면 에러 출력하는부분
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
        #unet = 디노이징을 수행하는 신경망
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. input값들이 제대로 들어왔는지 체크하고 self로 선언
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

        # 2. 프롬프트의 형태에 따라 batchsize를 결정됨.(프롬프트당 이미지 한개생성되니까)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. 프롬프트를 인코딩하는 부분.
        #텍스트 인코더의 lora_scale의 값을 가져오는부분
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        #encode_prompt함수를 이용해서 프롬프트임베딩을 생성
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

        #타임스텝과 추론 단계 수를 계산
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        #retrieve_timesteps에서 얻은 타임스텝을 기반으로 디노이징 과정을 조정함 strength이용해서 강도조정.
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps,
            strength,
            device,
            denoising_start=self.denoising_start if denoising_value_valid else None,
        )
        #inference steps is not < 1 면 말이안되니까 에러발생시킴
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        #이미지가 여러장 들어오면 각 이미지에 대해서 동일한 time_step을 적용하도록 하는건데 idm-vton은 1장의 이미지고 이미지가 1개씩 들어오니까 여기 1이됨.
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        #strength가 1.0이면 True가 되어, 초기 잠재 변수를 순수한 노이즈로 초기화하는것. 즉 이미지의 기존 특성을 모두 제거하고 새로운 노이즈에서 시작하도록 설정하는거
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        #사용자가 마진을 지정한 경우 즉 padding_mask_crop이 true인경우
        if padding_mask_crop is not None:
            #get_crop_region를 이용해서 이미지에서 크롭영역을 계산함.
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else: #padding_mask_crop가 None이면 크롭 안함
            crops_coords = None
            resize_mode = "default"
        #original_image에 사람사진 저장해서 원본을 저장함.
        original_image = image
        init_image = self.image_processor.preprocess(#이미지를 모델의 입력형식에 맞춰 전처리하는거로 height와 width로 크기조정하고 이미지의 크롭과 resizing을 수행함.
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        #전처리한 이미지의 데이터타입을 float32이렇게 맞춤.
        init_image = init_image.to(dtype=torch.float32)
        
        #mask = mask_image를 모델의 입력형식에 맞춰 전처리하는거로 height와 width로 크기조정하고 이미지의 크롭과 resizing을 수행함.
        mask = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        #mask_image = 잠재공간변환전 마스킹이미지
        #masked_image = 잠재공간 변환후 마스킹이미지
        if masked_image_latents is not None:
            masked_image = masked_image_latents
        elif init_image.shape[1] == 4: #잠재공간에서 일반적으로 4개의 채널을 사용하니까 잠재공간에 있는 이미지를 확인하는조건.
            #init_image가 잠재공간에 있으면 마스킹 못하니까 건너뜀. 근데 위에코드보면 그럴리 없을거 같음.
            masked_image = None
        else:#masked_image_latents가 제공되지 않았고 init_image가 잠재 공간이 아닌 경우
            #mask < 0.5 조건을 사용해 마스크가 검은색(값이 0에 가까운 영역)을 선택
            #(mask < 0.5)는 마스킹하지 않을 부분(검은색 영역)에서는 True 값을 반환하며, 흰색 영역에서는 False를 반환
            masked_image = init_image * (mask < 0.5)

            #이 결과로 생성된 masked_image는 마스킹이 적용된 초기 이미지가 됩니다. 흰색으로 표시된 영역은 덮어씌워져 새로운 이미지로 대체될 준비가 된 상태

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels #잠재공간의 채널수
        num_channels_unet = self.unet.config.in_channels#unet에서의 채널수
        return_image_latents = num_channels_unet == 4 #UNet의 채널 수가 4채널인지 확인하여 return_image_latents에 True 또는 False를 할당'
        #쨌든 unet의 결과를 반환하는거니까 unet채널이 4면 잠재공간에 있는 이미지를 반환하는거고 3이면 실제 사진을 반환하는거

        #self.denoising_start가 None인 경우 add_noise를 True로 설정하여, 디노이징 초기 상태에서 순수한 노이즈로 시작
        add_noise = True if self.denoising_start is None else False

        latents_outputs = self.prepare_latents(#이미지를 잠재공간으로 변환하고 노이즈를 더한 임베딩반환하는 함수 
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

        #latents = 초기입력상태로 완전히 노이즈낀 상태의 잠재공간텐서 ,
        #image_latents = 사람이미지를 잠재공간으로 변환한 텐서
        #noise = latents에 더해준 노이즈
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        #mask = 원본에서 마스킹한 데이터고
        #masked_image = init_image와 mask이미지를 비교해서 mask에서 검은색인 부분을 init_image에서 지운거.
        #negative까지 고려해서 크기를 다시 생성한 mask, 그리고 마스킹이미지를 잠재공간으로 변환한 masked_image_latents
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

        #사람의 pose_img를 인코딩해서 잠재공간으로 변환
        pose_img = self.vae.encode(pose_img).latent_dist.sample()
        pose_img = pose_img * self.vae.config.scaling_factor

        # pose_img = self._encode_vae_image(pose_img, generator=generator)

        #do_classifier_free_guidance사용하면 포즈이미지 2개가 필요하니까 2배로 복제
        pose_img = (
                torch.cat([pose_img] * 2) if self.do_classifier_free_guidance else pose_img
        )
        #옷사진도 잠재공간으로 변환.
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
        #기타 매개변수들 준비
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        height, width = latents.shape[-2:] #잠재변수의 마지막 두차원에서 해당 이미지텐서의 높이와 너비를 가져옴
        height = height * self.vae_scale_factor #scaling_factor로 원본 이미지의 크기와 동일하게조정
        width = width * self.vae_scale_factor#scaling_factor로 원본 이미지의 크기와 동일하게조정
        #이 스케일링된 높이와 너비는 VAE에서 잠재 공간을 다시 이미지 공간으로 복원할 때 사용

        #original_size**와 **target_size**가 미리 설정되지 않았다면, 방금 계산한 height와 width 값을 사용
        original_size = original_size or (height, width) #original_size: 원본 이미지의 크기를 지정하는 매개변수
        target_size = target_size or (height, width) #target_size: 생성할 이미지의 목표 크기를 지정

        # 10. Prepare added time ids & embeddings
        #negative_original_size와 negative_target_size가 미리 설정되지 않았다면, 각각 original_size와 target_size의 값을 사용
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size


        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1]) #텍스트 임베딩의 차원 수
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        #여러 정보들을 사용해서 denoising할때 time_step을 조정할때 사용할 변수들 생성
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
        #배치사이즈와 이미지수에맞게 복제해서 동일한 added_timestep을 사용할수있게함.
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        #do_classifier_free_guidance를 사용할 경우에
        #부정프롬프트와 긍정프롬프트를 결합해서 사용하는 dfg할때 사용하려고 이렇게 함
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)#prompt_embeds = 자세한 문장임베딩
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0) #text = 문맥
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            #time_idx도 부정+긍정같이 concat
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        #gpu에 올려주고
        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        #실제 idm-vton에서 ip_adapter_image는 옷사진의 원본이고
        #cloth는 배치차원추가하고 옷사진을 텐서로 만든 데이터
        #결국 같은 옷사진인데 cloth는 배치처리하기 쉽게 차원추가하고 텐서로 만든거.
        if ip_adapter_image is not None:
            #이미지 임베딩생성.
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image, device, batch_size * num_images_per_prompt
            )
            #image_embeds를 **UNet의 encoder_hid_proj**을 통해 프로젝션하여 모델의 입력 형식에 맞게 조정하고 프롬프트임베딩과 연산하니까 데이터타입 동일하게 수정
            image_embeds = self.unet.encoder_hid_proj(image_embeds).to(prompt_embeds.dtype)

        #여기까지 최종만들어진 변수들
        #image_embeds = 사람이미지의 임베딩버전
        #init_image,original_image = 원본이미지를 전처리한거(모델의 입력사이즈에 맞게 크기바꾸고 crop및 resize하고 데이터타입바꾼거) , 원본이미지
        #latents, noise, image_latents = 초기입력상태로 완전히 노이즈낀 상태의 잠재공간텐서 , latents에 더해준 노이즈, 사람이미지를 잠재공간으로 변환한 텐서
        # prompt_embeds = 자세한 문장임베딩
        #add_text_embeds ,add_neg_time_ids,  add_time_ids  = dfg를 위해서 negative랑 positive프롬프트 임베딩을 concat한거하고 added_timiestpe도 두개 합친거, 프롬프트는 문맥집중, 문장세부집중이 있음
        #height, width = 원본이미지 혹은 결과로 만들 이미지의 크기
        #cloth = latent space로 변환한 옷사진
        #pose_img = 사람포즈이미지
        #mask, masked_image_latents = 마스킹된사람이미지와 그 사진을 latent공간으로 변환한 텐서


        # 11. Denoising loop
        #len(timesteps): 전체 디노이징 과정에서 사용할 타임스텝의 수
        #num_inference_steps * self.scheduler.order: 실제 디노이징 단계에서 사용할 타임스텝 수를 계산
        # num_warmup_steps = 디노이징이 본격적으로 시작되기 전에 초기 단계에서 얼마나 많은 워밍업 단계가 필요한지를 정의
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        #denoising start와 end값이 적절한지 check하는 부분.
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
        #denoising_end값이 없거나 유효하지 않을때 denoising_end 값을 계산해서 지정
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
        #디노이징 과정에서 unet은 각 타입스텝의 정보를 입력으로 받아야하는데 이를 위해서 타입스텝을 고유한 임베딩으로 변환해서 모델에 제공함
        #그게 timestep_cond이고 proj_dim는 그 임베딩의 차원
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            #가이던스 스케일을 임베딩하여 디바이스와 데이터 타입에 맞게 변환
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)



        self._num_timetseps = len(timesteps) #: timesteps의 길이를 _num_timesteps에 저장
        with self.progress_bar(total=num_inference_steps) as progress_bar: #타입스텝마다 denoising반복하면서 progress_bar로 진행 상황을 표시
            for i, t in enumerate(timesteps):
                #interrupt가 True가 되면 denoising을 stop
                if self.interrupt:
                    continue
                # expand the latents if we are doing classifier free guidance
                #do_classifier_free_guidance가 True일 경우, latents를 두 배로 확장하여 텍스트와 비텍스트 조건 모두를 포함
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # concat latents, mask, masked_image_latents in the channel dimension
                #latent_model_input을 타임스텝 t에 맞춰 스케일링
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)


                # bsz = mask.shape[0]
                #UNet 채널 수가 13인 경우, mask, masked_image_latents, pose_img를 채널 차원에 결합하여 latent_model_input에 추가 정보를 제공
                if num_channels_unet == 13:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents,pose_img], dim=1)

                # if num_channels_unet == 9:
                #     latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

                # predict the noise residual

                #text와 tim_idx를 추가조건으로 설정
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                # IP 어댑터 이미지가 있는 경우 image_embeds도 조건에 추가하여, 추가 이미지 정보도 반영
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                # down,reference_features = self.UNet_Encoder(cloth,t, text_embeds_cloth,added_cond_kwargs= {"text_embeds": pooled_prompt_embeds_c, "time_ids": add_time_ids},return_dict=False)
                #옷에 대한 텍스트임베딩과 옷그리고 현재 타입스텝 t를 가지고 unet에서 인코딩을 수행한뒤 reference_features를 추출
                down,reference_features = self.unet_encoder(cloth,t, text_embeds_cloth,return_dict=False)
                # print(type(reference_features))
                # print(reference_features)

                #reference_features를 리스트화
                reference_features = list(reference_features)
                # print(len(reference_features))
                # for elem in reference_features:
                #     print(elem.shape)
                # exit(1)

                if self.do_classifier_free_guidance:
                    #인코딩해서 생성된 추출된특징의텐서와 그 텐서와크기가같지만 모두 0으로 채워진텐서를 생성해서 concat해줌.(cfg를 사용하기위해서)
                    reference_features = [torch.cat([torch.zeros_like(d), d]) for d in reference_features]

                #UNet을 사용해 현재 타임스텝에서 노이즈를 예측하고, 여러 조건을 반영하여 noise_pred를 생성
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    garment_features=reference_features, #garment net에서 생성된 feature를 통해서 tryonnet에 여기서 전달
                )[0]
                # noise_pred = self.unet(latent_model_input, t, 
                #                             prompt_embeds,timestep_cond=timestep_cond,cross_attention_kwargs=self.cross_attention_kwargs,added_cond_kwargs=added_cond_kwargs,down_block_additional_attn=down ).sample


                # perform guidance
                #cfg사용하면
                if self.do_classifier_free_guidance:
                    #noise_pred를 두 개로 분리하여 텍스트 조건이 없는 예측(noise_pred_uncond)과 텍스트 조건이 있는 예측(noise_pred_text)으로 나누어줌
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    #이들 간의 차이에 guidance_scale을 곱하여 조정한 후, 이를 noise_pred로 설정해 최종 예측 노이즈를 반영
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                #guidance_rescale가 존재하면 guidance_rescale만큼 텍스트 조건의 강도를 더 많이 반영하게함
                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1 (여기서 t시점에서 t-1시점의 노이즈를 예측하는거)
                #scheduler.step을 사용해 현재 t 타임스텝에서 예측한 noise_pred로 t-1번째 잠재 변수를 업데이트
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                #unet의 잠재변수채널이 4면 그대로 잠재공간의 이미지에 노이즈더한거로 저장
                if num_channels_unet == 4:
                    init_latents_proper = image_latents #image_latents는 디노이징이 완료되었을 때 참조할 초기 이미지 잠재 벡터 
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2) #mask는 positive와 negative를 고려해서 복제된 두개의 마스킹 이미지이므로 chunk2해서 하나만 가져오는거
                    else:
                        init_mask = mask
                    #현재 타임스텝이 마지막이 아닌 경우, 다음 타임스텝(timesteps[i + 1])을 가져와 init_latents_proper에 노이즈를 추가
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        #다음 타임스텝에서 디노이징을 시작할 때 필요한 노이즈가 미리 포함된 초기 잠재 변수init_latents_proper를 준비하는거
                        #그니까 여기서 다음 step에서 사용될 초기잠재변수를 지금 예측한 노이즈를 통해서 구하는거
                        init_latents_proper = self.scheduler.add_noise(
                            init_latents_proper, noise, torch.tensor([noise_timestep])
                        )
                    #1 - init_mask와 init_mask를 통해 마스킹되지 않은 부분과 마스킹된 부분을 나누고\
                    #마스크가 적용되지 않은 부분은 init_latents_proper의 값을 유지 즉 원본잠재벡터에 가까운 상태를 반영하는거고
                    #마스킹된 부분에는 latents값을 적용해서 마스킹된 부분에 디노이징의 결과가 반영되는거
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                    
                #callback_on_step_end이 정의되어 있다면 해당 타입스텝에서의 모든 변수들을 담아서 저장
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

                # call the callback, if provided - 진행바 업데이트
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE: #TPU 환경에서의 단계 표시
                    xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            #결과가 latent가 아닌 경우에만 이미지를 디코딩함.

            #VAE의 데이터 타입이 float16이고 force_upcast가 설정되어 있으면, needs_upcasting을 True로 설정
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                #self.upcast_vae()를 호출하여 VAE를 float32로 변환
                self.upcast_vae()
                #latents 텐서도 VAE와 같은 float32 형식으로 변환
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            #latent를 디코딩해서 이미지가져옴.
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed - float16으로 다시 다운캐스팅
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        # else:
        #     return StableDiffusionXLPipelineOutput(images=latents)

        #이미지 후처리작업
        image = self.image_processor.postprocess(image, output_type=output_type)

        #crop안하니까 필요 x
        if padding_mask_crop is not None:
            image = [self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]

        # 모델과 관련된 리소스를 해제
        self.maybe_free_model_hooks()

        # 필요에 따라 다중 이미지를 반환할 수 있도록, 하나의 이미지여도 튜플 형태로 감싸서 반환
        return (image,)
    

        # return StableDiffusionXLPipelineOutput(images=image)