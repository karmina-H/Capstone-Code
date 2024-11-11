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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from einops import rearrange

from diffusers.models.embeddings import (
    GaussianFourierProjection,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PositionNet,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from src.unet_block_hacked_garmnet import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UNetMidBlock2DSimpleCrossAttn,
    get_down_block,
    get_up_block,
)
from diffusers.models.resnet import Downsample2D, FirDownsample2D, FirUpsample2D, KDownsample2D, KUpsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModel


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    The output of [`UNet2DConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNet2DConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    """
    조건부 2D Unet Model(앞에 * 붙은 것만 필수)

    *sample_size (int or Tuple[int, int] | 기본값: (None))
        입출력 샘플의 높이와 너비.

    in_channels (int | 기본값: (4))
        입력 샘플의 채널 수.

    out_channels (int | 기본값: (4))
        출력 샘플의 채널 수.

    center_input_sample (bool | 기본값: (False))
        입력 샘플을 중심에 맞출지 여부.

    flip_sin_to_cos (bool | 기본값: (False))
        타임 임베딩에서 sin을 cos으로 전환할지 여부.

    freq_shift (int | 기본값: (0))
        타임 임베딩에 적용할 주파수 이동.

    down_block_types (Tuple[str] | 기본값: ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"))
        사용할 다운샘플 블록의 유형.

    mid_block_type (str | 기본값: ("UNetMidBlock2DCrossAttn"))
        UNet 중간 블록의 유형.

    up_block_types (Tuple[str] | 기본값: ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"))
        사용할 업샘플 블록의 유형.

    only_cross_attention (bool or Tuple[bool] | 기본값: (False))
        기본 transformer 블록에 셀프 어텐션을 포함할지 여부.

    block_out_channels (Tuple[int] | 기본값: (320, 640, 1280, 1280))
        각 블록의 출력 채널 수.

    layers_per_block (int | 기본값: (2))
        각 블록당 레이어 수.

    downsample_padding (int | 기본값: (1))
        다운샘플링 컨볼루션에 사용할 패딩.

    mid_block_scale_factor (float | 기본값: (1.0))
        중간 블록에 사용할 스케일 팩터.

    dropout (float | 기본값: (0.0))
        드롭아웃 확률.

    act_fn (str | 기본값: ("silu"))
        사용할 활성화 함수.

    norm_num_groups (int | 기본값: (32))
        정규화에 사용할 그룹 수.

    norm_eps (float | 기본값: (1e-5))
        정규화에 사용할 epsilon 값.

    cross_attention_dim (int or Tuple[int] | 기본값: (1280))
        크로스 어텐션 피처의 차원.

    transformer_layers_per_block (int, Tuple[int], or Tuple[Tuple] | 기본값: (1))
        각 블록당 BasicTransformerBlock의 수.

    reverse_transformer_layers_per_block (Tuple[Tuple] | 기본값: (None))
        업샘플링 블록의 BasicTransformerBlock 수.

    encoder_hid_dim (int | 기본값: (None))
        encoder_hidden_states가 cross_attention_dim으로 프로젝션될 때 사용할 차원.

    encoder_hid_dim_type (str | 기본값: (None))
        encoder_hidden_states가 cross_attention_dim으로 다운 프로젝션되는 방법.

    attention_head_dim (int | 기본값: (8))
        어텐션 헤드의 차원.

    num_attention_heads (int | 기본값: (None))
        어텐션 헤드 수. 정의되지 않으면 attention_head_dim이 기본값으로 사용.

    resnet_time_scale_shift (str | 기본값: ("default"))
        ResNet 블록의 타임 스케일 시프트 설정.

    class_embed_type (str | 기본값: (None))
        클래스 임베딩의 유형.

    addition_embed_type (str | 기본값: (None))
        타임 임베딩에 추가될 임베딩 유형.

    addition_time_embed_dim (int | 기본값: (None))
        타임스텝 임베딩의 차원.

    num_class_embeds (int | 기본값: (None))
        클래스 레이블 조건화 시 학습 가능한 임베딩 행렬의 입력 차원.

    time_embedding_type (str | 기본값: ("positional"))
        타임스텝에 사용할 포지션 임베딩의 유형.

    time_embedding_dim (int | 기본값: (None))
        프로젝션된 타임 임베딩의 차원을 재정의하는 옵션.

    time_embedding_act_fn (str | 기본값: (None))
        타임 임베딩에 한 번만 사용할 선택적 활성화 함수.

    timestep_post_act (str | 기본값: (None))
        타임스텝 임베딩에서 두 번째로 사용할 활성화 함수.

    time_cond_proj_dim (int | 기본값: (None))
        타임스텝 임베딩의 cond_proj 레이어의 차원.

    conv_in_kernel (int | 기본값: (3))
        conv_in 레이어의 커널 크기.

    conv_out_kernel (int | 기본값: (3))
        conv_out 레이어의 커널 크기.

    projection_class_embeddings_input_dim (int | 기본값: (None))
        class_embed_type="projection"일 때 class_labels 입력의 차원.

    class_embeddings_concat (bool | 기본값: (False))
        타임 임베딩과 클래스 임베딩을 연결할지 여부.

    mid_block_only_cross_attention (bool | 기본값: (None))
        UNetMidBlock2DSimpleCrossAttn을 사용할 때 중간 블록에서 크로스 어텐션을 사용할지 여부.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: Union[int, Tuple[int]] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: Union[int, Tuple[int]] = 1280,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: int = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        time_embedding_act_fn: Optional[str] = None,
        timestep_post_act: Optional[str] = None,
        time_cond_proj_dim: Optional[int] = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: Optional[int] = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: Optional[bool] = None,
        cross_attention_norm: Optional[str] = None,
        addition_embed_type_num_heads=64,
    ):
        super().__init__()

        self.sample_size = sample_size
        # 따로 설정해서는 못쓴다고 하네요
        if num_attention_heads is not None:
            raise ValueError(
                "At the moment it is not possible to define the number of attention heads via `num_attention_heads` because of a naming issue as described in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131. Passing `num_attention_heads` will only be supported in diffusers v0.19."
            )

        # If `num_attention_heads` is not defined (which is the case for most models)
        # it will default to `attention_head_dim`. This looks weird upon first reading it and it is.
        # The reason for this behavior is to correct for incorrectly named variables that were introduced
        # when this library was created. The incorrect naming was only discovered much later in https://github.com/huggingface/diffusers/issues/2011#issuecomment-1547958131
        # Changing `attention_head_dim` to `num_attention_heads` for 40,000+ configurations is too backwards breaking
        # which is why we correct for the naming here.
        num_attention_heads = num_attention_heads or attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(only_cross_attention, bool) and len(only_cross_attention) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `only_cross_attention` as `down_block_types`. `only_cross_attention`: {only_cross_attention}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(attention_head_dim, int) and len(attention_head_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `attention_head_dim` as `down_block_types`. `attention_head_dim`: {attention_head_dim}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )
        if isinstance(transformer_layers_per_block, list) and reverse_transformer_layers_per_block is None:
            for layer_number_per_block in transformer_layers_per_block:
                if isinstance(layer_number_per_block, list):
                    raise ValueError("Must provide 'reverse_transformer_layers_per_block` if using asymmetrical UNet.")

        # input
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=conv_in_kernel, padding=conv_in_padding
        )

        # time
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}.")
            self.time_proj = GaussianFourierProjection(
                time_embed_dim // 2, set_W_to_weight=False, log=False, flip_sin_to_cos=flip_sin_to_cos
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        self.time_embedding = TimestepEmbedding(
            timestep_input_dim,
            time_embed_dim,
            act_fn=act_fn,
            post_act_fn=timestep_post_act,
            cond_proj_dim=time_cond_proj_dim,
        )

        if encoder_hid_dim_type is None and encoder_hid_dim is not None:
            encoder_hid_dim_type = "text_proj"
            self.register_to_config(encoder_hid_dim_type=encoder_hid_dim_type)
            logger.info("encoder_hid_dim_type defaults to 'text_proj' as `encoder_hid_dim` is defined.")

        if encoder_hid_dim is None and encoder_hid_dim_type is not None:
            raise ValueError(
                f"`encoder_hid_dim` has to be defined when `encoder_hid_dim_type` is set to {encoder_hid_dim_type}."
            )

        if encoder_hid_dim_type == "text_proj":
            self.encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim)
        elif encoder_hid_dim_type == "text_image_proj":
            # image_embed_dim DOESN'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image_proj"` (Kadinsky 2.1)`
            self.encoder_hid_proj = TextImageProjection(
                text_embed_dim=encoder_hid_dim,
                image_embed_dim=cross_attention_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type == "image_proj":
            # Kandinsky 2.2
            self.encoder_hid_proj = ImageProjection(
                image_embed_dim=encoder_hid_dim,
                cross_attention_dim=cross_attention_dim,
            )
        elif encoder_hid_dim_type is not None:
            raise ValueError(
                f"encoder_hid_dim_type: {encoder_hid_dim_type} must be None, 'text_proj' or 'text_image_proj'."
            )
        else:
            self.encoder_hid_proj = None

        # 클래스 임베딩 - 임베딩 타입에 따른 코드들
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'projection' requires `projection_class_embeddings_input_dim` be set"
                )
            # The projection `class_embed_type` is the same as the timestep `class_embed_type` except
            # 1. the `class_labels` inputs are not first converted to sinusoidal embeddings
            # 2. it projects from an arbitrary input dimension.
            #
            # Note that `TimestepEmbedding` is quite general, being mainly linear layers and activations.
            # When used for embedding actual timesteps, the timesteps are first converted to sinusoidal embeddings.
            # As a result, `TimestepEmbedding` can be passed arbitrary vectors.
            self.class_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif class_embed_type == "simple_projection":
            if projection_class_embeddings_input_dim is None:
                raise ValueError(
                    "`class_embed_type`: 'simple_projection' requires `projection_class_embeddings_input_dim` be set"
                )
            self.class_embedding = nn.Linear(projection_class_embeddings_input_dim, time_embed_dim)
        else:
            self.class_embedding = None


        #여기서부터는 Addition Embeding Type에 관한 이야기
        if addition_embed_type == "text":
            if encoder_hid_dim is not None:
                text_time_embedding_from_dim = encoder_hid_dim
            else:
                text_time_embedding_from_dim = cross_attention_dim

            self.add_embedding = TextTimeEmbedding(
                text_time_embedding_from_dim, time_embed_dim, num_heads=addition_embed_type_num_heads
            )
        elif addition_embed_type == "text_image":
            # text_embed_dim and image_embed_dim DON'T have to be `cross_attention_dim`. To not clutter the __init__ too much
            # they are set to `cross_attention_dim` here as this is exactly the required dimension for the currently only use
            # case when `addition_embed_type == "text_image"` (Kadinsky 2.1)`
            self.add_embedding = TextImageTimeEmbedding(
                text_embed_dim=cross_attention_dim, image_embed_dim=cross_attention_dim, time_embed_dim=time_embed_dim
            )
        elif addition_embed_type == "text_time":
            self.add_time_proj = Timesteps(addition_time_embed_dim, flip_sin_to_cos, freq_shift)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)
        elif addition_embed_type == "image":
            # Kandinsky 2.2
            self.add_embedding = ImageTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type == "image_hint":
            # Kandinsky 2.2 ControlNet
            self.add_embedding = ImageHintTimeEmbedding(image_embed_dim=encoder_hid_dim, time_embed_dim=time_embed_dim)
        elif addition_embed_type is not None:
            raise ValueError(f"addition_embed_type: {addition_embed_type} must be None, 'text' or 'text_image'.")

        if time_embedding_act_fn is None:
            self.time_embed_act = None
        else:
            self.time_embed_act = get_activation(time_embedding_act_fn)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            if mid_block_only_cross_attention is None:
                mid_block_only_cross_attention = only_cross_attention

            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if mid_block_only_cross_attention is None:
            mid_block_only_cross_attention = False

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)
        if class_embeddings_concat:
            # The time embeddings are concatenated with the class embeddings. The dimension of the
            # time embeddings passed to the down, middle, and up blocks is twice the dimension of the
            # regular time embeddings
            blocks_time_embed_dim = time_embed_dim * 2
        else:
            blocks_time_embed_dim = time_embed_dim

        # Down 블럭드렝 대한 코드 -> Unet에서 축소과정
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # 중간 (MidBlock)
        if mid_block_type == "UNetMidBlock2DCrossAttn":
            self.mid_block = UNetMidBlock2DCrossAttn(
                transformer_layers_per_block=transformer_layers_per_block[-1],
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim[-1],
                num_attention_heads=num_attention_heads[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
                attention_type=attention_type,
            )
        elif mid_block_type == "UNetMidBlock2DSimpleCrossAttn":
            self.mid_block = UNetMidBlock2DSimpleCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                cross_attention_dim=cross_attention_dim[-1],
                attention_head_dim=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                skip_time_act=resnet_skip_time_act,
                only_cross_attention=mid_block_only_cross_attention,
                cross_attention_norm=cross_attention_norm,
            )
        elif mid_block_type == "UNetMidBlock2D":
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=blocks_time_embed_dim,
                dropout=dropout,
                num_layers=0,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=resnet_time_scale_shift,
                add_attention=False,
            )
        elif mid_block_type is None:
            self.mid_block = None
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # 바로 다시 Upconv
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = (
            list(reversed(transformer_layers_per_block))
            if reverse_transformer_layers_per_block is None
            else reverse_transformer_layers_per_block
        )
        only_cross_attention = list(reversed(only_cross_attention))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False
            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resolution_idx=i,
                resnet_groups=norm_num_groups,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_type=attention_type,
                resnet_skip_time_act=resnet_skip_time_act,
                resnet_out_scale_factor=resnet_out_scale_factor,
                cross_attention_norm=cross_attention_norm,
                attention_head_dim=attention_head_dim[i] if attention_head_dim[i] is not None else output_channel,
                dropout=dropout,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel




        # encode_output_chs = [
        #     # 320,
        #     # 320,
        #     # 320,
        #     1280, 
        #     1280, 
        #     1280, 
        #     1280,
        #     640,
        #     640
        # ]

        # encode_output_chs2 = [
        #     # 320,
        #     # 320,
        #     # 320,
        #     1280, 
        #     1280,
        #     640, 
        #     640, 
        #     640,
        #     320
        # ]

        # encode_num_head_chs3 = [
        #     # 5,
        #     # 5,
        #     # 10,
        #     20,
        #     20, 
        #     20,
        #     10,
        #     10, 
        #     10 
        # ]


        # encode_num_layers_chs4 = [
        #     # 1,
        #     # 1,
        #     # 2,
        #     10,
        #     10, 
        #     10,
        #     2,
        #     2, 
        #     2 
        # ]


        # self.warp_blks = nn.ModuleList([])
        # self.warp_zeros = nn.ModuleList([])

        # for in_ch, cont_ch,num_head,num_layers in zip(encode_output_chs, encode_output_chs2,encode_num_head_chs3,encode_num_layers_chs4):
        #     # dim_head = in_ch // self.num_heads
        #     # dim_head = dim_head // dim_head_denorm

        #     self.warp_blks.append(Transformer2DModel(
        #     num_attention_heads=num_head,
        #     attention_head_dim=64,
        #     in_channels=in_ch,
        #     num_layers = num_layers,
        #     cross_attention_dim = cont_ch,
        #     ))
            
        #     self.warp_zeros.append(zero_module(nn.Conv2d(in_ch, in_ch, 1, padding=0)))



        # Unet에 대한 전체 Output
        if norm_num_groups is not None:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps
            )

            self.conv_act = get_activation(act_fn)

        else:
            self.conv_norm_out = None
            self.conv_act = None

        conv_out_padding = (conv_out_kernel - 1) // 2
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=conv_out_kernel, padding=conv_out_padding
        )

        if attention_type in ["gated", "gated-text-image"]:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, tuple) or isinstance(cross_attention_dim, list):
                positive_len = cross_attention_dim[0]

            feature_type = "text-only" if attention_type == "gated" else "text-image"
            self.position_net = PositionNet(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )



    
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        """
        Returns:
            `dict` of attention processors: 모델에서 사용된 모든 attention 프로세서를 반환하며, weight 이름으로 인덱싱되어 있음.
        """
        # 재귀적으로 설정
        processors = {}

        # 프로세서를 재귀적으로 추가하는 함수
        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            # 서브 모듈을 순회하며 프로세서 추가
            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        # 최상위 모듈들을 순회하며 프로세서 설정
        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # attention processor 설정 함수
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        """
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                각 attention 레이어에 대해 사용될 processor 클래스 설정. 딕셔너리를 사용하는 경우, 경로에 맞춰 cross attention 프로세서를 지정해야 함.
        """
        count = len(self.attn_processors.keys())

        # 프로세서의 수가 맞지 않는 경우 오류 발생
        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        # 재귀적으로 attention processor를 설정하는 함수
        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            # 서브 모듈 순회
            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        # 최상위 모듈들을 순회하며 프로세서 설정
        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # 기본 attention processor 설정 함수
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        기본 attention 프로세서로 설정.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    # sliced attention 설정 함수
    def set_attention_slice(self, slice_size):
        """
        Enable sliced attention computation.

        입력 텐서를 슬라이스하여 메모리를 절약하며 주의 계산을 수행할 수 있도록 함.
        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to "auto"):
                슬라이스 크기 설정에 따라 메모리 사용량을 조정할 수 있음.
        """
        sliceable_head_dims = []

        # 슬라이스 가능한 차원을 재귀적으로 수집하는 함수
        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # attention 레이어 수집
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        # 슬라이스 크기 설정
        if slice_size == "auto":
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # 슬라이스 설정을 재귀적으로 설정하는 함수
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    # gradient checkpointing 설정 함수
    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # FreeU 메커니즘 활성화
    def enable_freeu(self, s1, s2, b1, b2):
        """Enables the FreeU mechanism from https://arxiv.org/abs/2309.11497.
        Args:
            s1 (`float`): Stage 1에서 스킵 피처의 기여도를 줄이기 위한 스케일링 팩터.
            s2 (`float`): Stage 2에서 스킵 피처의 기여도를 줄이기 위한 스케일링 팩터.
            b1 (`float`): Stage 1에서 백본 피처의 기여도를 증폭시키기 위한 스케일링 팩터.
            b2 (`float`): Stage 2에서 백본 피처의 기여도를 증폭시키기 위한 스케일링 팩터.
        """
        for i, upsample_block in enumerate(self.up_blocks):
            setattr(upsample_block, "s1", s1)
            setattr(upsample_block, "s2", s2)
            setattr(upsample_block, "b1", b1)
            setattr(upsample_block, "b2", b2)

    # FreeU 메커니즘 비활성화
    def disable_freeu(self):
        """Disables the FreeU mechanism."""
        freeu_keys = {"s1", "s2", "b1", "b2"}
        for i, upsample_block in enumerate(self.up_blocks):
            for k in freeu_keys:
                if hasattr(upsample_block, k) or getattr(upsample_block, k, None) is not None:
                    setattr(upsample_block, k, None)

    # QKV 프로젝션 합성 활성화
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections.

        셀프 어텐션 모듈에 대해 쿼리, 키, 밸류 프로젝션 행렬이 합쳐짐.
        <Tip warning={true}>
        이 API는 🧪 실험적임.
        </Tip>
        """
        self.original_attn_processors = None

        # KV 프로젝션이 추가된 경우 지원되지 않음
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        # 모듈 순회하여 프로젝션 합성
        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # QKV 프로젝션 합성 비활성화
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.
        <Tip warning={true}>
        이 API는 🧪 실험적임.
        </Tip>
        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    # forward 함수
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        [`UNet2DConditionModel`]의 forward 메서드.

        Args:
            sample (`torch.FloatTensor`):
                노이즈가 있는 입력 텐서. (배치, 채널, 높이, 너비) 형태.
            timestep (`torch.FloatTensor` or `float` or `int`): 입력을 디노이즈할 타임스텝.
            encoder_hidden_states (`torch.FloatTensor`):
                인코더 히든 상태 (배치, 시퀀스 길이, 피처 차원).
            class_labels (`torch.Tensor`, *optional*):
                클래스 라벨 임베딩. 타임스텝 임베딩과 더해짐.
            timestep_cond (`torch.Tensor`, *optional*):
                타임스텝 조건 임베딩. 주어진 경우 time_embedding 레이어를 통과한 후 더해짐.
            attention_mask (`torch.Tensor`, *optional*):
                인코더 히든 상태에 적용될 주의 마스크 (배치, 키 토큰).
            cross_attention_kwargs (`dict`, *optional*):
                `AttentionProcessor`에 전달될 추가 매개변수.
            added_cond_kwargs (`dict`, *optional*):
                UNet 블록에 전달될 추가 임베딩.
            down_block_additional_residuals (`tuple` of `torch.Tensor`, *optional*):
                하위 UNet 블록의 잔차 연결에 추가될 텐서.
            mid_block_additional_residual (`torch.Tensor`, *optional*):
                중간 UNet 블록의 잔차에 추가될 텐서.
            encoder_attention_mask (`torch.Tensor`):
                인코더 히든 상태에 적용될 주의 마스크.
            return_dict (`bool`, *optional*, defaults to `True`):
                결과를 딕셔너리 형태로 반환할지 여부 설정.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] 또는 `tuple`: 결과 텐서 반환.
        """
        # 샘플은 기본적으로 업샘플링 레이어의 전체 배수여야 함
        default_overall_up_factor = 2**self.num_upsamplers

        # 업샘플 크기를 포워드해야 하는지 확인
        forward_upsample_size = False
        upsample_size = None

        for dim in sample.shape[-2:]:
            if dim % default_overall_up_factor != 0:
                forward_upsample_size = True
                break

        # attention_mask를 편향으로 변환하여 attention 점수에 추가 가능하도록 변경
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # encoder_attention_mask를 편향으로 변환하여 사용
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 입력 중심화
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 시간 관련 임베딩 계산
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        # 여기서부터는 Addition Embeding Type에 관한 이야기
        # 클래스 임베딩 - 임베딩 타입에 따른 코드들
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)

            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        # 추가된 텍스트 임베딩 처리 - 추가 임베딩 타입에 따른 코드들
        if self.config.addition_embed_type == "text":
            aug_emb = self.add_embedding(encoder_hidden_states)
        elif self.config.addition_embed_type == "text_image":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("text_image 타입에서는 image_embeds가 추가 조건에 필요")

            image_embs = added_cond_kwargs.get("image_embeds")
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs, image_embs)
        elif self.config.addition_embed_type == "text_time":
            if "text_embeds" not in added_cond_kwargs or "time_ids" not in added_cond_kwargs:
                raise ValueError("text_time 타입에서는 text_embeds와 time_ids가 필요")

            text_embeds = added_cond_kwargs.get("text_embeds")
            time_ids = added_cond_kwargs.get("time_ids")
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            if "image_embeds" not in added_cond_kwargs:
                raise ValueError("image 타입에서는 image_embeds가 필요")
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        elif self.config.addition_embed_type == "image_hint":
            if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
                raise ValueError("image_hint 타입에서는 image_embeds와 hint가 필요")

            image_embs = added_cond_kwargs.get("image_embeds")
            hint = added_cond_kwargs.get("hint")
            aug_emb, hint = self.add_embedding(image_embs, hint)
            sample = torch.cat([sample, hint], dim=1)

        emb = emb + aug_emb if aug_emb is not None else emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # 인코더 히든 상태 프로젝션 - 설정에 따른 코드들
        if self.encoder_hid_proj is not None:
            if self.config.encoder_hid_dim_type == "text_proj":
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
            elif self.config.encoder_hid_dim_type == "text_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError("text_image_proj 타입에서는 image_embeds가 필요")
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)
            elif self.config.encoder_hid_dim_type == "image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError("image_proj 타입에서는 image_embeds가 필요")
                image_embeds = added_cond_kwargs.get("image_embeds")
                encoder_hidden_states = self.encoder_hid_proj(image_embeds)
            elif self.config.encoder_hid_dim_type == "ip_image_proj":
                if "image_embeds" not in added_cond_kwargs:
                    raise ValueError("ip_image_proj 타입에서는 image_embeds가 필요")
                image_embeds = added_cond_kwargs.get("image_embeds")
                image_embeds = self.encoder_hid_proj(image_embeds).to(encoder_hidden_states.dtype)
                encoder_hidden_states = torch.cat([encoder_hidden_states, image_embeds], dim=1)

        # 샘플 사전 처리
        sample = self.conv_in(sample)
        garment_features = []

        # GLIGEN 위치 네트워크 관련 설정
        if cross_attention_kwargs is not None and cross_attention_kwargs.get("gligen", None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {"objs": self.position_net(**gligen_args)}

        # down 블록
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if len(down_intrablock_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_intrablock_additional_residuals.pop(0)

                sample, res_samples, out_garment_feat = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
                garment_features += out_garment_feat
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, scale=lora_scale)
                if len(down_intrablock_additional_residuals) > 0:
                    sample += down_intrablock_additional_residuals.pop(0)

            down_block_res_samples += res_samples

        # 컨트롤넷 관련 처리
        if down_block_additional_residuals is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # mid 블록 처리
        if self.mid_block is not None:
            if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
                sample, out_garment_feat = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
                garment_features += out_garment_feat
            else:
                sample = self.mid_block(sample, emb)

            if len(down_intrablock_additional_residuals) > 0 and sample.shape == down_intrablock_additional_residuals[0].shape:
                sample += down_intrablock_additional_residuals.pop(0)

        if mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # up 블록 처리
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, out_garment_feat = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
                garment_features += out_garment_feat

        if not return_dict:
            return (sample,), garment_features

        return UNet2DConditionOutput(sample=sample), garment_features
