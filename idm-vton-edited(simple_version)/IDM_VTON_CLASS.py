import sys
sys.path.append('./')
from PIL import Image
# import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel


from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler,AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


'''
사용방법은 IDM-VTON-CLASS파일 가져와서 import IDM_VTON하고
IDM_VTON객체생성하고 CALL하면됨.
ex->
editclass = IDM_VTON()
edtied_image,masked_image = editclass(매개변수)
'''


#pil이미지를 입력받아서 바이너리 마스크로 변환하는 함수
def pil_to_binary_mask(pil_image, threshold=0):
    #pil이미지를 넘파이배열로 변환
    np_image = np.array(pil_image)
    #grayscale_image는 pil이미지에서 그레이스케일이미지 즉 각 픽셀을 0~255의 단일채널로 바꿈
    grayscale_image = Image.fromarray(np_image).convert("L")
    #그레이스 케일 이미지로 변환된 각 픽셀이 기준값보다 크면 True, 작으면 False로 변환된 binary_mask를 생성
    binary_mask = np.array(grayscale_image) > threshold
    #binary_mask와 동일한 크기의 배열을 생성하고, 모든 값을 0으로 초기화한 mask선언
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    #바이너리 마스크에서 정보추출해서 mask의 마스킹으로 활용 즉 mask = 이미지의 마스크된 정보
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    #mask 값을 255로 확장하여 최종 마스크를 생성하고, 데이터 타입을 uint8로 변환
    mask = (mask*255).astype(np.uint8)
    #mask 배열을 PIL 이미지로 변환하여 output_mask에 저장
    output_mask = Image.fromarray(mask)
    #결과반환
    return output_mask 


#옷사진에서 활용하는 신경망 - UNet2DConditionModel_ref
#tryon - UNet2DConditionModel

#pretrained모델 불러오는 부분
base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')

unet = UNet2DConditionModel.from_pretrained( #tryon net = UNet2DConditionModel
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

# "stabilityai/stable-diffusion-xl-base-1.0",
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained( #garment net = UNet2DConditionModel_ref
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

#tryonPipeline = tryon_pipeline.py에 있는 StableDiffusionXLInpaintPipeline를 의미함 
#그리고 unet은 tryon net을 이미하고 unet_encoder는 garment net을 의미함
pipe = TryonPipeline.from_pretrained(
        base_path,
        unet=unet,
        vae=vae,
        feature_extractor= CLIPImageProcessor(),
        text_encoder = text_encoder_one,
        text_encoder_2 = text_encoder_two,
        tokenizer = tokenizer_one,
        tokenizer_2 = tokenizer_two,
        scheduler = noise_scheduler,
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
)

pipe.unet_encoder = UNet_Encoder
        

#이미지 diffusion 실행하는 함수
def start_tryon(human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = human_img.convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        #모델을 사용해서 마스킹생성
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        #마스크도 이미지와 동일한크기로 만들어줌
        mask = mask.resize((768,1024))
    else:
        #사용자가 직접 마스킹된 이미지를 넣어줄때
        mask = pil_to_binary_mask(masked_img.convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
        #mask를 텐서로 변환하고 반전시킨 후, human_img 텐서와 곱하여 마스크가 적용된 이미지를 생성
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    #텐서를 PIL 이미지로 다시 변환
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    
    #densepose모델불러오기
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    #포즈이미지 생성
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, inaccurate"
                with torch.inference_mode():
                    #pipe에 있는encode_prompt함수이용해서 프롬프트임베딩생성
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    #옷사진 설명에 대한 프롬프트 임베딩생성.
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )


                    #pose_img와 garm_img를 텐서로 변환하고 맨 처음 차원에 배치차원을 추가
                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    #생서기 초기화해서 시드값고정 = 매번 동일한 이미지 생성가능. -> 보류
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    #주어진 변수들을 바탕으로 파이프라인실행(return값은 diffusion된 이미지)
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    # #edit된 이미지 임베딩 구하는 부분.
    # edited_image_hidden_states, _ = pipe.encode_image(images[0], device, 1, output_hidden_states=True)

    #크롭되었으면 복구하는데 우리 크롭안할꺼
    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        #diffusion된 이미지와 마스크된이미지를 반환.
        return images[0], mask_gray
    # return images[0], mask_gray


#이미지 diffusion 실행하는 함수
def Embedding_tryon(human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = human_img.convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        #모델을 사용해서 마스킹생성
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        #마스크도 이미지와 동일한크기로 만들어줌
        mask = mask.resize((768,1024))
    else:
        #사용자가 직접 마스킹된 이미지를 넣어줄때
        mask = pil_to_binary_mask(masked_img.convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
        #mask를 텐서로 변환하고 반전시킨 후, human_img 텐서와 곱하여 마스크가 적용된 이미지를 생성
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    #텐서를 PIL 이미지로 다시 변환
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    
    #densepose모델불러오기
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    #포즈이미지 생성
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, inaccurate"
                with torch.inference_mode():
                    #pipe에 있는encode_prompt함수이용해서 프롬프트임베딩생성
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    #옷사진 설명에 대한 프롬프트 임베딩생성.
                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )


                    #pose_img와 garm_img를 텐서로 변환하고 맨 처음 차원에 배치차원을 추가
                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    #생서기 초기화해서 시드값고정 = 매번 동일한 이미지 생성가능. -> 보류
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    #주어진 변수들을 바탕으로 파이프라인실행(return값은 diffusion된 이미지)
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    #edit된 이미지 임베딩 구하는 부분.
    edited_image_hidden_states, _ = pipe.encode_image(images[0], device, 1, output_hidden_states=True)

    #크롭되었으면 복구하는데 우리 크롭안할꺼
    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, edited_image_hidden_states, mask_gray
    else:
        #diffusion된 이미지와 마스크된이미지를 반환.
        return images[0], edited_image_hidden_states, mask_gray
    # return images[0], mask_gray


#이미지 diffusion 실행하는 함수
def Embedding2_tryon(human_img,masked_img,garm_img,edited_img_embed,is_checked,is_checked_crop,denoise_steps,seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img= garm_img.convert("RGB").resize((768,1024))
    human_img_orig = human_img.convert("RGB")    
    
    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))


    if is_checked:
        #모델을 사용해서 마스킹생성
        keypoints = openpose_model(human_img.resize((384,512)))
        model_parse, _ = parsing_model(human_img.resize((384,512)))
        mask, mask_gray = get_mask_location('hd', "upper_body", model_parse, keypoints)
        #마스크도 이미지와 동일한크기로 만들어줌
        mask = mask.resize((768,1024))
    else:
        #사용자가 직접 마스킹된 이미지를 넣어줄때
        mask = pil_to_binary_mask(masked_img.convert("RGB").resize((768, 1024)))
        # mask = transforms.ToTensor()(mask)
        # mask = mask.unsqueeze(0)
        #mask를 텐서로 변환하고 반전시킨 후, human_img 텐서와 곱하여 마스크가 적용된 이미지를 생성
    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    #텐서를 PIL 이미지로 다시 변환
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)


    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
     
    
    #densepose모델불러오기
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    # verbosity = getattr(args, "verbosity", None)
    #포즈이미지 생성
    pose_img = args.func(args,human_img_arg)    
    pose_img = pose_img[:,:,::-1]    
    pose_img = Image.fromarray(pose_img).resize((768,1024))
    
    with torch.no_grad():
        # Extract the images
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                #여기 다 바꾸기
                prompt = "model is wearing " + garment_des #여기바꾸기
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, inaccurate"
                with torch.inference_mode():
                    #pipe에 있는encode_prompt함수이용해서 프롬프트임베딩생성
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                    #옷사진 설명에 대한 프롬프트 임베딩생성.
                    prompt = "a photo of " + garment_des #여기바꾸기
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )


                    #pose_img와 garm_img를 텐서로 변환하고 맨 처음 차원에 배치차원을 추가
                    pose_img =  tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor =  tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    #생서기 초기화해서 시드값고정 = 매번 동일한 이미지 생성가능. -> 보류
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

                    #주어진 변수들을 바탕으로 파이프라인실행(return값은 diffusion된 이미지)
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength = 1.0,
                        pose_img = pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth = garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img, 
                        height=1024,
                        width=768,
                        ip_adapter_image = garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    # #edit된 이미지 임베딩 구하는 부분.
    # edited_image_hidden_states, _ = pipe.encode_image(images[0], device, 1, output_hidden_states=True)

    #크롭되었으면 복구하는데 우리 크롭안할꺼
    if is_checked_crop:
        out_img = images[0].resize(crop_size)        
        human_img_orig.paste(out_img, (int(left), int(top)))    
        return human_img_orig, mask_gray
    else:
        #diffusion된 이미지와 마스크된이미지를 반환.
        return images[0], mask_gray
    # return images[0], mask_gray


class IDM_VTON:
    #이거 단순하게 garment사진대신 edtied된 garment사진을 주는것.
    def Simple_tryon(self,human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
        edited_imaged, mask_gray = start_tryon(human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed)
        return edited_imaged,mask_gray
    
    def Embedding_tryon(self,human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed):
        edited_imaged, embedding, mask_gray = Embedding_tryon(human_img,masked_img,garm_img,garment_des,is_checked,is_checked_crop,denoise_steps,seed)
        return edited_imaged, embedding, mask_gray
    
    def Embedding2_tryon(self,human_img,masked_img,garm_img,edited_img_embed,is_checked,is_checked_crop,denoise_steps,seed):
        edited_imaged, mask_gray = Embedding2_tryon(human_img,masked_img,garm_img,edited_img_embed,is_checked,is_checked_crop,denoise_steps,seed)
        return edited_imaged, mask_gray

        
            