from IDM_VTON_CLASS import IDM_VTON
from PIL import Image
import torch
# 'image.jpg' 파일을 불러와서 PIL 이미지 객체로 변환
ex1 = IDM_VTON() 
human_img__front_path = 'dict_img.jpg' #사람 앞면사진
human_img__back_path = 'dict_img.jpg'
human_img__extra_path = 'dict_img.jpg'

#마스크된이미지사진
masked_human_img_path = 'masked_dict_img.jpg' #is_Checked가 true면 똥값넣어주기(true면 차피 안씀)

garment_front_path = 'garment_img.jpg'
garment_back_path = 'garment_img.jpg'
human_img__front = Image.open(human_img__front_path)
human_img__back = Image.open(human_img__back_path)
human_img__extra = Image.open(human_img__extra_path)

masked_human_img = Image.open(masked_human_img_path)
garm_img_front = Image.open(garment_front_path)
garm_img_back = Image.open(garment_back_path)

garment_des = '베이지색 깔끔하고 스타일리쉬한 옷'

#앞면 뒷면 제외하고 나머지 사진들 diffusion하는 함수


#앞면 임베딩 벡터 생성하고 diffusion하는 함수



#뒷면 임베딩 벡터생성하고 diffusion하는 함수




# 앞면 이미지 생성
img, front_embed, _ = ex1.tryon_front_or_back(human_img__front, masked_human_img, garm_img_front, garment_des, False, True, 40, 42) 

# 뒷면 이미지 생성
img, back_embed, _  = ex1.tryon_front_or_back(human_img__back_path, masked_human_img, garm_img_back, garment_des, False, True, 40, 42) 


#방법 1 단순히 합친다
combined_embed = torch.cat((front_embed, back_embed), dim=1)
#방법 2 앞면과 뒷면 어디에 가깝냐에 따라 동적으로 가중치를 조절해서 concat한다.
front_weight = 0.7
back_weight = 0.3

weighted_front = front_weight * front_embed
weighted_back = back_weight * back_embed
combined_embed = torch.cat((weighted_front, weighted_back), dim=1)

#앞면 뒷면 제외 이미지 생성 - 이거는 앞면 뒷면 맞춰서 넣어주는거 가우시안에디터에서 해야될듯

#added_embed = front_embed와 back_embed를 concat해서 둘다 고려해주기
img, d = ex1.tryon_extra(human_img__extra, masked_human_img, garm_img_front, garment_des, False, True, 40, 42, combined_embed)



#차례대로 원본사람사진, 원본사람사진에서 마스킹된사진, 옷사진, 옷설명 text
#그리고 40하고42는 더 공부해서 잘 조정해야할듯

# idm-vton결과를 파일로 저장하는거
img.save("output_img.jpg", "JPEG")
d.save("output_d.jpg", "JPEG")

