from IDM_VTON_CLASS import IDM_VTON
from PIL import Image
import os
import torch
# 'image.jpg' 파일을 불러와서 PIL 이미지 객체로 변환
ex1 = IDM_VTON() 
human_img__front_path = 'dict_img.jpg' #사람 앞면사진
human_img__back_path = 'dict_img.jpg'#사람 뒷면사진


masked_human_front_img_path = 'masked_dict_img.jpg' #마스크된이미지앞면사진
masked_human_back_img_path = 'masked_dict_img.jpg' #마스크된이미지뒷면사진

garment_front_path = 'garment_img.jpg' #옷앞면사진
garment_back_path = 'garment_img.jpg'#옷뒷면사진

###########################################################################

human_img__front = Image.open(human_img__front_path) #사람 앞면사진
human_img__back = Image.open(human_img__back_path) #사람 뒷면사진
human_img__extra = [] #앞면 뒷면 외 나머지 사진들(리스트형태로)

masked_human__front_img = Image.open(masked_human_front_img_path)  #마스크된이미지앞면사진
masked_human__back_img = Image.open(masked_human_back_img_path)  #마스크된이미지뒷면사진


garm_img_front = Image.open(garment_front_path)
garm_img_back = Image.open(garment_back_path)

garment_des = '베이지색 깔끔하고 스타일리쉬한 옷'

# 앞면 이미지 생성
front_edited_img, _ = ex1.Simple_tryon(human_img__front, masked_human__front_img, garm_img_front, garment_des, False, True, 40, 42) 

# 뒷면 이미지 생성
back_edited_img, _  = ex1.Simple_tryon(human_img__back, masked_human__back_img, garm_img_back, garment_des, False, True, 40, 42) 

#나머지 이미지 생성
edited_extra_img1 = []
front_or_back_list = [] #1이면 앞면 0이면 뒷면
for idx, image in (front_or_back_list, human_img__extra):
    if idx == 1:
        edited_img = ex1.Simple_tryon(image, masked_human__front_img, front_edited_img, garment_des, False, True, 40, 42)
    else:
        edited_img = ex1.Simple_tryon(image, masked_human__back_img, back_edited_img, garment_des, False, True, 40, 42)

#여기까지 단순하게 garment대신에 앞면뒷면으로 edited된 사진 넣어서 나머지 사진들에 대한 이미지 생성하는거
#########################################################################################################################################
#여기서부터는 프롬프트 임베딩 대신에 edited image임베딩을 넣어주는거(garment_des대신에)

# 앞면 이미지 생성 및 임베딩
front_edited_img, front_embed, _ = ex1.Embedding_tryon(human_img__front, masked_human__front_img, garm_img_front, garment_des, False, True, 40, 42) 

# 뒷면 이미지 생성 및 임베딩
back_edited_img, back_embed, _  = ex1.Embedding_tryon(human_img__back, masked_human__back_img, garm_img_back, garment_des, False, True, 40, 42) 

#위에 임베딩으로 나머지 이미지 생성
edited_extra_img2 = []
front_or_back_list = [] #1이면 앞면 0이면 뒷면
for idx, image in (front_or_back_list, human_img__extra):
    if idx == 1:
        edited_img = ex1.Embedding2_tryon(image, masked_human__front_img, garm_img_front, front_edited_img, False, True, 40, 42)
    else:
        edited_img = ex1.Embedding2_tryon(image, masked_human__back_img, garm_img_back, back_edited_img, False, True, 40, 42)


###########################################
#생성된 파일 저장.
output_folder_Simple = "output_images_Simple"
output_folder_embedding = "output_images_embedding"

# 폴더가 없다면 생성
os.makedirs(output_folder_Simple, exist_ok=True)
os.makedirs(output_folder_embedding, exist_ok=True)

# 이미지 리스트를 순회하면서 파일로 저장
for idx, image in enumerate(edited_extra_img1):
    file_path = os.path.join(output_folder_Simple, f"image_{idx + 1}.png")  # 각 파일 이름 설정
    image.save(file_path)  # 이미지 저장
    print(f"Saved {file_path}")

for idx, image in enumerate(edited_extra_img2):
    file_path = os.path.join(output_folder_embedding, f"image_{idx + 1}.png")  # 각 파일 이름 설정
    image.save(file_path)  # 이미지 저장
    print(f"Saved {file_path}")


