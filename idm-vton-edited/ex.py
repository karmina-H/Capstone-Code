from IDM_VTON_CLASS import IDM_VTON
from PIL import Image

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




# 앞면 혹은 뒷면 이미지 생성
img, d = ex1.front_or_back(human_img, masked_human_img, garm_img, garment_des, False, True, 40, 42) #is_front가 앞면이면 true뒷면이면 false
#앞면 뒷면 제외 이미지 생성
img, d = ex1.extra(human_img, masked_human_img, garm_img, garment_des, False, True, 40, 42, added_embed)



#차례대로 원본사람사진, 원본사람사진에서 마스킹된사진, 옷사진, 옷설명 text
#그리고 40하고42는 더 공부해서 잘 조정해야할듯

# idm-vton결과를 파일로 저장하는거
img.save("output_img.jpg", "JPEG")
d.save("output_d.jpg", "JPEG")

