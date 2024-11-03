import torch

from threestudio.utils.misc import get_device, step_check, dilate_mask, erode_mask, fill_closed_areas
from threestudio.utils.perceptual import PerceptualLoss
import ui_utils
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from torchvision import transforms

from idmvton.gradio_demo.app import start_tryon
import matplotlib.pyplot as plt




# Diffusion model (cached) + prompts + edited_frames + training config

class EditGuidance:#기존에 있던 guidance매개변수삭제하고 idm-vton함수를 직접 호출
    def __init__(self, garm_img_front, garm_img_back, gaussian, origin_frames, text_prompt, per_editing_step, edit_begin_step,
                 edit_until_step, lambda_l1, lambda_p, lambda_anchor_color, lambda_anchor_geo, lambda_anchor_scale,
                 lambda_anchor_opacity, train_frames, train_frustums, cams, server
                 ):
        # self.guidance = guidance
        self.garm_img_front = garm_img_front
        self.garm_img_back = garm_img_back
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.origin_frames = origin_frames
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.edit_frames = {}
        self.visible = True
        self.text_prompt = text_prompt
        # self.prompt_utils = StableDiffusionPromptProcessor(
        #     {
        #         "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
        #         "prompt": text_prompt,
        #     }
        # )()
        #애초에 idm-vton에서 prompt도 같이 처리해주니까 raw prompt를 start_tryon에 넣어주면 될듯.
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())


    def __call__(self, rendering, view_index, step):
        self.gaussian.update_learning_rate(step)

        # nerf2nerf loss
        if view_index not in self.edit_frames or (
                self.per_editing_step > 0
                and self.edit_begin_step
                < step
                < self.edit_until_step
                and step % self.per_editing_step == 0
        ):
            #이부분을 guidance로 result받아오지않고 start_tryon함수로 받아오기
            # result = self.guidance(
            #     rendering,
            #     self.origin_frames[view_index],
            #     self.prompt_utils,
            # )
            #print("self.origin_frames[view_index]", len(self.origin_frames[view_index]))
            #print("self.garm_img", self.garm_img)
            to_pil = transforms.ToPILImage()

            rgb =to_pil(self.origin_frames[view_index].squeeze(0).permute(2, 0, 1))

            print("view_index", view_index )
            
            #rgb.show()

            #input()

            
            
            result, masked_img = start_tryon(rgb,self.garm_img_front, self.garm_img_back, self.text_prompt, False, True, 20, 42)#이 부분을 미리 수행해서 gpu memory out을 방지하자!
            result = transforms.ToTensor()(result)
            result = result.permute(1, 2, 0).unsqueeze(0)
            #self.edit_frames[view_index] = result["edit_images"].detach().clone() # 1 H W C
            self.edit_frames[view_index] = result.detach().clone() # 1 H W C
            self.train_frustums[view_index].remove()
            self.train_frustums[view_index] = ui_utils.new_frustums(view_index, self.train_frames[view_index],
                                                                    self.cams[view_index], self.edit_frames[view_index], self.visible, self.server)
            # print("edited image index", cur_index)

        gt_image = self.edit_frames[view_index]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # rendering과 gt_image를 같은 디바이스로 이동
        rendering = rendering.to(device)
        gt_image = gt_image.to(device)
        loss = self.lambda_l1 * torch.nn.functional.l1_loss(rendering, gt_image) + \
               self.lambda_p * self.perceptual_loss(rendering.permute(0, 3, 1, 2).contiguous(),
                                                    gt_image.permute(0, 3, 1, 2).contiguous(), ).sum()

        # anchor loss
        if (
                self.lambda_anchor_color > 0
                or self.lambda_anchor_geo > 0
                or self.lambda_anchor_scale > 0
                or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss += self.lambda_anchor_color * anchor_out['loss_anchor_color'] + \
                    self.lambda_anchor_geo * anchor_out['loss_anchor_geo'] + \
                    self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity'] + \
                    self.lambda_anchor_scale * anchor_out['loss_anchor_scale']

        return loss

