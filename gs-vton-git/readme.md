
<p align="center">

</p>




## Installation
Our environment has been tested on Ubuntu 22, CUDA 11.8 with 3090, A5000 and A6000.
1. Clone our repo and create conda environment
```
git clone https://github.com/buaacyw/GaussianEditor.git && cd GaussianEditor

# (Option one) Install by conda
conda env create -f environment.yaml

# (Option two) You can also install by pip
# CUDA version 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# CUDA version 11.8
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# (Option three) If the below two options fail, please try this:
# For CUDA 11.8
bash install.sh
```

2. (Optional) Install our forked viser [Required by WebUI)
```
mkdir extern && cd extern
git clone https://github.com/heheyas/viser 
pip install -e viser
cd ..
```

3. (Optional) Download Wonder3D checkpoints [Required by <b>Add</b>]
```bash
sh download_wonder3d.sh
```

## WebUI Guide
Please be aware that our WebUI is currently in a beta version. Powered by [Viser](https://github.com/nerfstudio-project/viser/tree/main), you can use our WebUI even if you are limited to remote server. For details, please follow [WebUI Guide](https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md).

## How to achieve better result

The demand for 3D editing is very diverse. For instance, if you only want to change textures and materials or significantly modify geometry, it's clear that a one-size-fits-all hyperparameter won't work. Therefore, we cannot provide a default hyperparameter setting that works effectively in all scenarios. Therefore, if your results do not meet expectations, please refer to our [hyperparameter tuning](https://github.com/buaacyw/GaussianEditor/blob/master/docs/hyperparameter.md) document. In it, we detail the function of each hyperparameter and advise on which parameters to adjust when you encounter specific issues. 

## Command Line
We also provide a command line version of GaussianEditor. Like WebUI, you need to specify your path to the pretrained Gaussians and COLMAP outputs as mentioned in [here](https://github.com/buaacyw/GaussianEditor/blob/1fa96851c132258e0547ba73372f37cff83c92c3/docs/webui.md?plain=1#L20).
Please check scripts in `sciprt` folder. Simply change `data.source` to your COLMAP output directory and 
`system.gs_source` to your pretrained Gaussians and run our demo scripts.


## TODO

- [x] Add 3dgs segmentation model 


## Acknowledgement

Our code is based on these wonderful repos:

* [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
* [Wonder3D](https://github.com/xxlong0/Wonder3D)
* [Threestudio](https://github.com/threestudio-project/threestudio)
* [Viser](https://github.com/nerfstudio-project/viser)
* [InstructNerf2Nerf](https://github.com/ayaanzhaque/instruct-nerf2nerf)
* [InstructPix2Pix](https://github.com/timothybrooks/instruct-pix2pix)
* [Controlnet](https://github.com/lllyasviel/ControlNet)
* [IDM-VTON] (https://github.com/yisol/IDM-VTON)



