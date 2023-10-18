# TensorRT support for webui-SDXL

This is an extension made for webui, which make your sdxl model in webui can be accelerated by tensorRT.

You can check NVIDIA official tensorRT example for all kinds of diffusion model here [NVIDIA](https://github.com/rajeevsrao/TensorRT/blob/release/8.6/demo/Diffusion/README.md)

This extension is built on new python tensorRT, no need to build complex system enviroment.
## Before install
First you need to modify original file in webui, which power your webui to change the Unet of SDXL.

Let's start from `your_webui_path/modules/sd_hijack.py`, and you may find these python scripts in line `242`
````
if not hasattr(ldm.modules.diffusionmodules.openaimodel, 'copy_of_UNetModel_forward_for_webui'):
    ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui = ldm.modules.diffusionmodules.openaimodel.UNetModel.forward
ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = sd_unet.UNetModel_forward
````

we just need to add 2 more lines, which make it looks like:
````
if not hasattr(ldm.modules.diffusionmodules.openaimodel, 'copy_of_UNetModel_forward_for_webui'):
   ldm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui = ldm.modules.diffusionmodules.openaimodel.UNetModel.forward
   sgm.modules.diffusionmodules.openaimodel.copy_of_UNetModel_forward_for_webui = sgm.modules.diffusionmodules.openaimodel.UNetModel.forward
ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = sd_unet.UNetModel_forward
sgm.modules.diffusionmodules.openaimodel.UNetModel.forward = sd_unet.UNetModel_forward
````

Congratulations! Now your webui can integrate with tensorRT for SDXL now.

## How to install

You can just add this extension to webui just like any other extensions normally, or you can clone the extension and copy it to you extension path manually:
````
git clone https://github.com/Happenmass/stable-diffusion-webui-tensorr-sdxl.git

copy stable-diffusion-webui-tensorr-sdxl stable-diffusion-webui/extensions/
````

You need to choose the same version of CUDA as python's torch library is using. For torch 2.0.1 it is CUDA 11.8.

then you need to install requirments of the extension manually(Linux):
````
cd stable-diffusion-webui
. venv/bin/activate
cd extensions/stable-diffusion-webui-tensorr-sdxl
pip install -r requirements.txt
````
### Windows
Create a new environment variables named `INCLUDE` and add your nvtx path like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\include`

Right click mouse and open terminal under your stable diffusion webui path
````
.\venv\Scripts\activate
cd extensions/stable-diffusion-webui-tensorr-sdxl
pip install -r requirements.txt
````
Now the installing progress of the extension has finished

## How to use
1. Download the onnx opt model of Unet supplied by stability AI from [stabilityAI](https://huggingface.co/stabilityai/stable-diffusion-xl-1.0-tensorrt/tree/main/sdxl-1.0-base/unetxl.opt), please remember to download both model file and data file.
2. Create a folder named `Unet-onnx` under the path `stable-diffusion-webui/models`
3. Copy the model file and data file you downloaded into the path below.
4. Start your webui and you can find a new tab named TensorRT after tab train.
5. In `Convert OPT ONNX to TensorRT` tab, Input you Opt ONNX model file name `model`.
6. In `Convert OPT ONNX to TensorRT` tab, Input Output tensorRT model filename you want such as `Unet`.
7. Input Max and Min height and width for tensorRT model, `1024` for all recommended.
8. Min batch size `1` recommended, Max batch size `4` recommended
9. Press `Convert ONNX to TensorRT`
   * This takes very long - from 15 minues to an hour.
   * After the conversion has finished, you will find an `.plan` file with model in `models/Unet-trt` directory.
10. In settings, in `Stable Diffusion` page, use `SD Unet` option to select newly generated TensorRT model.
11. Generate pictures.
