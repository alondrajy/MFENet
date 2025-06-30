# Multi-scale Frequency Enhancement Network for Blind Image Deblurring

## 1. Abstract
>Image deblurring is a fundamental preprocessing technique aimed at recovering clear and detailed images from blurry inputs. However, existing methods often struggle to effectively integrate multi-scale feature extraction with frequency enhancement, limiting their ability to reconstruct fine textures, especially in the presence of non-uniform blur. To address these challenges, we propose a Multi-scale Frequency Enhancement Network (MFENet) for blind image deblurring. MFENet introduces a Multi-scale Feature Extraction Module (MS-FE) based on depthwise separable convolutions to capture rich multi-scale spatial and channel information. Furthermore, it employs a Frequency Enhanced Blur Perception Module (FEBP) that utilizes wavelet transforms to extract high-frequency details and multi-strip pooling to perceive non-uniform blur.
>
>Experimental results on the GoPro and HIDE datasets demonstrate that our method achieves superior deblurring performance in both visual quality and objective evaluation metrics. Notably, in downstream object detection tasks, our blind image deblurring algorithm significantly improves detection accuracy, further validating its effectiveness and robustness in practical applications.

## 2. MFENet: Multi-scale Frequency Enhancement Network
The overall architecture of the proposed MFENet:
![image](https://github.com/alondrajy/MFENet-for-deblurring/blob/main/network.png)

## 3. Requirements
The model is built in PyTorch 2.1.0 and tested on Ubuntu environment with NVIDIA GPU + CUDA CuDNN.

For installing, follow these intructions
```
conda create -n MFENet python==3.8
conda activate MFENet
pip install tqdm, scikit-image
```

Dependencies
- python 3.8.2
- torch 2.1.0
- numpy 1.21.4
- pillow 9.5.0
- torchvision 0.16.0
- scikit-image 0.21.0
- cuda 12.0


## 4. Dataset：<a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Train/Val/Test**  | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :- | :-:
01   | [**GoPro**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)   | 2017 | CVPR | 3214 | Synthetic | 2103/0/1111  | [link](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
02 | [**HIDE**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)   | 2019 | CVPR | 8422 | Synthetic | 6397/0/2025  | [link](https://github.com/joanshen0508/HA_deblur)

After preparing data set, the ```GoPro``` data folder should be like the format below:
```
GOPRO
├─ train
│ ├─ blur    % 2103 image pairs
│ │ ├─ xxxx.png
│ │ ├─ ......
│ │
│ ├─ sharp
│ │ ├─ xxxx.png
│ │ ├─ ......
│
├─ test    % 1111 image pairs
│ ├─ ...... (same as train)

```

## 5. Getting Started
### Training
```train.py``` contains the main training function code, and some parameters and dataset loactions need to be specified.
```python
python main.py --model_name "MFENet" --mode "train" --data_dir "dataset/GOPRO" 
```

### Testing
Run following commands to test and verify
```python
python main.py --model_name "MFENet" --mode "test" --data_dir "dataset/GOPRO" 
```

PSNR and SSIM
```python
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img1 = cv2.imread('img path1')
img2 = cv2.imread('img path2')
psnr = peak_signal_noise_ratio(img1, img2)
ssim = structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
```


## Citation
If this [paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.70036) help you, please cite us:
```BibTeX

@article{xiang2025multi,
  title={Multi-Scale Frequency Enhancement Network for Blind Image Deblurring},
  author={Xiang, YaWen and Zhou, Heng and Zhang, Xi and Li, ChengYang and Li, ZhongBo and Xie, YongQiang},
  journal={IET Image Processing},
  volume={19},
  number={1},
  pages={e70036},
  year={2025},
  publisher={Wiley Online Library}
}
```
