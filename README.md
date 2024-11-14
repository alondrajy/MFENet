# Multi-scale Frequency Enhancement for Effective Blind Image Deblurring
## 1 This repository provides the official PyTorch implementation of the following paper:
### Multi-scale Frequency Enhancement for Effective Blind Image Deblurring
>Abstract:Image deblurring is a fundamental preprocessing technique aimed at recovering clear and detailed images from blurry inputs. However, existing methods often struggle to effectively integrate multi-scale feature extraction with frequency enhancement, limiting their ability to reconstruct fine textures, especially in the presence of non-uniform blur. To address these challenges, we propose a Multi-scale Frequency Enhancement Network (MFENet) for blind image deblurring. MFENet introduces a Multi-scale Feature Extraction Module (MS-FE) based on depthwise separable convolutions to capture rich multi-scale spatial and channel information. Furthermore, it employs a Frequency Enhanced Blur Perception Module (FEBP) that utilizes wavelet transforms to extract high-frequency details and multi-strip pooling to perceive non-uniform blur. Experimental results on the GoPro and HIDE datasets demonstrate that our method achieves superior deblurring performance in both visual quality and objective evaluation metrics. Notably, in downstream object detection tasks, our blind image deblurring algorithm significantly improves detection accuracy, further validating its effectiveness and robustness in practical applications.
#### Network
![image](https://github.com/alondrajy/MFENet-for-deblurring/blob/main/network.png)
## 2 Installation
### The model is built in PyTorch 2.1.0 and tested on Ubuntu environment.

For installing, follow these intructions
## 3 Dependencies
  * Python
  * Pytorch (2.1)
  * scikit-image
  * opencv-python
## 4 Dataset：<a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Train/Val/Test**  | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :- | :-:
01   | [**GoPro**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)   | 2017 | CVPR | 3214 | Synthetic | 2103/0/1111  | [link](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
02 | [**HIDE**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)   | 2019 | CVPR | 8422 | Synthetic | 6397/0/2025  | [link](https://github.com/joanshen0508/HA_deblur)
## 5 train
### Run following commands to train
python main.py --model_name "MFENet" --mode "train" --data_dir "dataset/GOPRO" 
## 6 test
### Run following commands to train
python main.py --model_name "MFENet" --mode "test" --data_dir "dataset/GOPRO" 

