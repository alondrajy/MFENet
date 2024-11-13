# Multi-scale Frequency Enhancement for Effective Blind Image Deblurring
## 1 This repository provides the official PyTorch implementation of the following paper:
### Multi-scale Frequency Enhancement for Effective Blind Image Deblurring
## 2 Model 
## 3 Dependencies
  * Python
  * Pytorch (2.1)
  * scikit-image
  * opencv-python
## 4 Dataset
#  6. Motion Deblurring Datasets:  <a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
**No.** |**Dataset** | **Year** | **Pub.** |**Size** |  **Types** | **Train/Val/Test**  | **Download**
:-: | :-: | :-: | :-:  | :-:  | :-: | :- | :-:
01   | [**GoPro**](https://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)   | 2017 | CVPR | 3214 | Synthetic | 2103/0/1111  | [link](https://github.com/SeungjunNah/DeepDeblur-PyTorch)
02 | [**HIDE**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shen_Human-Aware_Motion_Deblurring_ICCV_2019_paper.pdf)   | 2019 | CVPR | 8422 | Synthetic | 6397/0/2025  | [link](https://github.com/joanshen0508/HA_deblur)
## 5 The results of the visual comparison of the proposed methods are as follows.
### On the GoPro dataset
![image](https://github.com/alondrajy/MFENet-for-deblurring/blob/main/image/GoPro.png)
### On the HIDE dataset
![image](https://github.com/alondrajy/MFENet-for-deblurring/blob/main/image/HIDE.png)
## Visualization of the result of object detection on the deblurred image.
![image](https://github.com/alondrajy/MFENet-for-deblurring/blob/main/image/目标检测.png)
