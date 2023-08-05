# Pixel Adaptive Deep Unfolding Transformer for Hyperspectral Image Reconstruction 


<a target='_blank'> Miaoyu Li <sup>1</sup> </a>&emsp;
    <a href='https://ying-fu.github.io/' target='_blank'>Ying Fu<sup>1</a>&emsp;
    <a target='_blank'> Ji Liu <sup>2</sup> </a>&emsp;
    <a href='http://yulunzhang.com/' target='_blank'>Yulun Zhang <sup>3</sup></a>&emsp;
<div align="center">
<br>
<div >
    <sup>1</sup> Beijing Institute of Technology &emsp; <sup>2</sup> Baidu Inc. &emsp; <sup>3</sup> ETH Zurich &emsp; 
</div>
<br>
<i><strong><a target='_blank'>ICCV 2023</a></strong></i>
<br>
<br>
</div>

[Arxiv]() 
## 1. Comparison with State-of-the-art Methods

|  Method        | Params (M) | FLOPS (G) | PSNR  | SSIM  |  Model Zoo   |  Result |   
:----------------------------------------------------------: | :--------: | :-------: | :---: | :---: | :----------------------------------------------------------: | :-------:| 
 |  [DAUHST-L](https://github.com/caiyuanhao1998/MST)      |    6.15  |     79.50         | 38.36 | 0.967 |  [Repo](https://github.com/caiyuanhao1998/MST) | [Repo](https://github.com/caiyuanhao1998/MST) 
 |  PADUT-3stg        |    1.35    |   22.91  | 36.95| 0.962 |  [Google Driver](https://drive.google.com/file/d/14Rb5ALZWNMdyD7m_RKAfgXlgVcte3i-K/view?usp=sharing) | [Google Driver](https://drive.google.com/drive/folders/1q3Vktwf1K6Od3uJVKXZaIBTtSM4vzkRR?usp=sharing)|
 |  PADUT-5stg        |    2.24    |   37.90  | 37.84 | 0.967 | [Google Driver](https://drive.google.com/file/d/1Ers5ljefXmHuKXxx7NjCr4vhAyrW24n3/view?usp=sharing) | [Google Driver](https://drive.google.com/drive/folders/1q3Vktwf1K6Od3uJVKXZaIBTtSM4vzkRR?usp=sharing)|
 |  PADUT-7stg        |    3.14    |   52.90  | 38.41 | 0.971 | [Google Driver](https://drive.google.com/file/d/1RZJeyOwtZ7TVx0QDJNrPHrnLJ4nD8YO4/view?usp=drive_link) | [Google Driver](https://drive.google.com/drive/folders/1q3Vktwf1K6Od3uJVKXZaIBTtSM4vzkRR?usp=sharing) |
 |  PADUT-12stg        |    5.38    |   90.46   | 38.89 | 0.972 | [Google Driver](https://drive.google.com/file/d/1rhgJQ1IeNk0tk3B5bKsrnQfjgaLnWZFZ/view?usp=sharing) | [Google Driver](https://drive.google.com/drive/folders/1q3Vktwf1K6Od3uJVKXZaIBTtSM4vzkRR?usp=sharing) |

## 2. Create Environment
pip install -r requirements.txt

## 3. Data Preparation

Download cave_1024_28 ([Baidu Disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ), code: `fo0q` | [One Drive](https://bupteducn-my.sharepoint.com/:f:/g/personal/mengziyi_bupt_edu_cn/EmNAsycFKNNNgHfV9Kib4osB7OD4OSu-Gu6Qnyy5PweG0A?e=5NrM6S)), CAVE_512_28 ([Baidu Disk](https://pan.baidu.com/s/1ue26weBAbn61a7hyT9CDkg), code: `ixoe` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EjhS1U_F7I1PjjjtjKNtUF8BJdsqZ6BSMag_grUfzsTABA?e=sOpwm4)), KAIST_CVPR2021 ([Baidu Disk](https://pan.baidu.com/s/1LfPqGe0R_tuQjCXC_fALZA), code: `5mmn` | [One Drive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/lin-j21_mails_tsinghua_edu_cn/EkA4B4GU8AdDu0ZkKXdewPwBd64adYGsMPB8PNCuYnpGlA?e=VFb3xP)), TSA_simu_data ([Baidu Disk](https://pan.baidu.com/s/1LI9tMaSprtxT8PiAG1oETA), code: `efu8` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFDwE-7z1fzeWCRDA?e=ofvwrD)), TSA_real_data ([Baidu Disk](https://pan.baidu.com/s/1RoOb1CKsUPFu0r01tRi5Bg), code: `eaqe` | [One Drive](https://1drv.ms/u/s!Au_cHqZBKiu2gYFTpCwLdTi_eSw6ww?e=uiEToT)), and then put them into the corresponding folders of `datasets/` and recollect them as the following form:

```shell

    |--real
    	|-- test_code
    	|-- train_code
    |--simulation
    	|-- test_code
    	|-- train_code
    |--datasets
        |--cave_1024_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene205.mat
        |--CAVE_512_28
            |--scene1.mat
            |--scene2.mat
            ：  
            |--scene30.mat
        |--KAIST_CVPR2021  
            |--1.mat
            |--2.mat
            ： 
            |--30.mat
        |--TSA_simu_data  
            |--mask.mat   
            |--Truth
                |--scene01.mat
                |--scene02.mat
                ： 
                |--scene10.mat
        |--TSA_real_data  
            |--mask.mat   
            |--Measurements
                |--scene1.mat
                |--scene2.mat
                ： 
                |--scene5.mat
```

Following TSA-Net and DGSMP, we use the CAVE dataset (cave_1024_28) as the simulation training set. Both the CAVE (CAVE_512_28) and KAIST (KAIST_CVPR2021) datasets are used as the real training set. 

## 4. Simulation Experiement
### 4.1　Training
```
cd simulation
python train.py --template dauhst --outf ./exp/padut_3stg/ --method padut_3

python train.py --template dauhst --outf ./exp/padut_5stg/ --method padut_5

python train.py --template dauhst --outf ./exp/padut_7stg/ --method padut_7

python train.py --template dauhst --outf ./exp/padut_12stg/ --method padut_12
```
### 4.2　Testing
```
python test.py --template dauhst --outf ./exp/padut_3stg/ --method padut_3 --pretrained_model_path ./checkpoints/3.pth

python test.py --template dauhst --outf ./exp/padut_5stg/ --method padut_5 --pretrained_model_path ./checkpoints/5.pth

python test.py --template dauhst --outf ./exp/padut_7stg/ --method padut_7 --pretrained_model_path ./checkpoints/7.pth

python test.py --template dauhst --outf ./exp/padut_12stg/ --method padut_12 --pretrained_model_path ./checkpoints/12.pth
```
## 5. Real Experiement
### 5.1　Training
```
python train.py  --template dauhst --outf ./exp/padut_3stg/ --method padut_3  
```
### 5.2　Testing
```
python test.py  --template dauhst --outf ./exp/padut_3stg/ --method padut_3    --pretrained_model_path ./checkpoints/3.pth
```
## 6. Acknowledgements 
This code repository's implementation is based on these two works:

```shell

@inproceedings{mst,
  title={Mask-guided Spectral-wise Transformer for Efficient Hyperspectral Image Reconstruction},
  author={Yuanhao Cai and Jing Lin and Xiaowan Hu and Haoqian Wang and Xin Yuan and Yulun Zhang and Radu Timofte and Luc Van Gool},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
@inproceedings{res,
  title={Residual Degradation Learning Unfolding Framework with Mixing Priors across Spectral and Spatial for Compressive Spectral Imaging},
  author={Yubo Dong and Dahua Gao and Tian Qiu and Yuyan Li and Minxi Yang and Guangming Shi},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
## Citation
```shell
@inproceedings{PADUT,
  title={Pixel Adaptive Deep Unfolding Transformer for Hyperspectral Image Reconstruction},
  author={Miaoyu Li and Ying fu and Ji Liu and Yulun Zhang},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision(ICCV)},
  year={2023}
}

```