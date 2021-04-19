![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [CVPR 2021] Harmonious Semantic Line Detection via Maximal Weight Clique Selection
### Dongkwon Jin, Wonhui Park, Seong-Gyun Jeong, and Chang-Su Kim
<!--
![Overview](Overview.png)
-->

Official pytorch implementation for **"Harmonious Semantic Line Detection via Maximal Weight Clique Selection"** 
[[paper]](https://arxiv.org/pdf/2104.06903.pdf) [[supp]](http://mcl.korea.ac.kr/~dkjin/cvpr2021/04585-supp.pdf) [[video]](http://mcl.korea.ac.kr/dkjin_cvpr2021/).

<!--
### Requirements
- PyTorch 1.3.1
- CUDA 10.0
- CuDNN 7.6.5
- python 3.6

### Installation
Create conda environment:
```
    $ conda create -n DRM python=3.6 anaconda
    $ conda activate DRM
    $ pip install opencv-python==3.4.2.16
    $ conda install pytorch==1.3.1 torchvision cudatoolkit=10.0 -c pytorch
```

Download repository:
```
    $ git clone https://github.com/dongkwonjin/Semantic-Line-MWCS.git
```
-->

### Instruction

To be released.

<!--
1. Download the following datasets to ```root/```. ```SEL``` and ```SEL_Hard``` are datasets for semantic line detection. Others are datasets for applications. We obtain the edge detection results in  ```edge``` folder, by employing [HED algorithm](https://github.com/sniklaus/pytorch-hed).

|        Dataset      |            Custom          |      Original path     |
|:-------------------:|:--------------------------:|:----------------------:|
|          SEL        |          [Download](https://drive.google.com/file/d/1K_lc284Mie-i3o4jEHF4dhObqOS_ITLc/view?usp=sharing)        |          [here](https://github.com/dongkwonjin/Semantic-Line-SLNet)        |
|       SEL_Hard      |          [Download](https://drive.google.com/file/d/1tsSlT7in6BdPV5SfvVR4qCEdOYA05zAz/view?usp=sharing)        |                        |
|    AVA landscape    |          [Download](https://drive.google.com/file/d/1RTqOQ7-JCvcKJncwQ-lo-i1svev9xyrw/view?usp=sharing)        |          [here](https://faculty.ist.psu.edu/zzhou/projects/vpdetection/)        |
|         ICCV        |          [Download](https://drive.google.com/file/d/1Tq5nriVoQbL7thHXSVcBY9gPrX2DlCHJ/view?usp=sharing)        |          [here](https://sites.google.com/view/symcomp17/)        |
|          NYU        |          [Download](https://drive.google.com/file/d/1G71Yspg1T-BkffxaoxDwpvlvad8IuHy2/view?usp=sharing)        |          [here](https://symmetry.cs.nyu.edu/)        |
|       SYM_Hard      |          [Download](https://drive.google.com/file/d/1dydxRGN7UsfFcg6tzNzrm_o0WJZEEKP8/view?usp=sharing)        |                        |


2. Download our model parameters to ```root/(task_folder_name)/``` if you want to get the performance of the paper.

|                 Task                 |     Model parameters     |
|:------------------------------------:|:------------------------:|
|        Semantic line detection       |        [Download](https://drive.google.com/file/d/18-T-gKj0x5QtOhauXVRAgLq3quMqxiKB/view?usp=sharing)        |
|   Dominant parallel line detection   |        [Download](https://drive.google.com/file/d/1r3LVK8FaNI4TDjVewJwG64QAsTB7wEeF/view?usp=sharing)        |
|  Reflection symmetry axis detection  |        [Download](https://drive.google.com/file/d/1pfo7fYMZe8kFOXnLOS5DY8WrvlsikWth/view?usp=sharing)        |



3. Edit `config.py`. Please modify ```dataset_dir``` and ```paper_weight_dir```. If you want to get the performance of the paper, please input ```run_mode``` to 'test_paper'.

4. Run with 
```
cd Semantic-Line-DRM-master/(task_folder_name)/(model_folder_name)/code/
python main.py
```
-->

### Reference
```
@Inproceedings{
    Jin2021MWCS,
    title={Harmonious Semantic Line Detection via Maximal Weight Clique Selection},
    author={Dongkwon Jin, Wonhui Park, Seong-Gyun Jeong, and Chang-Su Kim},
    booktitle={CVPR},
    year={2021}
}
```
