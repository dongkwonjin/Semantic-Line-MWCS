![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# [CVPR 2021] Harmonious Semantic Line Detection via Maximal Weight Clique Selection
### Dongkwon Jin, Wonhui Park, Seong-Gyun Jeong, and Chang-Su Kim
<!--
![Overview](Overview.png)
-->

Official implementation for **"Harmonious Semantic Line Detection via Maximal Weight Clique Selection"** 
[[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Jin_Harmonious_Semantic_Line_Detection_via_Maximal_Weight_Clique_Selection_CVPR_2021_paper.pdf) [[supp]](http://mcl.korea.ac.kr/~dkjin/cvpr2021/04585-supp.pdf) [[video]](https://www.youtube.com/watch?v=CFQ168_6jw8).

### Requirements
- PyTorch >= 1.3.1
- CUDA >= 10.0
- CuDNN >= 7.6.5
- python >= 3.6

### Installation
Create conda environment:
```
    $ conda create -n MWCS python=3.6 anaconda
    $ conda activate MWCS
    $ pip install opencv-python==3.4.2.16
    $ conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

Download repository:
```
    $ git clone https://github.com/dongkwonjin/Semantic-Line-MWCS.git
```

### Instruction

1. Download preprocessed data for SEL, SEL_Hard, and NKL(SL5K) datasets to ```root/```. You can download ```SEL``` and ```SEL_Hard``` datasets in here. ```NKL``` dataset is provided in here.

|        Dataset      |            Custom          |      Original path     |
|:-------------------:|:--------------------------:|:----------------------:|
|          SEL        |          [Download](https://drive.google.com/file/d/1K_lc284Mie-i3o4jEHF4dhObqOS_ITLc/view?usp=sharing)        |          [here](https://github.com/dongkwonjin/Semantic-Line-SLNet)        |
|       SEL_Hard      |          [Download](https://drive.google.com/file/d/1tsSlT7in6BdPV5SfvVR4qCEdOYA05zAz/view?usp=sharing)        |                        |


2. Download our model parameters to ```root/(task_folder_name)/``` if you want to get the performance of the paper.

|                 Task                 |     Model parameters     |
|:------------------------------------:|:------------------------:|
|        Semantic line detection       |        [Download](https://drive.google.com/file/d/18-T-gKj0x5QtOhauXVRAgLq3quMqxiKB/view?usp=sharing)        |
|   Dominant parallel line detection   |        [Download](https://drive.google.com/file/d/1r3LVK8FaNI4TDjVewJwG64QAsTB7wEeF/view?usp=sharing)        |
|  Reflection symmetry axis detection  |        [Download](https://drive.google.com/file/d/1pfo7fYMZe8kFOXnLOS5DY8WrvlsikWth/view?usp=sharing)        |



3. Edit `config.py`. Please modify ```dataset_dir``` and ```paper_weight_dir```. If you want to get the performance of the paper, please input ```run_mode``` to 'test_paper'.

4. Run with 
```
cd Semantic-Line-MWCS-master/(task_folder_name)/(model_folder_name)/code/
python main.py
```

### Reference
```
@Inproceedings{
    Jin2021MWCS,
    title={Harmonious Semantic Line Detection via Maximal Weight Clique Selection},
    author={Jin, Dongkwon and Park, Wonhui and Jeong, Seong-Gyun and Kim, Chang-Su},
    booktitle={CVPR},
    year={2021}
}
```
