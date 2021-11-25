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

1. Download [preprocessed data](https://drive.google.com/file/d/1DKXr7shuThxHtZNTqzW4D9matq_NWGZM/view?usp=sharing) for SEL, SEL_Hard, and NKL(SL5K) datasets to ```root/Preprocessing/```. You can generate these data using ```Preprocessing/code/```. The original ```SEL``` and ```SEL_Hard``` datasets are provided in [here](https://github.com/dongkwonjin/Semantic-Line-DRM). ```NKL``` dataset is provided in [here](https://kaizhao.net/nkl).

2. Download our [model parameters](https://drive.google.com/file/d/1o64ilwfJhpCaIx-VYIyTFU4yYlwsL7g3/view?usp=sharing) to ```root/Modeling/``` if you want to get the performance of the paper.


3. Edit `config.py`. Please modify ```settings for path``` in the script file. Also, if you want to get the performance of the paper, please input ```run_mode``` to 'test_paper'.

4. Run with 
```
cd Semantic-Line-MWCS-master/(Modeling or Prerpocessing)/(..)/code/
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
