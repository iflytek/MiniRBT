[**Chinese**](https://github.com/iflytek/minirbt) | [**English**](https://github.com/iflytek/minirbt/blob/main/README_EN.md)


<p align="center">
    <br>
    <img src="./pics/banner.png" width="600"/>
    <br>
</p>
<p align="center">
    <a href="https://github.com/ymcui/LERT/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/iflytek/MiniRBT.svg?color=blue&style=flat-square">
    </a>
</p>

In the field of natural language processing, pre-trained language models have become a very important basic technology. In order to further promote the research and development of Chinese information processing, HFL launched a Chinese small pre-training model **MiniRBT** based on the self-developed knowledge distillation tool [TextBrewer](https://github.com/airaria/TextBrewer), combined with Whole Word Masking technology and Knowledge Distillation technology.  

----

[Chinese LERT](https://github.com/ymcui/LERT) | [Chinese PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [TextBrewer](https://github.com/airaria/TextBrewer) | [TextPruner](https://github.com/airaria/TextPruner)  

More resources by HFL: https://github.com/iflytek/HFL-Anthology

## Guide

| Section | Description |
|-|-|
| [Introduction](#introduction) | Introduce technical solutions applied to small pre-trained models |
| [Model download](#model-download) | Download links for small pretrained models |
| [Quick Load](#quick-load) | Learn how to quickly load our models through[🤗Transformers](https://github.com/huggingface/transformers) |
| [Model Comparison](#model-comparison) | Compare the models published in this repository |
| [Distillation parameters](#distillation-parameters) | Pretrained distillation hyperparameter settings |
| [Baselines](#baselines) | Baseline results for several Chinese NLP datasets (partial) |
| [Two-stage Knowledge Distillation](#two-stage) | The results of two-stage distillation and one-stage distillation |
| [Pre-training](#pre-training) | How to use the pre-training code |
| [Useful Tips](#useful-tips) | Provide several useful tips for using small pretrained models |
| [FAQ](#faq) | Frequently Asked Questions |
| [Citation](#citation) | Citation |

## Introduction

At present, there are some problems with the pre-training model, such as large amount of parameters, long inference time, and difficult to deploy. In order to reduce model parameters and storage space and speed up inference, we have launched a small Chinese pre-training model with strong practicability and wide applicability. We used the following techniques:

* **Whole Word Masking (wwm)**，if part of a WordPiece subword of a complete word is masked, other parts of the same word will also be masked. For more detailed instructions and examples, please refer to:：**[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)**.In this work, [LTP](http://ltp.ai) is used as a word segmentation tool.

* **Two-stage Knowledge Distillation**,the intermediate model is used to assist in the distillation of the teacher to the student, that is, the teacher is first distilled to the teacher assistant model, and the student is obtained by distilling the assistant model, so as to improve the performance of the student in downstream tasks.  

* **Build Narrower and Deeper Student Models**,a narrower and deeper network structure is constructed as the student MiniRBT (6 layers, hidden layer dimension 256 and 288) to improve the performance of the student on downstream tasks when the model parameters (excluding the embedding layer) are similar.
  

**MiniRBT** currently has two branch models, namely **MiniRBT-H256** and **MiniRBT-H288**, indicating that the hidden layer dimensions are 256 and 288, both of which are 6-layer Transformer structures, obtained by two-stage distillation. At the same time, in order to facilitate the comparison of experimental results, we also provide the download of the **RBT4-H312** model of the TinyBERT structure.

We will provide a complete technical report in the near future, so stay tuned.

## Model download

| Model Name                       | Layer | Hid-size | Att-Head | Params |                         Google Drive                         |                          Baidu Disk                          |
| :------------------------------- | :---: | :------: | :------: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **MiniRBT-h288**                 |   6   |   288    |    8     | 12.3M  | [[PyTorch]](https://drive.google.com/file/d/1DDSUZbWORlkpKFfjsZJnsoyh5cQKZhEA/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/1pIwzx0Zu62fLOSQASxTohQ?pwd=7313)<br/>（pw：7313） |
| **MiniRBT-h256**                 |   6   |   256    |    8     | 10.4M  | [[PyTorch]](https://drive.google.com/file/d/1M5U1VzfrD82SOGinhauW1N4Xupui68xj/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/16ZMOoliMLsa2KqMKpwT0IA?pwd=iy53)<br/>（pw：iy53） |
| **RBT4-h312** (same as TinyBERT) |   4   |   312    |    12    | 11.4M  | [[PyTorch]](https://drive.google.com/file/d/1NmvrWvGsJwdXVd6C1Qs48K1G6KYcXVKk/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/11I9ojsnGK-7eZXMkl1k7EQ?pwd=ssdw)<br/>（pw：ssdw） |

Alternatively, download from (PyTorch & TF2)：<https://huggingface.co/hfl>

Steps: select one of the model in the page above → click "list all files in model" at the end of the model page → download bin/json files from the pop-up window

## Quick Load

### Huggingface-Transformers

With [Huggingface-Transformers](https://github.com/huggingface/transformers), the models above could be easily accessed and loaded through the following codes.

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

**Notice: Please use BertTokenizer and BertModel for loading these model. DO NOT use RobertaTokenizer/RobertaModel!**  
The corresponding MODEL_NAME is as follows:
| Model       |            MODEL_NAME     |
| ------------- | ------------------------- |
| MiniRBT-H256 | "hfl/minirbt-h256" |
| MiniRBT-H288  | "hfl/minirbt-h288"  |
| RBT4-H312 | "hfl/rbt4-h312" |

## Model Comparison

Some model details are summarized as follows

| Model | Layers | Hidden_size | FFN_size | Head_num | Model_size | Model_size(W/O embeddings) | Speedup |
| :------ | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa | 12 | 768 | 3072 | 12 | 102.3M (100%) | 85.7M(100%) | 1x |
| RBT6 (KD) | 6 | 768 | 3072 | 12 | 59.76M (58.4%) | 43.14M (50.3%) | 1.7x |
| RBT3 | 3 | 768 | 3072 | 12 | 38.5M (37.6%) | 21.9M (25.6%) | 2.8x |
| **RBT4-H312**| 4 | 312 | 1200 | 12 | 11.4M (11.1%) | 4.7M (5.5%) | 6.8x |
| **MiniRBT-H256** | 6 | 256 | 1024 | 8 | 10.4M (10.2%) | 4.8M (5.6%) | 6.8x |
| **MiniRBT-H288** | 6 | 288 | 1152 | 8 | 12.3M (12.0%) | 6.1M (7.1%) | 5.7x |

* RBT3：initialized by three layers of RoBERTa-wwm-ext and continue to pre-train to get.For more detailed instructions, please refer to:**[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm/#%E5%B0%8F%E5%8F%82%E6%95%B0%E9%87%8F%E6%A8%A1%E5%9E%8B)**
* RBT6 (KD)：Teacher Assistant,initialized by six layers of RoBERTa-wwm-ext and distilled from the RoBERTa
* MiniRBT-*：distilled from the TA model RBT6 (KD)
* RBT4-H312: distilled directly from the RoBERTa

## Distillation parameters

| Model | Batch Size | Training Steps | Learning Rate | Temperature | Teacher |
| :------ | :---------: | :---------: | :---------: | :---------: |  :---------: |
| RBT6 (KD) | 4096 | 100k<sup>MAX512 | 4e-4 | 8 | RoBERTa_wwm_ext |
| **RBT4-H312**| 4096 | 100k<sup>MAX512 | 4e-4 | 8 | RoBERTa_wwm_ext |
| **MiniRBT-H256** | 4096 | 100k<sup>MAX512 | 4e-4 | 8 | RBT6 (KD) |
| **MiniRBT-H288** | 4096 | 100k<sup>MAX512 | 4e-4 | 8 |  RBT6 (KD)  |

## Baselines

We experiment on several Chinese datasets.

* [**CMRC 2018**: Span-Extraction Machine Reading Comprehension (Simplified Chinese)](https://github.com/ymcui/cmrc2018)
* [**DRCD**: Span-Extraction Machine Reading Comprehension (Traditional Chinese)](https://github.com/DRCKnowledgeTeam/DRCD)
* [**OCNLI**: Original Chinese Natural Language Inference](https://github.com/CLUEbenchmark/OCNLI/tree/main/data/ocnli)
* [**LCQMC**: Sentence Pair Matching](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
* [**BQ Corpus**: Sentence Pair Matching](http://icrc.hitsz.edu.cn/Article/show/175.html)
* [**TNEWS**: Text Classification](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)
* [**ChnSentiCorp**: Sentiment Analysis](https://huggingface.co/datasets/seamew/ChnSentiCorp/tree/main)
  

After a learning rate search, we verified that models with small parameters require higher learning rates and more iterations. The following are the learning rates for each dataset.

Best Learning Rate:

| Model | CMRC 2018 | DRCD | OCNLI | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :------ | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa | 3e-5 | 3e-5 | 2e-5 | 2e-5 | 3e-5 | 2e-5 | 2e-5 |
| * | 1e-4 | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |

\* represents all small models (RBT3, RBT4-H312, MiniRBT-H256, MiniRBT-H288)

**Note: In order to ensure the reliability of the results, for the same model, we set the epochs to 2, 3, 5, and 10, run at least 3 times (different random seeds), and report the maximum value of the average model performance. Not surprisingly, the results of your runs should probably fluctuate around this average.All the following experimental results are experimental results on the development set.**  

**Experimental results:**
| Task               | CMRC 2018  | DRCD        | OCNLI  | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :-------           | :---------:| :---------: | :-----:| :---: | :------:  | :---: | :---------:  |
| RoBERTa            | 87.3/68    | 94.4/89.4   | 76.58  | 89.07 | 85.76     | 57.66 |     94.89    |
| RBT6 (KD)               | 84.4/64.3  | 91.27/84.93 | 72.83  | 88.52 | 84.54     | 55.52 |     93.42    |
| RBT3               | 80.3/57.73 | 85.87/77.63 | 69.80  | 87.3  | 84.47     | 55.39 |     93.86    |
| **RBT4-H312**       | 77.9/54.93 | 84.13/75.07 | 68.50  | 85.49 | 83.42     | 54.15 |     93.31    |
| **MiniRBT-H256** | 78.47/56.27| 86.83/78.57 | 68.73  | 86.81 | 83.68     | 54.45 |     92.97    |
| **MiniRBT-H288** | 80.53/58.83| 87.1/78.73  | 68.32  | 86.38 | 83.77     | 54.62 |     92.83    |

**Relative performance:**

| Task               | CMRC 2018   | DRCD        | OCNLI | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :-------           | :---------: | :---------: | :----:| :---: | :-------: | :---: | :---------:  |
| RoBERTa            | 100%/100%   | 100%/100%   | 100%  | 100%  |    100%   | 100%  |     100%     |
| RBT6 (KD)               | 96.7%/94.6% | 96.7%/95%   | 95.1% | 99.4% |    98.6%  | 96.3% |     98.5%    |
| RBT3               | 92%/84.9%   | 91%/86.8%   | 91.1% | 98%   |    98.5%  | 96.1% |     98.9%    |
| **RBT4-H312**       | 89.2%/80.8% | 89.1%/84%   | 89.4% | 96%   |    97.3%  | 93.9% |     98.3%    |
| **MiniRBT-H256** | 89.9%/82.8% | 92%/87.9%   | 89.7% | 97.5% |    97.6%  | 94.4% |     98%      |
| **MiniRBT-H288** | 92.2%/86.5% | 92.3%/88.1% | 89.2% | 97%   |    97.7%  | 94.7% |     97.8%    |

<h2 id="two-stage">Two-stage knowledge distillation<sup>†</sup></h2>

We compared the two-stage distillation (RoBERTa→RBT6(KD)→MiniRBT-H256) with the one-stage distillation (RoBERTa→MiniRBT-H256), and the experimental results are as follows. The experimental results show that the effect of two-stage distillation is better.

| Model                   | CMRC 2018      | OCNLI     | LCQMC     | BQ Corpus | TNEWS     |
| :-------               | :---------:    | :-------: | :-------: | :-------: | :-------: |
| MiniRBT-H256 (two-stage)| **77.97/54.6** | **69.11** | **86.58** | **83.74** | **54.12** |
| MiniRBT-H256 (one-stage)| 77.57/54.27    | 68.32     | 86.39     | 83.55     |     53.94 |

<sup>†</sup>:The pre-trained model in this part is distilled with 30,000 steps, which is different from the published model.

## Pre-training

We used the [**TextBrewer**](https://github.com/airaria/TextBrewer) toolkit to implement the process of pretraining distillation. The complete training code is located in the pretraining directory.

### Project Structure

* `dataset`:
  * `train`: training set
  * `dev`： development set
* `distill_configs`: student config
* `jsons`: configuration file for training dataset
* `pretrained_model_path`:
  * `ltp`: weight of ltp word segmentation model,including`pytorch_model.bin`，`vocab.txt`，`config.json`
  * `RoBERTa`: weight of teacher，including`pytorch_model.bin`，`vocab.txt`，`config.json`
* `scripts`: generation script for TA initialization weights
* `saves`: output_dir
* `config.py`: configuration file for training parameters
* `matches.py`: matching different layers of the student and the teacher
* `my_datasets.py`: load datasets
* `run_chinese_ref.py`: generate reference file
* `train.py`：project entry
* `utils.py`: helpful functions for distillation
* `distill.sh`: Training scripts

### Requirements

This part of the library has only be tested with Python3.8,PyTorch v1.10.1. There are few specific dependencies to install before launching a distillation, you can install them with the command `pip install -r requirements.txt`

### Model preparation

Download ltp and RoBERTa from [huggingface](https://huggingface.co/models), and unzip it into `${project-dir}/pretrained_model_path/`

### Data Preparation

For Chinese models, we need to generate a reference files (which requires the ltp library), because it's tokenized at the character level.

```sh
python run_chinese_ref.py
```

Because the pre-training data set is large, it is recommended to pre-process the reference file after it is generated. You only need to run the following command

```sh
python my_datasets.py
```

### training

We provide example training scripts for training with KD with different combination of training units and objectives in distill.sh.The script supports multi-GPU training and we explain the arguments in following:

* `teacher_name or_path`：weight of teacher
* `student_config`: student config
* `num_train_steps`: total training steps
* `ckpt_steps`：the frequency of the saving model
* `learning_rate`: max learning rate for pre_training
* `train_batch_size`: batchsize for training
* `data_files_json`: data json
* `data_cache_dir`：cache path
* `output_dir`: output dir
* `output encoded layers`：set hidden layer output to True
* `gradient_accumulation_steps`：gradient accumulation steps
* `temperature`：temperature value,this is recommended to be set to be 8
* `fp16`：speed up training

Training with distillation is really simple once you have pre-processed the data. An example for training MiniRBT-H256 is as follows:

```bash
sh distill.sh
```

**Tips**: Starting distilled training with good initialization of the model weights is crucial to reach decent performance. In our experiments, we initialized our TA model from a few layers of the teacher (RoBERTa) itself! Please refer to `scripts/init_checkpoint_TA.py`to create a valid initialization checkpoint and use `--student_pretrained_weights` argument to use this initialization for the distilled training!

## Useful Tips

* The initial learning rate is a very important parameter and needs to be adjusted according to the target task.
* The optimal learning rate of the small parameter model is quite different from `RoBERT-wwm`, so be sure to adjust the learning rate when using the small parameter model (based on the above experimental results, the small parameter model requires a higher initial learning rate, more iterations).
* In the case where the parameters (excluding the embedding layer) are basically the same, the effect of **MiniRBT-H256** is better than that of **RBT4-H312**, and it is also proved that the narrower and deeper model structure is better than the wide and shallow model structure.
* On tasks related to reading comprehension, **MiniRBT-H288** performs better. The effects of other tasks **MiniRBT-H288** and **MiniRBT-H256** are the same, and the corresponding model can be selected according to actual needs.

## FAQ

**Q: How to use this model?**  
A: Refer to [Quick Load](#quick-load).It is used in the same way as **[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)**.

**Q : Why a reference file?**  
A : Suppose we have a Chinese sentence like: `天气很好`. The original BERT will tokenize it as `['天','气','很','好']`(character level). But in Chinese `天气` is a complete word. To implement whole word masking, we need a reference file to tell the model where `##` should be added, so something like `['天', '##气', '很', '好']` will be generated.  
**Note: This is an auxiliary reference file and does not affect the original input of the model (ie, has nothing to do with the word segmentation results).**

**Q: Why is the effect of RBT6 (KD) in downstream tasks so much lower than that of RoBERTa? Why is the effect of MiniRBT-H256/MiniRBT-H288/RBT4-H312 so low? How to improve the effect?**  
A: The RBT6 (KD) described above is directly distilled by RoBERTa-wwm-ext on the pre-training task, and then fine-tuning in the downstream task, not by distillation on the downstream task. Similar to other models, we only do distillation for pre-training tasks. If you want to further improve the effect on downstream tasks, knowledge distillation can be used again in the fine-tuning stage.

**Q: How can I download XXXXX dataset?**  
A: Some datasets provide download addresses. For datasets without a download address, please search by yourself or contact the original author to obtain the data.

## Citation

[1] [Pre-training with whole word masking for chinese bert](https://ieeexplore.ieee.org/document/9599397)(Cui et al., ACM TASLP 2021)  
[2] [TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](https://aclanthology.org/2020.acl-demos.2) (Yang et al., ACL 2020)   
[3] [CLUE: A Chinese Language Understanding Evaluation Benchmark](https://aclanthology.org/2020.coling-main.419) (Xu et al., COLING 2020)  
[4] [TinyBERT: Distilling BERT for Natural Language Understanding](https://aclanthology.org/2020.findings-emnlp.372) (Jiao et al., Findings of EMNLP 2020)

## Follow us

Welcome to follow the official WeChat account of HFL to keep up with the latest technical developments.  
![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)

## Issues

If you have questions, please submit them in a GitHub Issue.

* Before submitting an issue, please check whether the FAQ can solve the problem, and it is recommended to check whether the previous issue can solve your problem.
* Duplicate and unrelated issues will be handled by [stable-bot](stale · GitHub Marketplace).
* We will try our best to answer your questions, but there is no guarantee that your questions will be answered.
* Politely ask questions and build a harmonious discussion community
