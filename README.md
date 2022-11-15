[**中文说明**](https://github.com/iflytek/minirbt) | [**English**](https://github.com/iflytek/minirbt/blob/main/README_EN.md)

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


在自然语言处理领域中，预训练语言模型（Pre-trained Language Models）已成为非常重要的基础技术。为了进一步促进中文信息处理的研究发展，哈工大讯飞联合实验室（HFL）基于自主研发的[知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer)，结合了全词掩码（Whole Word Masking）技术和知识蒸馏（Knowledge Distillation）技术推出中文小型预训练模型**MiniRBT**。

----

[中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner)

查看更多哈工大讯飞联合实验室（HFL）发布的资源：https://github.com/iflytek/HFL-Anthology

## 内容导引

| 章节 | 描述 |
|-|-|
| [简介](#简介) | 介绍小型预训练模型所应用的技术方案 |
| [模型下载](#模型下载) | 提供了小型预训练模型的下载地址 |
| [快速加载](#快速加载) | 介绍了如何使用[🤗Transformers](https://github.com/huggingface/transformers)快速加载模型 |
| [模型对比](#模型对比) | 提供了本目录中模型的参数对比 |
| [蒸馏参数](#蒸馏设置) | 预训练蒸馏超参设置 |
| [中文基线系统效果](#中文基线系统效果) | 列举了部分中文基线系统效果 |
| [两段式蒸馏方法](#two-stage) | 列举了两段式蒸馏与一段式蒸馏的效果对比 |
| [预训练](#预训练) | 说明预训练代码的使用方法 |
| [使用建议](#使用建议) | 提供了若干使用中文小型预训练模型的建议 |
| [FAQ](#faq) | 常见问题答疑 |
| [参考文献](#参考文献) | 参考文献 |

## 简介

目前预训练模型存在参数量大，推理时间长，部署难度大的问题，为了减少模型参数及存储空间，加快推理速度，我们推出了实用性强、适用面广的中文小型预训练模型**MiniRBT**，我们采用了如下技术：

* **Whole Word Masking (wwm)**：全词掩码技术是预训练阶段的训练样本生成策略。简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask（替换成[MASK]；保持原词汇；随机替换成另外一个词）。而在WWM中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask。更详细的说明及样例请参考：**[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)**，本工作中我们使用[哈工大LTP](http://ltp.ai)作为分词工具。

* **两段式蒸馏**：相较于教师模型直接蒸馏到学生模型的传统方法，我们采用中间模型辅助教师模型到学生模型蒸馏的两段式蒸馏方法，即教师模型先蒸馏到助教模型（Teacher Assistant），学生模型通过对助教模型蒸馏得到，以此提升学生模型在下游任务的表现。并在下文中贴出了下游任务上两段式蒸馏与一段式蒸馏的实验对比，结果表明两段式蒸馏能取得相比一段式蒸馏更优的效果。

* **构建窄而深的学生模型**。相较于宽而浅的网络结构，如TinyBERT结构（4层，隐层维数312），我们构建了窄而深的网络结构作为学生模型MiniRBT（6层，隐层维数256和288），实验表明窄而深的结构下游任务表现更优异。

**MiniRBT**目前有两个分支模型，分别为**MiniRBT-H256**和**MiniRBT-H288**，表示隐层维数256和288，均为6层Transformer结构，由两段式蒸馏得到。同时为了方便实验效果对比，我们也提供了TinyBERT结构的**RBT4-H312**模型下载。

我们会在近期提供完整的技术报告，敬请期待。

## 模型下载

| 模型简称                       | 层数 | 隐层大小 | 注意力头 | 参数量 |                          Google下载                          |                          百度盘下载                          |
| :----------------------------- | :--: | :------: | :------: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **MiniRBT-h288**               |  6   |   288    |    8     | 12.3M  | [[PyTorch]](https://drive.google.com/file/d/1DDSUZbWORlkpKFfjsZJnsoyh5cQKZhEA/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/1pIwzx0Zu62fLOSQASxTohQ?pwd=7313)<br/>（密码：7313） |
| **MiniRBT-h256**               |  6   |   256    |    8     | 10.4M  | [[PyTorch]](https://drive.google.com/file/d/1M5U1VzfrD82SOGinhauW1N4Xupui68xj/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/16ZMOoliMLsa2KqMKpwT0IA?pwd=iy53)<br/>（密码：iy53） |
| **RBT4-h312** (TinyBERT同大小) |  4   |   312    |    12    | 11.4M  | [[PyTorch]](https://drive.google.com/file/d/1NmvrWvGsJwdXVd6C1Qs48K1G6KYcXVKk/view?usp=sharing) | [[PyTorch]](https://pan.baidu.com/s/11I9ojsnGK-7eZXMkl1k7EQ?pwd=ssdw)<br/>（密码：ssdw） |

也可以直接通过huggingface官网下载模型（PyTorch & TF2）：<https://huggingface.co/hfl>

下载方法：点击任意需要下载的模型 → 选择"Files and versions"选项卡 → 下载对应的模型文件。

## 快速加载

### 使用Huggingface-Transformers

依托于[🤗transformers库](https://github.com/huggingface/transformers)，可轻松调用以上模型。

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
```

**注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！**

对应的`MODEL_NAME` 如下所示：

| 原模型        | 模型调用名                |
| ------------- | ------------------------- |
| MiniRBT-H256 | "hfl/minirbt-h256" |
| MiniRBT-H288  | "hfl/minirbt-h288"  |
| RBT4-H312 | "hfl/rbt4-h312" |

## 模型对比

模型结构细节与参数量汇总如下。

| 模型 | 层数 | 隐层大小 | FFN大小 | 注意力头数 | 模型参数量 | 参数量(除去嵌入层) | 加速比 |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa | 12 | 768 | 3072 | 12 | 102.3M (100%) | 85.7M (100%) | 1x|
| RBT6 (KD) | 6 | 768 | 3072 | 12 | 59.76M (58.4%) | 43.14M (50.3%) | 1.7x|
| RBT3 | 3 | 768 | 3072 | 12 | 38.5M (37.6%) | 21.9M (25.6%) | 2.8x|
| **RBT4-H312**| 4 | 312 | 1200 | 12 | 11.4M (11.1%) | 4.7M (5.5%) | 6.8x|
| **MiniRBT-H256** | 6 | 256 | 1024 | 8 | 10.4M (10.2%) | 4.8M (5.6%) | 6.8x|
| **MiniRBT-H288** | 6 | 288 | 1152 | 8 | 12.3M (12.0%) | 6.1M (7.1%) | 5.7x|

括号内参数量百分比以原始base模型（即RoBERTa-wwm-ext）为基准

* RBT的名字是RoBERTa三个音节首字母组成
* RBT3：由RoBERTa-wwm-ext 3层进行初始化继续预训练得到，更详细的说明请参考：**[Chinese-BERT-wwm 小参数量模型](https://github.com/ymcui/Chinese-BERT-wwm/#%E5%B0%8F%E5%8F%82%E6%95%B0%E9%87%8F%E6%A8%A1%E5%9E%8B)**
* RBT6 (KD)：助教模型，由RoBERTa-wwm-ext 6层进行初始化，通过对RoBERTa-wwm-ext蒸馏得到
* MiniRBT-*：通过对助教模型RBT6(KD)蒸馏得到
* RBT4-H312: 通过对RoBERTa直接蒸馏得到

## 蒸馏设置

| 模型 | Batch Size | Training Steps | Learning Rate | Temperature | Teacher |
| :------- | :---------: | :---------: | :---------: | :---------: |  :---------: |
| RBT6 (KD) | 4096 | 100K<sup>MAX512 | 4e-4 | 8 | RoBERTa-wwm-ext |
| **RBT4-H312**| 4096 | 100K<sup>MAX512 | 4e-4 | 8 | RoBERTa-wwm-ext |
| **MiniRBT-H256** | 4096 | 100K<sup>MAX512 | 4e-4 | 8 | RBT6 (KD) |
| **MiniRBT-H288** | 4096 | 100K<sup>MAX512 | 4e-4 | 8 |  RBT6 (KD)  |

## 中文基线系统效果

为了对比基线效果，我们在以下几个中文数据集上进行了测试。

* [**CMRC 2018**：篇章片段抽取型阅读理解（简体中文）](https://github.com/ymcui/cmrc2018)
* [**DRCD**：篇章片段抽取型阅读理解（繁体中文）](https://github.com/DRCKnowledgeTeam/DRCD)
* [**OCNLI**：原生中文自然语言推断](https://github.com/CLUEbenchmark/OCNLI/tree/main/data/ocnli)
* [**LCQMC**：句对匹配](http://icrc.hitsz.edu.cn/info/1037/1146.htm)
* [**BQ Corpus**：句对匹配](http://icrc.hitsz.edu.cn/Article/show/175.html)
* [**TNEWS**：文本分类](https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip)
* [**ChnSentiCorp**: 情感分析](https://huggingface.co/datasets/seamew/ChnSentiCorp/tree/main)

经过学习率搜索，我们验证了小参数量模型需要更高的学习率和更多的迭代次数，以下是各数据集的学习率

**最佳学习率:**

| 模型 | CMRC 2018 | DRCD | OCNLI | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :------- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| RoBERTa | 3e-5 | 3e-5 | 2e-5 | 2e-5 | 3e-5 | 2e-5 | 2e-5 |
| * | 1e-4 | 1e-4 | 5e-5 | 1e-4 | 1e-4 | 1e-4 | 1e-4 |

*代表所有小型预训练模型 (RBT3, RBT4-H312, MiniRBT-H256, MiniRBT-H288)

**注意：为了保证结果的可靠性，对于同一模型，我们设置epoch分别为2、3、5、10，运行至少3遍（不同随机种子），汇报模型性能平均值的最大值。不出意外，你运行的结果应该很大概率围绕该平均值上下浮动。以下所有实验结果均是在开发集上的实验结果。**

**实验结果：**

| Task               | CMRC 2018  | DRCD        | OCNLI  | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :-------           | :---------:| :---------: | :-----:| :---: | :------:  | :---: | :---------:  |
| RoBERTa            | 87.3/68    | 94.4/89.4   | 76.58  | 89.07 | 85.76     | 57.66 |     94.89    |
| RBT6 (KD)   | 84.4/64.3  | 91.27/84.93 | 72.83  | 88.52 | 84.54     | 55.52 |     93.42    |
| RBT3               | 80.3/57.73 | 85.87/77.63 | 69.80  | 87.3  | 84.47     | 55.39 |     93.86    |
| **RBT4-H312**       | 77.9/54.93 | 84.13/75.07 | 68.50  | 85.49 | 83.42     | 54.15 |     93.31    |
| **MiniRBT-H256** | 78.47/56.27| 86.83/78.57 | 68.73  | 86.81 | 83.68     | 54.45 |     92.97    |
| **MiniRBT-H288** | 80.53/58.83| 87.1/78.73  | 68.32  | 86.38 | 83.77     | 54.62 |     92.83    |

**相对效果：**

| Task               | CMRC 2018   | DRCD        | OCNLI | LCQMC | BQ Corpus | TNEWS | ChnSentiCorp |
| :-------           | :---------: | :---------: | :----:| :---: | :-------: | :---: | :---------:  |
| RoBERTa            | 100%/100%   | 100%/100%   | 100%  | 100%  |    100%   | 100%  |     100%     |
| RBT6 (KD) | 96.7%/94.6% | 96.7%/95%   | 95.1% | 99.4% |    98.6%  | 96.3% |     98.5%    |
| RBT3               | 92%/84.9%   | 91%/86.8%   | 91.1% | 98%   |    98.5%  | 96.1% |     98.9%    |
| **RBT4-H312**       | 89.2%/80.8% | 89.1%/84%   | 89.4% | 96%   |    97.3%  | 93.9% |     98.3%    |
| **MiniRBT-H256** | 89.9%/82.8% | 92%/87.9%   | 89.7% | 97.5% |    97.6%  | 94.4% |     98%      |
| **MiniRBT-H288** | 92.2%/86.5% | 92.3%/88.1% | 89.2% | 97%   |    97.7%  | 94.7% |     97.8%    |

<h2 id="two-stage">两段式蒸馏对比<sup>†</sup></h2>

我们对两段式蒸馏(RoBERTa→RBT6(KD)→MiniRBT-H256)与一段式蒸馏(RoBERTa→MiniRBT-H256)做了比较。实验结果证明两段式蒸馏效果较优。

| 模型                   | CMRC 2018      | OCNLI     | LCQMC     | BQ Corpus | TNEWS     |
| :-------               | :---------:    | :-------: | :-------: | :-------: | :-------: |
| MiniRBT-H256（两段式）| **77.97/54.6** | **69.11** | **86.58** | **83.74** | **54.12** |
| MiniRBT-H256（一段式）| 77.57/54.27    | 68.32     | 86.39     | 83.55     |     53.94 |

<sup>†</sup>:该表中预训练模型经过3万步蒸馏，不同于中文基线效果中呈现的模型。

## 预训练

我们使用了[TextBrewer工具包](https://github.com/airaria/TextBrewer)实现知识蒸馏预训练过程。完整的训练代码位于`pretraining`目录下。

### 代码结构

* `dataset`:
  * `train`: 训练集
  * `dev`： 验证集
* `distill_configs`: 学生模型结构配置文件
* `jsons`: 数据集配置文件
* `pretrained_model_path`:
  * `ltp`: ltp分词模型权重,包含`pytorch_model.bin`，`vocab.txt`，`config.json`，共计3个文件
  * `RoBERTa`: 教师模型权重，包含`pytorch_model.bin`，`vocab.txt`，`config.json`，共计3个文件
* `scripts`: 模型初始化权重生成脚本
* `saves`: 输出文件夹
* `config.py`: 训练参数配置
* `matches.py`: 教师模型和学生模型的匹配配置
* `my_datasets.py`: 训练数据处理文件
* `run_chinese_ref.py`: 生成含有分词信息的参考文件
* `train.py`：预训练主函数
* `utils.py`: 预训练蒸馏相关函数定义
* `distill.sh`: 预训练蒸馏脚本

### 环境准备

预训练代码所需依赖库仅在python3.8，PyTorch v1.8.1下测试过，一些特定依赖库可通过`pip install -r requirements.txt`命令安装

#### 预训练模型准备

可从[huggingface官网](https://huggingface.co/models)下载`ltp`分词模型权重与`RoBERTa-wwm-ext`预训练模型权重，并存放至`${project-dir}/pretrained_model_path/`目录下相应文件夹

#### 数据准备

对于中文模型，我们需先生成含有分词信息的参考文件，可直接运行以下命令

```sh
python run_chinese_ref.py
```

因为预训练数据集较大，推荐生成参考文件后进行预处理，仅需运行以下命令

```sh
python my_datasets.py
```

#### 运行训练脚本

一旦你对数据做了预处理，进行预训练蒸馏就非常简单。我们在`distill.sh`中提供了预训练示例脚本。 该脚本支持单机多卡训练，主要包含如下参数:

* `teacher_name or_path`：教师模型权重文件
* `student_config`: 学生模型结构配置文件
* `num_train_steps`: 训练步数
* `ckpt_steps`：每ckpt_steps保存一次模型
* `learning_rate`: 预训练最大学习率
* `train_batch_size`: 预训练批次大小
* `data_files_json`: 数据集json文件
* `data_cache_dir`：训练数据缓存文件夹
* `output_dir`: 输出文件夹
* `output encoded layers`：设置隐层输出为True
* `gradient_accumulation_steps`：梯度累积
* `temperature`：蒸馏温度
* `fp16`：开启半精度浮点数训练

直接运行以下命令可实现MiniRBT-H256的预训练蒸馏

```bash
sh distill.sh
```

**提示**：以良好的模型权重初始化有助于蒸馏预训练。在我们的实验中，我们通过教师模型的6层初始化我们的助教模型RBT6(KD) ! 请参考`scripts/init_checkpoint_TA.py`来创建有效的初始化权重，并使用`--student_pretrained_weights`参数将此初始化用于蒸馏训练!

## 使用建议

* 初始学习率是非常重要的一个参数，需要根据目标任务进行调整。
* 小参数量模型的最佳学习率和`RoBERT-wwm`相差较大，所以使用小参数量模型时请务必调整学习率（基于以上实验结果，小参数量模型需要的初始学习率高，迭代次数更多）。
* 在参数量（不包括嵌入层）基本相同的情况下，**MiniRBT-H256**的效果优于**RBT4-H312**，亦证明窄而深的模型结构优于宽而浅的模型结构
* 在阅读理解相关任务上，**MiniRBT-H288**的效果较好。其他任务**MiniRBT-H288**和**MiniRBT-H256**效果持平，可根据实际需求选择相应模型。

## FAQ

**Q: 这个模型怎么用？**  
A: 参考[快速加载](#快速加载)。使用方式和HFL推出的中文预训练模型系列如RoBERTa-wwm相同。 

**Q:为什么要单独生成含有分词信息的参考文件？**  
A: 假设我们有一个中文句子:`天气很好`，BERT将它标记为`['天'，'气'，'很'，'好']`(字符级别)。但在中文中`天气`是一个完整的单词。为了实现全词掩码，我们需要一个参考文件来告诉模型应该在哪个位置添加`##`，因此会生成类似于`['天'，'##气'，'很'，'好']`的结果。  
**注意：此为辅助参考文件，并不影响模型的原始输入（即与分词结果无关）。**

**Q: 为什么RBT6 (KD)在下游任务中的效果相较RoBERTa下降这么多?  为什么miniRBT-H256/miniRBT-H288/RBT4-H312效果这么低？如何提升效果？**  
A: 上文中所述RBT6 (KD)直接由RoBERTa-wwm-ext在预训练任务上蒸馏得到，然后在下游任务中fine-tuning，并不是通过对下游任务蒸馏得到。其他模型类似，我们仅做了预训练任务的蒸馏。如果希望进一步提升在下游任务上的效果，可在fine-tuning阶段再次使用知识蒸馏。

**Q: 某某数据集在哪里下载？**  
A: 部分数据集提供了下载地址。未标注下载地址的数据集请自行搜索或与原作者联系获取数据。

## 参考文献

[1] [Pre-training with whole word masking for chinese bert](https://ieeexplore.ieee.org/document/9599397) (Cui et al., IEEE/ACM TASLP 2021)  
[2] [TextBrewer: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing](https://aclanthology.org/2020.acl-demos.2) (Yang et al., ACL 2020)  
[3] [CLUE: A Chinese Language Understanding Evaluation Benchmark](https://aclanthology.org/2020.coling-main.419) (Xu et al., COLING 2020)  
[4] [TinyBERT: Distilling BERT for Natural Language Understanding](https://aclanthology.org/2020.findings-emnlp.372) (Jiao et al., Findings of EMNLP 2020)

## 关注我们

欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)

## 问题反馈

如有问题，请在GitHub Issue中提交。

* 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
* 重复以及与本项目无关的issue会被[stable-bot](stale · GitHub Marketplace)处理，敬请谅解。
* 我们会尽可能的解答你的问题，但无法保证你的问题一定会被解答。
* 礼貌地提出问题，构建和谐的讨论社区。
