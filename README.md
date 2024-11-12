# Human Activity Recognition through Augmented WiFi CSI Signals by Lightweight Attention-GRU

**<span style="font-size:100px;">This repository contains the source code for [Human Activity Recognition through Augmented Wi-Fi CSI Signals using Lightweight Attention-GRU model using Pruning].</span>**

## Overview

**<span style="font-size:100px;">This project provides the source code for reproducing the algorithms and experiments described in the [Human Activity Recognition through Augmented Wi-Fi CSI Signals using Lightweight Attention-GRU model using Pruning] paper. Through this source code, you can reproduce the results of the paper and conduct additional experiments. </span>**

**<span style="font-size:60px;">The figure below shows the model architecture, which includes a single layer of GRU and an attention module.</span>**  
<div align="center">
  <img src="https://github.com/harikang/prunedAttentionGRU/blob/main/fig/pretrain_git.png" alt="Model Architecture">
</div>

**<span style="font-size:60px;"> We applied pruning to the model to reduce its size, and the process is illustrated in the figure below.</span>**  

![Overview](https://github.com/harikang/prunedAttentionGRU/blob/main/fig/overview_git.png)

**<span style="font-size:60px;"> By sharing the results of only the ARIL Dataset, we can see the distribution of weights before and after pruning as shown in the figure below. Additionally, the confusion matrix and the history of loss and accuracy help us understand the performance of our model.  </span>**

[1] Distribution of Weight before and after pruning  

<div align="center">
  <img src="https://github.com/harikang/prunedAttentionGRU/blob/main/fig/distribution.png" alt="Distribution">
</div>

[2] Confusion Matrix of ARIL Dataset  

<div align="center">
  <img src="https://github.com/harikang/prunedAttentionGRU/blob/main/fig/128_32_confusionmatrix.png" alt="confusionmatrix">
</div>

[3] Plot of Loss and Accuracy history  

![history](https://github.com/harikang/prunedAttentionGRU/blob/main/fig/128_32.png)  

## Contents

- `tools/`: Experimental results and visualization materials
- `ARIL/`, `HAR/`, `SignFi/`, `StanFi/`: Datasets used for experiments
- `test.py`, `train.py`: Scripts for experiments and model training
- `results/`: Experimental results and visualization materials

## Installation and Usage

### Requirements

- Python 3.x
- Pytorch: 1.9.x
- NumPy: 1.2x.x

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/harikang/prunedAttentionGRU.git
    cd prunedAttentionGRU
    ```

2. Install the required libraries:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Prepare the data. (ARIL, HAR, StanWiFi, SignFi)
2. Run the main experiment:

    ```bash
    python main.py --dataset har-1 --batchsize 128 --learningrate 1e-3 --epochs 100 --verbose
    ```

3. Check the results. The results can be verified through graphs of 'loss' and 'Accuracy' history, as well as through confusion matrix and f1-score.

## Research Support

The authors acknowledge the grant support offered by the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education, Science and Technology (Grant number: NRF-2021R1A2C1093425) and the National Research Foundation of Korea under the Basic Research Laboratory program (Grant number: NRF-2022R1A4A2000748).

## Contact

If you have any questions or would like to discuss, please contact hariver1220@yonsei.ac.kr.

The datasets used in the experiments can be downloaded from the following sources:

- ARIL: https://github.com/geekfeiw/ARIL
- SignFi: https://yongsen.github.io/SignFi/
- StanWiFi: https://www.researchgate.net/figure/Summary-of-the-StanWiFi-dataset_tbl2_366717201
- HAR: https://www.semanticscholar.org/paper/Human-Activity-Recognition-Using-CSI-Information-Sch%C3%A4fer-Barrsiwal/9ddb0cf17a3ac4e9d73bd7df525ff66ab2af73d1

Please check the dataset licenses before use.

### Experimental Environment

- Operating System: Windows 11 Enterprise 64-bit (version 23H2, OS build 22631.3880)
- Hardware:
  - CPU: 11th Gen Intel(R) Core(TM) i7-11700K @ 3.60GHz
  - RAM: 32GB
  - GPU: 2 x RTX3060
- Software:
  - Python: Python 3.8.8
  - Pytorch: 2.0.1+cpu
  - NumPy: 1.21.0
