<img width="1000" alt="image" src="Asset/Title.png">
   <p align="center">
    <a href="https://scholar.google.com.hk/citations?user=1yhGS5sAAAAJ&hl=zh-CN"><strong>Gan Pei <sup>1</sup><sup>*</sup></strong></a>
    .
    <a href=""><strong>Junhao Ning<sup>1</sup></strong></a>
    .
    <a href=""><strong>Chenrui Niu<sup>1</sup></strong></a>
    .
    <a href="https://scholar.google.com.hk/citations?user=E6zbSYgAAAAJ&hl=zh-CN"><strong>Guangtao Zhai<sup>2</sup></strong></a>
    .
    <a href=""><strong>Siqiong Yao<sup>2</sup></strong></a>
    .
    <a href="https://scholar.google.com.hk/citations?user=8-Vo9cUAAAAJ&hl=zh-CN"><strong>Menghan Hu<sup>1</sup><sup>#</sup></strong></a>
</p>
<p align="center">
    <strong><sup>1</sup>East China Normal University</strong> &nbsp;&nbsp;&nbsp; <strong><sup>2</sup>Shanghai Jiao Tong University</strong>
   
### ✨Abstract
For non-contact respiratory rate (RR) measurement, effectively addressing the interference from continuous motion artifacts remains a significant challenge. Most existing research focuses on the removal of weak motion artifacts in a two-dimensional plane, and the fixed spatial scale of the scenes limits the generalization of these methods to real-world scenarios, especially in real walking scenarios. To tackle this issue, we propose an RR measurement framework based on a multi-strategy fusion motion artifact suppression algorithm and have constructed a real-world walking dataset. Specifically, the framework consists of three core modules: an ROI automatic selection and adaptive enhancement module to guide the selection of high-quality corner points; a signal quality evaluation module that adaptively assesses whether the signal is noisy, preventing blind denoising; and a multi-strategy fusion motion artifact removal module that dynamically selects the appropriate strategy to suppress motion interference. To the best of our knowledge, this is the first study to investigate the task of video-based RR measurement in real walking scenarios. Experimental results demonstrate that the method achieves state-of-the-art performance across multiple datasets, with a mean absolute error (MAE) of **1.04 breaths per minute (bpm) on the COHFACE**, **3.17 bpm on the OVRM-Walking dataset**, and an average MAE of just **2.41 bpm on the in-house real-world walking dataset**, which includes both indoor and outdoor scenarios. This study broadens the applicability of camera-based non-contact RR detection technology.
   
### ✨Highlight
[1]  An adaptive edge enhancement module that integrates RGB three-channel features is proposed, enabling high-quality corner point selection in distant, low-light, and low-resolution ROI regions. 

[2]  A time-domain feature-based waveform quality assessment module is proposed, enabling on-demand activation of motion artifact removal to prevent performance degradation caused by excessive denoising.

[3]  A multi-strategy fusion-based adaptive noise removal method is proposed, which adaptively chooses SCR, ASS, and TF-FastICA denoising algorithms based on the signal’s spectral characteristics, effectively removing motion artifacts. To the best of our knowledge, this is the first study to investigate the task of video-based RR measurement in real walking scenarios. The proposed method exhibits superior performance compared to state-of-the-art (SOTA) methods on the in-house Walking Breathing dataset, OVRM-Walking dataset and COHFACE dataset.

[4]  A real-word walking dataset has been constructed, consisting of 600 video samples collected from both indoor and outdoor natural lighting environments, filling the gap of missing real-word walking datasets.


### ✨Pipeline
<p align="center">
<img width="1000" alt="image" src="Asset/Pipeline.png">
 <p align="center">

### ✨In-house Walking Breathing Dataset
The dataset includes two scenes, indoor and outdoor, with 300 samples for each scene. For dataset requests, please contact the author via email.

### Contact
```
51265904018@stu.ecnu.edu.cn
```
