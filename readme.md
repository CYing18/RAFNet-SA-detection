# RAFNet: Restricted attention fusion network for sleep apnea detection

## Abstract
Sleep apnea (SA) is a common sleep-related breathing disorder, which would lead to damage of multiple systemic organs or even sudden death. In clinical practice, portable device is an important tool to monitor sleep conditions and detect SA events by using physiological signals. However, SA detection performance is still limited due to physiological signals with time-variability and complexity. In this paper, we focus on SA detection with single lead ECG signals, which can be easily collected by a portable device. Under this context, we propose a restricted attention fusion network called RAFNet for sleep apnea detection. Specifically, RR intervals (RRI) and R-peak amplitudes (Rpeak) are generated from ECG signals and divided into one-minute-long segments. To alleviate the problem of insufficient feature information of the target segment, we combine the target segment with two pre- and post-adjacent segments in sequence, (i.e. a five-minute-long segment), as the input. Meanwhile, by leveraging the target segment as the query vector, we propose a new restricted attention mechanism with cascaded morphological and temporal attentions, which can effectively learn the feature information and depress redundant feature information from the adjacent segments with adaptive assigning weight importance. To further improve the SA detection performance, the target and adjacent segment features are fused together with the channel-wise stacking scheme. Experiment results on the public Apnea-ECG dataset and the real clinical FAH-ECG dataset with sleep apnea annotations show that the RAFNet greatly improves SA detection performance and achieves competitive results, which are superior to those achieved by the state-of-the-art baselines.

## Dataset
Apnea-ECG Dataset(public dataset) and FAH-ECG Dataset(real clinical dataset)

## Usage
Download the dataset Apnea-ECG
Run Preprocessing.py to get a file named apnea-ecg.pkl
Per-segment classification
Run RAFNet.py
Per-recording classification
Run evaluate.py
The performance is shown in table_RAFNet.csv


## Requirements
Python==3.6 Keras==2.3.1 TensorFlow==1.14.0

## Cite
If our work is helpful to you, please cite:

```html
@article{chen2023rafnet,
  title={RAFNet: Restricted attention fusion network for sleep apnea detection},
  author={Chen, Ying and Yue, Huijun and Zou, Ruifeng and Lei, Wenbin and Ma, Wenjun and Fan, Xiaomao},
  journal={Neural Networks},
  volume={162},
  pages={571--580},
  year={2023},
  publisher={Elsevier}
}
```
