## Anchor-Based Spatial-Temporal Attention Model for Dynamic 3D Point Cloud Sequences

Created by Guangming Wang</a>, Hanwen Liu</a>, Muyao Chen</a>, Yehui Yang</a>, Zhe Liu</a> and Hesheng Wang</a> from ShangHai Jiao Tong University.

[[arXiv]](https://arxiv.org/abs/2012.10860)


## Citation
If you find this work useful in your research, please cite:
```
@article{wang2021anchor,
title={Anchor-Based Spatio-Temporal Attention 3-D Convolutional Networks for Dynamic 3-D Point Cloud Sequences},
author={Wang, Guangming and Liu, Hanwen and Chen, Muyao and Yang, Yehui and Liu, Zhe and Wang, Hesheng},
journal={IEEE Transactions on Instrumentation and Measurement},
volume={70},
pages={1--11},
year={2021},
publisher={IEEE}
}
```

## Abstract

With the rapid development of measurement technology,  LiDAR and depth cameras are widely used in the perception of the 3D environment. Recent learning based methods for robot perception most focus on the image or video, but deep learning methods for dynamic 3D point cloud sequences are underexplored. Therefore, developing efficient and accurate perception method compatible with these advanced instruments is pivotal to autonomous driving and service robots. An Anchor-based Spatio-Temporal Attention 3D Convolution operation (ASTA3DConv) is proposed in this paper to process dynamic 3D point cloud sequences. The proposed convolution operation builds a regular receptive field around each point by setting several virtual anchors around each point. The features of neighborhood points are firstly aggregated to each anchor based on the spatio-temporal attention mechanism. Then, anchor-based 3D convolution is adopted to aggregate these anchors' features to the core points. The proposed method makes better use of the structured information within the local region and learns spatio-temporal embedding features from dynamic 3D point cloud sequences. Anchor-based Spatio-Temporal Attention 3D Convolutional Neural Networks (ASTA3DCNNs) are built for classification and segmentation tasks based on the proposed ASTA3DConv and evaluated on action recognition and semantic segmentation tasks. The experiments and ablation studies on MSRAction3D and Synthia datasets demonstrate the superior performance and effectiveness of our method for dynamic 3D point cloud sequences. Our method achieves the state-of-the-art performance among the methods with dynamic 3D point cloud sequences as input on MSRAction3D and Synthia datasets.

## Installation

Install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code is tested under TF1.9.0 GPU version, g++ 5.4.0, CUDA 9.0 and Python 3.5 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`. It's highly recommended that you have access to GPUs.




## Compile Customized TF Operators
The TF operators are included under `tf_ops`, you have to compile them first by `make` under each ops subfolder (check `Makefile`). **Update** `arch` **in the Makefiles for different** <a href="https://en.wikipedia.org/wiki/CUDA#GPUs_supported">CUDA Compute Capability</a> **that suits your GPU if necessary**.

## Action Classification Experiments on MSRAction3D

The code for action classification experiments on <a href="https://drive.google.com/file/d/1djwAK3oZTAIFbCz531eClxINmsZgGO_H/view?usp=sharing">MSRAction3D dataset</a> is in `action/`. Check `action_cls/README.md` for more information on data preprocessing and experiments.

## Semantic Segmentation Experiments on Synthia

The code for semantic segmentation experiments on <a href="http://synthia-dataset.net/downloads/">Synthia dataset</a> is in `semantic/`. Check `semantic/semantic_seg_synthia/README.md` for more information on data preprocessing and experiments.


## Acknowlegements

We are grateful to Xingyu Liu for his <a href="https://github.com/xingyul/meteornet">github repository</a>. Our code is based on theirs.
