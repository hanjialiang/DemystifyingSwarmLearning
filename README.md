# Demystifying Swarm Learning

This repository contains code for the paper [**Demystifying Swarm Learning: An Emerging Decentralized Federated Learning System**](https://ieeexplore.ieee.org/document/10701316).

If you find this code useful, please cite our work:

```latex
@inproceedings{DBLP:conf/ccgrid/HanHZJ024,
  author       = {Jialiang Han and
                  Yudong Han and
                  Ying Zhang and
                  Xiang Jing and
                  Yun Ma},
  title        = {Demystifying Swarm Learning: An Emerging Decentralized Federated Learning
                  System},
  booktitle    = {24th {IEEE} International Symposium on Cluster, Cloud and Internet
                  Computing, CCGrid 2024, Philadelphia, PA, USA, May 6-9, 2024},
  pages        = {367--373},
  publisher    = {{IEEE}},
  year         = {2024}
}
```
## Abstract

Federated learning (FL) is a privacy-preserving deep learning paradigm. An important type of FL is cross-silo FL, which enables a moderate number of organizations to cooperatively train a shared model while keeping private data locally and aggregating parameters on a central parameter server. However, the central server may be vulnerable to malicious attacks or software failures. To address this problem, Swarm Learning (SL) has emerged to perform FL in a decentralized manner by introducing a blockchain to securely onboard members and dynamically elect the leader for parameter aggregation. Despite tremendous attention to SL recently, few measurement studies provide comprehensive knowledge of best practices and precautions for deploying SL in real-world scenarios. To this end, we conduct the *first* empirical study of SL, to fill the knowledge gap between SL research and real-world deployment. We conduct various experiments on 3 public datasets for 4 research questions, present interesting findings, quantitatively analyze the reasons behind these findings, and provide developers and researchers with practical suggestions. The findings have evidenced that SL is supposed to be suitable for most application scenarios, even when the dataset is unbalanced or polluted. However, some SL edge nodes consume much more network bandwidth than others, leading to unfairness among participants.

## Datasets

[NIH chest X-ray](https://www.kaggle.com/nih-chest-xrays/data)

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

[IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)
