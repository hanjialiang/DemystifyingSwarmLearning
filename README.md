# SwarmSense

## SwarmUpdating: Optimization of Swarm Edge Node Updating Strategies for Swarm Learning

## Background

+ Existing Swarm Learning (SL) frameworks update all SL nodes uniformly, with little consideration of the individualized or differentiated requirements of the nodes, such as medical images of different races, chip images of different batches of different industries, or mainstream product categories of different e-commerce platforms.
+ Existing SL frameworks cannot predict unbalanced labels effectively.

### Approach

+ **Local node enhanced calibration**: introduce the delta of the gradient of the current node into the process of gradient update for calibration
+ **Neighbor node enhanced up-sampling**: introduce the delta of the gradient of the minority class of neighbor nodes into the process of gradient update for up-sampling
  - Different from up-sampling from simulated data, better performance in theory

### Evaluation

+ Because we cannot acquire the core code of updating strategy of [Swarm Learning](https://github.com/HewlettPackard/swarm-learning), we simulate the experimental results in a distributed machine learning manner.
  - Expect to achieve similar accuracy to that of SL
+ Dataset: Sense++

## An Empirical Study of Swarm Learning

## Background

Swarm Learning (SL) is a newly emerging distributed machine learning paradigm for decentralized, privacy-preserving, secure, *p2p* machine learning, without a central coordinator. There lacks a comprehensive study on performance, applicability, and challenges of SL.

## Dataset

+ Sense++
+ [User behavior](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649)
+ [NIH chest X-ray](https://www.kaggle.com/nih-chest-xrays/data)
+ [Places-365](http://places2.csail.mit.edu/)
+ [ImageNet-2012](http://www.image-net.org/)
+ [Places-LT](https://liuziwei7.github.io/projects/LongTail.html)
+ [ImageNet-LT](https://liuziwei7.github.io/projects/LongTail.html)
+ [iNaturalist-2021](https://github.com/visipedia/inat_comp/tree/master/2021)

## RQ1 What's the performance of SL?

1. Accuracy
2. Time consuming
3. Computation resource consuming
4. Communication overhead

## RQ2 What's the applicability of SL?

1. Does SL support most mainstream neural network structures?
2. Does SL support most mainstream ML platform and GPU accelerator?
3. Does SL support customized neural network modules or optimization methods?

## RQ3 Are there any challenges in SL?

1. Unbalanced labeled classes in each node
2. Unbalanced samples among swarm nodes
3. Individualized or differentiated requirements of edge nodes









