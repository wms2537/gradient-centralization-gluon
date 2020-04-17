# gradient-centralization-gluon

Gradient Centralization implementation (unofficial) in mxnet gluon.
# Gradient Centralization

## [Gradient Centralization: A New Optimization Technique for Deep Neural Networks](https://arxiv.org/abs/2004.01461)

***

## Introduction

* Gradient Centralization (GC) is a simple and effective optimization technique for Deep Neural Networks (DNNs), which operates directly on gradients by centralizing the gradient vectors to have zero mean. It can both speedup training process and improve the final generalization performance of DNNs. GC is very simple to implement and can be easily embedded into existing gradient based DNN optimizers with only few lines of code. It can also be directly used to finetune the pre-trained DNNs.

<div  align="center"><img src="imgs/gradient.png" height="45%" width="45%" alt="Illustration of the GC operation on gradient matrix/tensor of weights in the fully-connected layer (left) and convolutional layer (right)."/></div>

* GC can be viewed as a projected gradient descent method with a constrained loss function.  The Lipschitzness of the constrained loss function and its gradient is better so that the training process becomes more efficient and stable.   Our experiments on various applications, including `general image classification`, `fine-grained image classification`, `detection and segmentation` and `Person ReID` demonstrate that GC can consistently improve the performance of DNN learning. 

<div  align="center"><img src="imgs/projected_Grad.png" height="50%" width="50%" alt=""/></div>

## Reference
[Original Repo](https://github.com/Yonghongwei/Gradient-Centralization)
