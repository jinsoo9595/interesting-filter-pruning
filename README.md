# interesting-filter-pruning

Among the state-of-the-art neural network pruning methods, we looked at the papers that approached weights in convolutional layers from a new value of "geometric median" rather than "small-norm-less-important".

> He, Yang, et al. "Filter pruning via geometric median for deep convolutional neural networks acceleration." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019. [Filter pruning via geometric median](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html)

# co-researcher
Please visit my **co-researcher's** github as well! (he's~ hansome)  
https://github.com/cloudpark93

# Requirements
- Python 3.8
- Keras 2.4.3
- kerassurgeon 0.2.0


# Models
**VGG16-cifar10**  
This is a customized model that can train CIFAR10, not the ImageNet dataset, and we implemented it based on [easy-filter-pruning](https://github.com/cloudpark93/easy-filter-pruning).

**Original ResNet**  
We proceeded with the pre-trained model provided by Keras [ResNet](https://keras.io/api/applications/).


# To Do
I will add tests for models and datasets from ResNet56(CIFAR10) and ResNet18(ImageNet).
