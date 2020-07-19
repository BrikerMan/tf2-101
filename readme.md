<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<h1 align="center">
    <a>Tensorflow 2.0 技术实战详解</a>
</h1>

这里整理了本书所有的代码、基础数据集、扩展数据集和一些扩展阅读资源。建议在阅读每一章之前先把基础数据集下载到指定的路径，再配合该章节的笔记本阅读书上内容。

**本书所有代码除特殊说明均在 jupyer lab 上执行**

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->

# 目录

- [目录](#目录)
- [第 1 章 环境配置](#第-1-章-环境配置)
  - [1.1 云端 Notebook 环境介绍](#11-云端-notebook-环境介绍)
  - [1.2 本地 Notebook 环境准备](#12-本地-notebook-环境准备)
  - [1.3 代码规范介绍](#13-代码规范介绍)
- [第 2 章 常见工具介绍](#第-2-章-常见工具介绍)
  - [2.1 扩展资料](#21-扩展资料)
- [第 3 章 从零搭建神经网络](#第-3-章-从零搭建神经网络)
  - [3.1 扩展资料](#31-扩展资料)
- [第 4 章 深度学习基础](#第-4-章-深度学习基础)
  - [4.1 扩展资料](#41-扩展资料)
- [第 5 章 泰坦尼克幸存者预测](#第-5-章-泰坦尼克幸存者预测)
  - [5.1 扩展资料](#51-扩展资料)
- [第 6 章 TensorFlow 2.0 介绍](#第-6-章-tensorflow-20-介绍)
  - [6.1 扩展资料](#61-扩展资料)
- [第 7 章 图像识别入门](#第-7-章-图像识别入门)
  - [7.1 扩展资料](#71-扩展资料)
- [第 8 章 图像识别进阶](#第-8-章-图像识别进阶)
  - [8.1 扩展资料](#81-扩展资料)

# 第 1 章 环境配置

在本章中，你将学习如何准备和使用深度学习 Notebook 环境以及本书的代码规范。
本书代码环境为 Python 3.6+ 和 Tensorflow 2.0，所有的代码需要在 Notebook 环境中执行。

## 1.1 云端 Notebook 环境介绍

配置本地 GPU 环境比较麻烦，推荐读者们使用云端 Notebook 平台。
云 Notebook 环境都提供了 CPU 环境和 GPU 环境，对于不要求很大算力的项目，建议使用 CPU 环境，以免浪费资源。

| 平台                | 是否收费 | 需要外网 | 相关文章                            |
| ------------------- | -------- | -------- | -----------------------------------|
| [OpenBayes]（推荐） | 是       | 否       | [OpenBayes 下识别手写数字]          |
| [Kaggle]            | 否       | 否       | [如何用 Kaggle Kernels 免费使用GPU] |
| [Colab]             | 否       | 是       | [设置Google-colab使用免费GPU]       |

## 1.2 本地 Notebook 环境准备

配置本地 Notebook 环境建议使用 Anaconda，在 Ubuntu/Mac OS X 系统环境安装。

- [Anaconda 官网](https://www.anaconda.com/distribution/#download-section)

## 1.3 代码规范介绍

- [PEP8 代码规范](https://juejin.im/post/58b129b32f301e006c035a62)
- [全面理解 Python 中的类型提示（Type Hints）](https://sikasjc.github.io/2018/07/14/type-hint-in-python/)

# 第 2 章 常见工具介绍

在本章中，你将学习 Python 数据处理中最常用的三个工具 Numpy, Pandas, Matplotlib。几乎每一个实验都会用到这几个工具。熟练掌握它们是学习深度学习中的第一步。

- [代码 Notebook 文件](chapter-02.ipynb) 建议使用 CPU 运行环境。

## 2.1 扩展资料

- [NumPy 官方快速入门教程(译)](https://juejin.im/post/5a76d2c56fb9a063557d8357)
- [十分钟的 pandas 入门教程](https://ericfu.me/10-minutes-to-pandas/)
- [Matplotlib 教程 | 菜鸟教程](https://www.runoob.com/w3cnote/matplotlib-tutorial.html)
- [Pandas Profiling-一键生成数据报告](https://mathpretty.com/11152.html)

# 第 3 章 从零搭建神经网络

在本章中您将通过动手实现一个神经网络来学习神经网络基础知识。由于本章重点在于动手实现，有不少知识点一带而过，所以实现过程遇到不懂的概念和公式不要慌，继续按照代码示例把神经网络实现了。实现完成后继续看第4章，看完第4章回过头来再看一遍第3章就能理解大部分内容。至于数学公式和推导，您只需要知道哪个阶段用了什么公式即可，并不要求掌握具体的推导过程。

- [代码 Notebook 文件](chapter-03.ipynb) 建议使用 CPU 运行环境。

## 3.1 扩展资料

- [Machine Learning for Beginners: An Introduction to Neural Networks](https://victorzhou.com/blog/intro-to-neural-networks/)

# 第 4 章 深度学习基础

本章您将学习深度学习的基本概念、模型评估方案以及如何解决模型的欠拟合过拟合。尽管深度学习中概念非常多，背后涉及大量的数学知识，但初学阶段不用太担心，建议先大体了解这些概念，再通过一个个实践项目去深入理解。

## 4.1 扩展资料

- [Google 机器学习速成课程 - 机器学习简介](https://developers.google.cn/machine-learning/crash-course/ml-intro)

# 第 5 章 泰坦尼克幸存者预测

在本章中你将通过搭建一个神经网络模型，了解深度学习的工作流程。读者们如果能够按照本章代码重现实验结果，完成 hello world 项目，就达成了学习目标。

- [代码 Notebook 文件](chapter-05.ipynb) 建议使用 CPU 运行环境。

## 5.1 扩展资料

- [Kaggle 竞赛 - Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

# 第 6 章 TensorFlow 2.0 介绍

在本章中，您将学习 TensorFlow 2.0 版本的新特性，模型保存方法，训练回调函数和可视化。模型保存和训练回调是所有深度学习任务必须要掌握的技能，训练可视化则可以帮助您更好地理解和调试模型。

- [代码 Notebook 文件](chapter-06.ipynb) 建议使用 CPU 运行环境。

## 6.1 扩展资料

- [初学者的 TensorFlow 2.0 教程](https://www.tensorflow.org/tutorials/quickstart/beginner)

# 第 7 章 图像识别入门

在本章中，您将构建一个简单的模型入门图像识别，然后用卷积神经网络来优化图像识别的效果。

- [代码 Notebook 文件](chapter-07.ipynb) 建议使用 CPU 运行环境。

## 7.1 扩展资料

- [cs231n - Convolutional Neural Networks (CNNs/ConvNets)](https://cs231n.github.io/convolutional-networks/)

# 第 8 章 图像识别进阶

本章我们通过一个花朵种类分类问题进一步学习图像识别。真实的图像识别问题中，需要从磁盘读取图片文件，进行预处理和数据增强才能开始训练模型。除了数据增强，我们还可以通过迁移学习的方案大幅度降低训练成本，快速获得表现很好的模型。

**数据集下载**

下载 TensorFlow 官方提供的花分类数据集
（[https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)），
并且解压到 `data/flower_photos` 目录下。

**新增依赖**

```bash
pip install pillow
pip install tensorflow_hub
```

- [代码 Notebook 文件](chapter-07.ipynb) 建议使用 GPU 运行环境。

## 8.1 扩展资料

- [Pillow 框架介绍](https://www.liaoxuefeng.com/wiki/1016959663602400/1017785454949568)
- [TensorFlow Hub 官网](https://www.tensorflow.org/hub)

<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
[OpenBayes]: https://openbayes.com/
[OpenBayes 下识别手写数字]: https://openbayes.com/docs/tutorial-mnist/
[Kaggle]: https://www.kaggle.com
[如何用 Kaggle Kernels 免费使用GPU]: https://zhuanlan.zhihu.com/p/36824585
[Colab]: https://colab.research.google.com/
[设置Google-colab使用免费GPU]: https://gabriel1225.github.io/%E8%AE%BE%E7%BD%AEGoogle-colab%E4%BD%BF%E7%94%A8%E5%85%8D%E8%B4%B9GPU.html
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->