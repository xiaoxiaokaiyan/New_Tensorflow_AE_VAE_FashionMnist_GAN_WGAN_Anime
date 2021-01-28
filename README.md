# 心得：**AE、VAE、GAN网络的创建**

## Theory
* AE原理图
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/theory/AE%E5%8E%9F%E7%90%86%E5%9B%BE.png" width = 100% height =50%  div align=left />

* VAE原理图
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/theory/VAE%E5%8E%9F%E7%90%86%E5%9B%BE1.png" width = 100% height = 50% div align=left />
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/theory/VAE%E5%8E%9F%E7%90%86%E5%9B%BE2.png" width = 100% height =50% div align=left />
&nbsp;
<br/>


## Dependencies:
* &gt; GeForce GTX 1660TI
* Windows10
* python==3.6.12
* tensorflow-gpu==2.0.0
* GPU环境安装包，下载地址：https://pan.baidu.com/s/14Oisbo9cZpP7INQ6T-3vwA 提取码：z4pl （网上找的）
```
  Anaconda3-5.2.0-Windows-x86_64.exe
  cuda_10.0.130_411.31_win10.exe
  cudnn-10.0-windows10-x64-v7.4.2.24.zip
  h5py-2.8.0rc1-cp36-cp36m-win_amd64.whl
  numpy-1.16.4-cp36-cp36m-win_amd64.whl
  tensorflow_gpu-1.13.1-cp36-cp36m-win_amd64.whl
  torch-1.1.0-cp36-cp36m-win_amd64.whl
  torchvision-0.3.0-cp36-cp36m-win_amd64.whl
```
<br/>


## Visualization Results
* AE生成结果对比
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/AE%E7%94%9F%E6%88%90%E7%BB%93%E6%9E%9C%E5%AF%B9%E6%AF%94%E5%9B%BE%E7%89%87.png" width = 50% height =50%  div align=center />

* VAE随机生成第1代
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/VAE%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E7%AC%AC1%E4%BB%A3%E5%9B%BE%E7%89%87.png" width = 50% height =50%  div align=center />


* VAE随机生成第9代
<img src="https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_GAN_FashionMnist/blob/master/result/VAE%E9%9A%8F%E6%9C%BA%E7%94%9F%E6%88%90%E7%AC%AC9%E4%BB%A3%E5%9B%BE%E7%89%87.png" width = 50% height =50% div align=center />
&nbsp;
<br/>


## Public Datasets:
* fashion_mnist，是一个替代MNIST手写数字集的图像数据集。它是由Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自10种类别的共7万个不同商品的正面图片。Fashion-MNIST的大小、格式和训练集/测试集划分与原始的MNIST完全一致。60000/10000的训练测试数据划分，28x28的灰度图片。你可以直接用它来测试你的机器学习和深度学习算法性能，且不需要改动任何的代码。
<br/>


## Experience：
### （1）代码问题
```
      # [b, 28, 28] => [b, 28, 28]
      x_concat1 = tf.concat([x, x_hat], axis=0)

      # [b, 28, 28] => [2b, 28, 28]
      x_concat1 = tf.reshape(tf.concat([x, x_hat], axis=0),[-1, 28, 28])  ---------此处必须重新reshape，才能得到[2b, 28, 28]
```   

### （2）关于VAE和GAN的区别
  * 1.VAE和GAN都是目前来看效果比较好的生成模型，本质区别我觉得这是两种不同的角度，VAE希望通过一种显式(explicit)的方法找到一个概率密度，并通过最小化对数似函数的下限来得到最优解；
GAN则是对抗的方式来寻找一种平衡，不需要认为给定一个显式的概率密度函数。（李飞飞）
  * 2.简单来说，GAN和VAE都属于深度生成模型（deep generative models，DGM）而且属于implicit DGM。他们都能够从具有简单分布的随机噪声中生成具有复杂分布的数据（逼近真实数据分布），而两者的本质区别是从不同的视角来看待数据生成的过程，从而构建了不同的loss function作为衡量生成数据好坏的metric度量。
  * 3.要求得一个生成模型使其生成数据的分布 能够最小化与真实数据分布之间的某种分布差异度量，例如KL散度、JS散度、Wasserstein距离等。采用不同的差异度量会导出不同的loss function，比如KL散度会导出极大似然估计，JS散度会产生最原始GAN里的判别器，Wasserstein距离通过dual form会引入critic。而不同的深度生成模型，具体到GAN、VAE还是flow model，最本质的区别就是从不同的视角来看待数据生成的过程，从而采用不同的数据分布模型来表达。 [https://www.zhihu.com/question/317623081](https://www.zhihu.com/question/317623081)
  * 4.描述的是分布之间的距离而不是样本的距离。[https://blog.csdn.net/Mark_2018/article/details/105400648](https://blog.csdn.net/Mark_2018/article/details/105400648)
<br/>


## References:
* 深度学习与TensorFlow 2入门实战（完整版）---龙曲良
* [https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) [https://medium.com/@joseph.rocca](Joseph Rocca)

