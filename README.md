# 心得：**此代码可以很好地体现残差网络的构建**



## The complexity and accuracy of the neural network model
![img1](https://github.com/xiaoxiaokaiyan/New_Tensorflow_Resnet18_cifar100/blob/main/complexity%20and%20accuracy.png)



## Dependencies:
* Windows10
* python==3.6.12
* > GeForce GTX 1660TI
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


## Visualization Results
* resnet18 训练结果 1
![img1](https://github.com/xiaoxiaokaiyan/New_Tensorflow_Resnet18_cifar100/blob/main/result_1.PNG)
* resnet101 训练结果 2
![img1](https://github.com/xiaoxiaokaiyan/New_Tensorflow_Resnet18_cifar100/blob/main/result_2.PNG)



## Public Datasets:
* cifar100



## Experience：
  ### 关于VAE和GAN的区别
  * VAE和GAN都是目前来看效果比较好的生成模型，本质区别我觉得这是两种不同的角度，VAE希望通过一种显式(explicit)的方法找到一个概率密度，并通过最小化对数似函数的下限来得到最优解；
GAN则是对抗的方式来寻找一种平衡，不需要认为给定一个显式的概率密度函数。（李飞飞）
  * 简单来说，GAN和VAE都属于深度生成模型（deep generative models，DGM）而且属于implicit DGM。他们都能够从具有简单分布的随机噪声中生成具有复杂分布的数据（逼近真实数据分布），而两者的本质区别是从不同的视角来看待数据生成的过程，从而构建了不同的loss function作为衡量生成数据好坏的metric度量。
  * 要求得一个生成模型使其生成数据的分布 能够最小化与真实数据分布之间的某种分布差异度量，例如KL散度、JS散度、Wasserstein距离等。采用不同的差异度量会导出不同的loss function，比如KL散度会导出极大似然估计，JS散度会产生最原始GAN里的判别器，Wasserstein距离通过dual form会引入critic。而不同的深度生成模型，具体到GAN、VAE还是flow model，最本质的区别就是从不同的视角来看待数据生成的过程，从而采用不同的数据分布模型来表达。
  [https://www.zhihu.com/question/317623081](https://www.zhihu.com/question/317623081)
  * 描述的是分布之间的距离而不是样本的距离。[https://blog.csdn.net/Mark_2018/article/details/105400648](https://blog.csdn.net/Mark_2018/article/details/105400648)
          
```
      os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'**-----------------------------------（本代码按此方法解决）
       
      or
      
      physical_devices = tf.config.experimental.list_physical_devices('GPU')
      if len(physical_devices) > 0:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
```   
  * 原因二：You have incompatible versions of CUDA, TensorFlow, NVIDIA drivers, etc.
      * you can see [https://blog.csdn.net/qq_41683065/article/details/108702408](https://blog.csdn.net/qq_41683065/article/details/108702408)
        **my cudnn==7.6.4 cuda10.0_0  cudatoolkit==10.0.130**
        
   ### tensorflow-gpu版本代码出现numpy错误
  * 其中一种解决方法：**pip install --upgrade numpy**
  
  

## References:
* 深度学习与TensorFlow 2入门实战（完整版）---龙曲良

