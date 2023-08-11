## 项目说明
本项目借助https://github.com/Accenture/Labs-Federated-Learning/blob/clustered_sampling 的代码进行模型训练，并在训练好的iid和non-iid的mnist\cifar模型基础上编写测试的逻辑并执行，从而将测试结果其作为本工作的输入。

## 模型训练

模型训练的代码相对于原代码的唯一更改是尝试使用cuda进行加速（原代码只会使用cpu进行训练）

通过如下的四个命令跑mnist和cifar10的iid与non-iid的模型
```
python FL.py MNIST_iid random any 0 50 0.01 1.0 0.1 False 
python FL.py MNIST_shard random any 0 50 0.01 1.0 0.1 False 
python FL.py CIFAR10_iid random any 0 100 0.05 1.0 0.1 False
python FL.py CIFAR10_bbal_0.001 random any 0 100 0.05 1.0 0.1 False
```
其中，根据原代码的readme，只有第一个参数的设置与数据集和独立同分布性相关，因此我们只对第一个参数进行更改，除了第一个参数外的所有参数都使用源代码的默认参数。
第一个参数的设置如下：
- `MNIST_iid`：使用mnist的iid的数据集训练模型
- `MNIST_shard`：使用mnist的non-iid（通过shard的方式得到）的数据集训练模型
- `CIFAR10_iid`：使用cifar10的iid的数据集训练模型
- `CIFAR10_bbal_0.001`：使用cifar10的non-iid（通过dirichlet方法，α的值为0.001）的数据集训练模型

模型训练的具体细节可以查看https://github.com/Accenture/Labs-Federated-Learning/blob/clustered_sampling

## 模型测试

然后进入sampling_test目录，分别运行如下4个命令跑跑mnist和cifar10的iid与non-iid的测试结果
```
python test.py mnist True
python test.py mnist False
python test.py cifar True
python test.py cifar False
```

其中，第一个参数可选`mnist`和`cifar`两种，表示使用由哪种数据集训练的模型进行测试；
第二个参数可选`True`和`False`两种，表示使用是否为iid的模型进行测试。

最终测试结果会存储在目录sampling_test/saved_exp_info中