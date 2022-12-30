# GAN网络训练并生成动漫头像



## 一、GAN模型的训练与测试：

### 训练：

```shell
bash GAN_train.sh $dataset_path
```

$dataset_path为训练集路径

### 测试：

```shell
bash GAN_test.sh $result_path
```

$dataset_path为生成图片输出路径



## 二、WGAN-GP模型的训练与测试：

### 训练：

```shell
bash WGANGP_train.sh $dataset_path
```

$dataset_path为训练集路径

### 测试：

```shell
bash WGANGP_test.sh $result_path
```

$dataset_path为生成图片输出路径