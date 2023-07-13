# 2022_Huawei_Digix_classfication_Top2
 2022全球Ai算法精英赛-车道线瑕疵分类第二名 


配置文件在configs 目录下
### 环境
需要的第三方依赖库在  requirements.txt文件

使用了5个模型：
convnext base
convnext small
convnext tiny
ibna101
swin tiny
swin small

训练尺度用的1280 768
学习率用的1e-4
用了warm up
数据增强 resize flip 放射变换
使用了5折交叉验证
推理采用TTA 增强
损失函数采用BCEWithLogitsLoss

训练方法：
python3 main.py  --config_file './configs/convnext_base.yaml'      
注意：依次替换  convnext_base.yaml - >  convnext_small.yaml;   convnext_tiny.yaml ; resnet_ibna_101.yaml ;swin_tiny.yaml; swin_small.yaml

测试方法
python3 test.py 依次得到5个csv文件
然后python3 fusion.py 进行融合
