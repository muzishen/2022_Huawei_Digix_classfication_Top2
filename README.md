# 2022_Huawei_Digix_classfication_Top2
 2022全球Ai算法精英赛-车道线瑕疵分类第二名 

### 代码说明
#### 配置文件在configs 目录下，使用了5个模型：
convnext base  
convnext small  
convnext tiny  
ibna101  
swin tiny  
swin small  


#### 环境
 依赖环境：requirements.txt  
`
pip install -r requirements.txt
`


#### 有用的trick
训练尺度用的1280 768  
学习率用的1e-4  
用了warm up  
数据增强 resize flip 放射变换  
使用了5折交叉验证  
推理采用TTA 增强  
损失函数采用BCEWithLogitsLoss  

#### 训练方法
`
python3 main.py  --config_file './configs/convnext_base.yaml'      
`  
注意：依次替换  convnext_base.yaml - >  convnext_small.yaml;   convnext_tiny.yaml ; resnet_ibna_101.yaml ;swin_tiny.yaml; swin_small.yaml  

#### 测试方法
`
python3 test.py 依次得到5个csv文件  
`  
`
python3 fusion.py 进行融合
`  

#### 致谢
:smile: 感谢蒋鑫师弟的合作  
Click on the star  :star:, Thank you :heart:


