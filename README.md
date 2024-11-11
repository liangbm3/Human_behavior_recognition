# 基于无人机的人体行为识别

## 1. 项目介绍
这个仓库是2024年第六届全球校园人工智能算法精英大赛——算法挑战赛的全国总决赛的基于无人机的人体行为识别赛题的实现仓库。

参赛队伍：我不是吴恩达  
队员：闫雨昊、梁倍铭

具体介绍参见[算法说明书](./algorithm_description.md)

## 2. 环境配置
1. 创建conda环境，在项目根目录下运行命令
    ```bash
    conda env create -f environment.yaml #注意CUDA版本为12.1
    ```
2. 在`./Model_inference/Mix_GCN/model/Temporal_shift`目录下，运行如下命令来安装cuda扩展

    ```bash
    chmod +x run.sh
    ./run.sh 
    ```

## 3. 数据预处理
先从官网或如下百度网盘链接获取数据集：  
通过网盘分享的文件：data.zip  
链接: https://pan.baidu.com/s/1JbiRriqTbhFTRFc2-39w_Q?pwd=3ubf 提取码: 3ubf  

将数据集解压到`./Process_data/`路径下，data目录如下：
+ test_joint.npy
+ train_joint.npy
+ train_label.npy
+ val_joint.npy
+ val_label.npy

`train`为训练集，`val`为验证集，`test`为测试集。训练集用来训练模型，验证集用来检验模型训练效果，测试集用来预测未知的骨骼点序列，并生成置信度文件用来竞赛打分。为了方便生成置信度文件，我们需要生成一个和测试集等长的全零label，在`./Process_data`目录下运行如下命令即可：

```bash
python gen_label.py
```

然后进行以下数据预处理，来生成骨骼点的其他模态数据，在`./Process_data/`路径，按顺序运行以下命令

1. 处理出bone模态数据
    ```bash
    python gen_modal.py --modal bone --use_mp True
    ```
2. 处理出motion模态数据
    ```bash
    python gen_modal.py --modal motion
    ```
3. 处理得到合并模态数据
    ```bash
    python gen_modal.py --modal jmb --use_mp True
    ```

接下来，需要提取二维数据，以输入到2D骨架识别网络进行训练和测试，我们提取C通道的X和Z两个通道，同时生成训练以及在测试集评估时的`.npz`文件，在`./Process_data/`路径下，运行命令
```bash
python extract_2dpose.py
```

我们使用`joint`、`bone`、`joint_motion`和`bone_motion`进行训练和测试，在`./Process_data/`路径下，运行命令
```bash
python estimate_3dpose.py --config ./configs/3d/joint3d.yaml
python estimate_3dpose.py --config ./configs/3d/bone3d.yaml
python estimate_3dpose.py --config ./configs/3d/jointmotion3d.yaml
python estimate_3dpose.py --config ./configs/3d/bonemotion3d.yaml
```

## 3.训练

进入目录`./Model_inference/Mix_GCN/`，在该目录下运行命令

```bash
python main.py --config ./config/aagcn/aagcn_V2_B_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/aagcn/aagcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/aagcn/aagcn_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/aagcn/aagcn_V2_J_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/aagcn/aagcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/aagcn/aagcn_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_B_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_BM_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_J_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_JM_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/ctrgcn/ctrgcn_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_B_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_J_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/msg3d/msg3d_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_B_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_J_2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mstgcn/mstgcn_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn/tdgcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn/tdgcn_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn/tdgcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tdgcn/tdgcn_V2_JM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tegcn/tegcn_V2_B_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tegcn/tegcn_V2_BM_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tegcn/tegcn_V2_J_3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/tegcn/tegcn_V2_JM_3d.yaml --phase train --save-score True --device 0
```

进入目录`./Model_inference/Mix_Former/`，在该目录下运行命令
```bash
python main.py --config ./config/mixformer/mixformer_V2_B2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_B3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_BM2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_BM3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_J2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_J3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_JM2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer/mixformer_V2_JM3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_B2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_B3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_BM2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_BM3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_J2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_J3d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_JM2d.yaml --phase train --save-score True --device 0
python main.py --config ./config/mixformer_K2/mixformer_V2_k2_JM3d.yaml --phase train --save-score True --device 0
```

## 4.测试

进入目录`./Model_inference/Mix_GCN/`，使用如下类似的命令
```bash
python main.py --config ./config/aagcn/aagcn_V2_B_3d.yaml --phase test --save-score True --weights ./output/aagcn_B_3d/runs-38-9918.pt --device 0 --result-path ../../result/aagcn/aagcn_b3d.npy

```

进入目录`./Model_inference/Mix_Former/`，使用如下类似的命令
```bash
python main.py --config ./config/mixformer/mixformer_V2_B2d.yaml --phase test --save-score True --weights ./output/skmixf_V2_B_2d/runs-53-13833.pt --device 0 --result-path ../../result/skeformer/skeformer_b2d.npy

```

我们的对测试集的推理结果备份在`./output`

在`./`目录下运行如下命令
```bash
python parameter_search.p
python ensemble.py
```
最后在根目录下生成的`pred.npy`就是目标文件

