# 毕业设计
graduate python files

## 代码
```
使用了SDCN、NOCD、openne库等的开源代码。
https://github.com/thunlp/OpenNE
https://github.com/shchur/overlapping-community-detection
https://github.com/bdy9527/SDCN
```

论文请添加openne库的cites及以下cites:
```
@inproceedings{sdcn2020,
  author    = {Deyu Bo and
               Xiao Wang and
               Chuan Shi and
               Meiqi Zhu and
               Emiao Lu and
               Peng Cui},
  title     = {Structural Deep Clustering Network},
  booktitle = {{WWW}},
  pages     = {1400--1410},
  publisher = {{ACM} / {IW3C2}},
  year      = {2020}
}

@article{
    shchur2019overlapping,
    title={Overlapping Community Detection with Graph Neural Networks},
    author={Oleksandr Shchur and Stephan G\"{u}nnemann},
    journal={Deep Learning on Graphs Workshop, KDD},
    year={2019},
}
```



## 流程
```
用途：宏基因组微生物数据的OTU模块划分
数据来源 : ncbi, fastq data. ENA；BioProject No：PRJEB21169
预处理：将72个fastq转换成fasta工具，忽略其质量信息。
使用linux文本命令cat合并72个fna文件。
使用qiime工具的pick_open_references.py将整个fna中的序列分类到物种，即个OTU。
根据每个OTU所含序列的标签信息（序列名包含run下标）得到每个OTU在72个run中的序列数占每个run的百分比。
丢弃含量少于一定数量的OTU，得到25023行OTU、72列丰度值数据，命名为bio72.txt。
根据pearson相关系数获取一阶网络，一个节点是一个OTU，两个OTU间相关系数大于一定阈值构成一条边。
用openne的DeepWalk，Node2Vec，Line，Lle获取一阶图128维的嵌入表示，获取节点的网络表示数据。
记为bio72.adjlist文件。
根据128维的嵌入数据继续根据pearson相关系数获取二阶网络。记为dpwk.adjlist等。
bio72.txt是GCN的输入向量X，一阶网络、二阶网络是GCN的图。
GCN直接学习并输出community detection的community标签数值。
NOTE：一切过程注意networkx读文件的有序性。
```

## 使用
```
OTU获取预处理代码：./components/my_preprcs.py
后续OTU预处理代码：./preprocess.ipynb
GCN学习和社区发现：./myNocd.ipynb
```


## 信息
```       
language: python, pytorch, gpu used  
Platform: linux Server, Memory 256G, GPU Nvidia Tesla P40  
Platform: Windows Client: Chrome jupyter lab   
author:   zhangzq  
create date:	2021/04/21  
```

## 实验结果

### 实验笔记
```
dpwk/n2v/lle/line.adjlist constructing time : 8h    
towards disconnectivity : robust  
It is a remarkable fact that NOCD is good at disjoint graphs, rather Spectral Cluster, LabelPropogation, SCD, EdMot, GEMSEC don't work at the same time. In other words, NOCD is robust . 
```

### 时间消耗
| Machine Learning tmie | model | GCN time | hidden layer | 
|:----:|:----:|:----:|:----:|
| 5 min  | LabelProp | TODO min | [] |
| 45 min | SCD       | 16 min | [512] |
| 77 min | EdMot     | TODO min | [512,1024] |
| 4h     | GEMSEC    | 
| 3 min  | Spectral  | 

### 分数记录 Total Modularity Records 
towards 2order pearson graphs, size 0 hidden layer [] : bad  

| name | thresh | Spec_Clt | LabelProp | SCD | EdMot | GEMSEC | nocd | 
|:----:|:----:| :----: |:----:| :----: | :----:| :----:| :----:|
| bio72 | ----| 0.4100     |  0.5332   | 0.3128 | 0.0182 | 0.0531 | 0.5397 |
| dpwk  | 0.5 | 0.6568     |  0.7481   | TODO   | 0.0035 | TODO   | 0.7003   |
| n2v   | 0.8 | 0.2766(TODO) | TODO | TODO | TODO | TODO | TODO | 
| lle   | 0.5 | 0.5459(TODO) |  TODO | TODO | TODO | TODO | TODO |
| line  | 0.7 | 0.4481(TODO) |   TODO | TODO | TODO | TODO | TODO |


### 分数记录 myNOCD  Hyperparameters & results

#### nocd -> bio72 features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 3000| <1e-4> | 0.4051 | good | 0.4903 | btfl | [512] |
| 8 | <1e-1> | 1000| 1e-3 | 0.4114 | good | 0.5371 | btfl | [512] |
| 8 | 1e-2 | 1000| 1e-3 | 0.4090 | good | 0.5358 | btfl | [512] |
| 8 | 1e-2 | 1000| 1e-3 | 0.3950 | good | 0.5228 | btfl | [512,1024] |
| 8 | 1e-2 | 1000| 1e-3 | 0.4125 | wdfl | 0.5252 | btfl | [512,512,1024] |
| 8 | 1e-2 | 1000| 1e-3 | 0.4000 | wdfl | 0.5397 | btfl | [512,512,1024,128] |
| 8 | 1e-2 | 1000| 1e-3 | 0.4097 | wdfl | 0.5334 | btfl | [512,1024,2048,512,128]  |

#### nocd -> dpwk features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | modul | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | 0.7652 | [512] | pearson | 
| 8 | 1e-2 | 1000| 1e-3 | 0.2137 | 0.7645 | [512,1024] | pearson | 
| 8 | 1e-2 | 1000| 1e-3 | 0.2879 | 0.7003 | [512] | merged | 

#### nocd -> n2v features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss |  modul | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| 8 | 1e-2 | 1000| 1e-3 | 0.1638 | 0.7318 | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.1648 | 0.7743 | [512,1024] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.2479 | 0.7189 | [512] | merged | 

#### nocd -> lle features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | modul | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | 
| 8 | 1e-2 | 1000| 1e-3 | 0.2428 | 0.7626 | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.2516 | 0.7624 | [512,1024] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.2605 | 0.7270 | [512] | merged | 

#### nocd -> line features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | modul | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.1553 | 0.8207 | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.1635 | 0.8107 | [512,1024] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.2260  | 0.7955 | [512] | merged | 
