# 毕业设计
graduate python files

## 代码
```
本文的主要理论基础来源于 基于网络嵌入方法的肠道微生物组大数据网络分析 [J]
使用了SDCN、NOCD、openne库等的开源代码。
https://github.com/thunlp/OpenNE
https://github.com/shchur/overlapping-community-detection
https://github.com/bdy9527/SDCN
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

# TODO
### Total Scores 
```     
method     LabelProp    SCD       EdMot     GEMSEC    Spectral    nocd   
time       5 min        45 min    77 min    4h        3 min               
bio72      0.5332       0.3128    0.0182    0.0531    0.4100      0.5397                
dpwk(0.5)  0.7481       0.5534    0.0035    0.4448    0.6568      0.7064  
n2v(0.8)   0.7435       0.5713    ------    0.4275	  0.6546			  
lle(0.5)   0.7943       0.6969    ------    0.3922	  0.3614						 
line(0.7)  0.8429       0.6928    ------    0.5468    0.7289  			  
```




### Cites
论文请添加openne库的cites及以下cites:
```
李倩莹, 蔡云鹏, 张凯. 基于网络嵌入方法的肠道微生物组大数据网络分析 [J]. 集成技术, 2019, 8(5): 34-48.
Li QY, Cai YP, Zhang K. Inferring gut microbial interaction network from microbiome data using network embedding
algorithm [J]. Journal of Integration Technology, 2019, 8(5): 34-48.

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