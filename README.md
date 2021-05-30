# OTUCD

## codes
```
we use open source code of SDCN、NOCD、openne...
https://github.com/thunlp/OpenNE
https://github.com/shchur/overlapping-community-detection
https://github.com/bdy9527/SDCN
```

## use
```
get OTUs：./components/my_preprcs.py
preprocess：./preprocess.ipynb
OTUCD：./OTUCD.ipynb
```

## information
```       
language: python, pytorch, gpu used  
Platform: linux Server, Memory 256G, GPU Nvidia Tesla P40  
Platform: Windows Client: Chrome jupyter lab   
author:   zhangzq  
create date:	2021/04/21  
```

## records
```
dpwk/n2v/lle/line.adjlist constructing time : 8h    
towards disconnectivity : robust  
It is a remarkable fact that NOCD is good at disjoint graphs, rather Spectral Cluster, LabelPropogation, SCD, EdMot, GEMSEC don't work at the same time. In other words, NOCD is robust . 
```

## Cites
please cite papers as follows:
```
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
