## 2021/04/27 主文件夹README 备份 

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
