# Graduate 
graduate python files

## Usage
go into *.ipynb files & click run buttons

## Information
language: 	python  
author:	 	zhangzq  
create date:	2021/04/21  

## Experiment results

### spectral_cluster 、sdcn 、nocd

#### Experiments Notes 

constructing dpwk/n2v/lle/line.adjlist time for each is 8h.  
It is a remarkable fact that NOCD is good at disjoint graphs, rather Spectral Cluster, LabelPropogation, SCD, EdMot, GEMSEC don't work at the same time. In other words, NOCD is robust towards disconnectivity.  
Towards 2order graphs, size 0 hidden layer [] is bad.

| time | hidden layer | 
|:----:|:----:|
| 1 min | [] |
| 5 min | [512] |
| 55 min | [512,1024] |

| name | thresh | spec_clt | !sdcn! | nocd | LabelProp | SCD | EdMot | GEMSEC |
|:----:|:----:| :----: | :----: |:----:| :----: | :----:| :----:| :----:|
| bio72 | ----| 0.4205(TODO) | ------ | 0.5397(4min) | 0.5332(5min) | 0.3128(45min) | 0.0182(77min) | 0.0531(4h) |
| dpwk  | 0.5 | 0.7206(TODO) | 0.3582 |  
| n2v   | 0.8 | 0.2766(TODO) | 0.3308 |  
| lle   | 0.5 | 0.5459(TODO) | 0.3760 |  
| line  | 0.7 | 0.4481(TODO) | 0.3186 |  


### myNOCD  Hyperparameters & results

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
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | good | 0.7652 | btfl | [] | pearson TODO | 
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | good | 0.7652 | btfl | [512] | pearson | 
| 8 | 1e-2 | 1000| 1e-3 | 0.2137 | good | 0.7645 | btfl | [512,1024] | pearson | 



#### nocd -> n2v features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | good | 0.7652 | btfl | [] | pearson TODO| 
| 8 | 1e-2 | 1000| 1e-3 | 0.1638 | good | 0.7318 | just | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.1648 | good | 0.7743 | just | [512,1024] | pearson |







#### nocd -> lle features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | good | 0.7652 | btfl | [] | pearson TODO| 
| 8 | 1e-2 | 1000| 1e-3 | 0.2428 | good | 0.7626 | btfl | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.2516 | good | 0.7624 | btfl | [512,1024] | pearson |







#### nocd -> line features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes| graph_type |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.2051 | good | 0.7652 | btfl | [] | pearson TODO| 
| 8 | 1e-2 | 1000| 1e-3 | 0.1553 | good | 0.8207 | btfl | [512] | pearson |
| 8 | 1e-2 | 1000| 1e-3 | 0.1635 | good | 0.8107 | btfl | [512,1024] | pearson |


