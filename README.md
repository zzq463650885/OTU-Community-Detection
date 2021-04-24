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

| data & graph | graph_filter | spectral_cluster | !!!sdcn | nocd | 
| :----: | :----: | :----: | :----: | :----: |
| bio72 | ---- | 0.4205 | ------ | TODO | 
| dpwk  | 0.5  | 0.7206 | 0.3582 | TODO | 
| n2v   | TODO | 0.2766 | 0.3308 | TODO | 
| lle   | 0.5  | 0.5459 | 0.3760 | TODO | 
| line  | TODO | 0.4481 | 0.3186 | TODO | 


### myNOCD  Hyperparameters & results

#### nocd -> bio72 features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 3000| <1e-4> | 0.4051 | good | 0.4903 | btfl | [512] fake |
