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

constructing dpwk/n2v/lle/line.adjlist time for each is 8h.  

| name | thresh | spec_clt | !sdcn! | nocd | LabelProp | SCD | EdMot | GEMSEC |
|:----:|:----:| :----: | :----: |:----:| :----: | :----:| :----:| :----:|
| bio72 | ----| 0.4205 | ------ | 0.5397(4min) | 0.5332(5min) | 0.3128(45min) | 0.0182(77min) | 0.0531(4h) |
| dpwk  | 0.5 | 0.7206 | 0.3582 |  
| n2v   | 0.8 | 0.2766 | 0.3308 |  
| lle   | 0.5 | 0.5459 | 0.3760 |  
| line  | 0.7 | 0.4481 | 0.3186 |  


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
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.4051 | good | 0.4903 | btfl | [512] | TODO 



#### nocd -> n2v features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.4051 | good | 0.4903 | btfl | [512] | TODO 



#### nocd -> lle features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.4051 | good | 0.4903 | btfl | [512] | TODO 



#### nocd -> line features & graphs
| n_cluss | l2_reg | max_epochs | lr | bnl_loss | Z_hist  | modul | cd_plot | hidden_sizes|  
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| 8 | 1e-2 | 1000| 1e-3 | 0.4051 | good | 0.4903 | btfl | [512] | TODO