# hierarchical_fashion_graph_network
This is our Tensorflow implementation for the paper:
> Xingchen Li, Xiang Wang, Xiangnan He, Long Chen, Jun Xiao, and Tat-Seng Chua. Hierarchical Fashion Graph Network for Personalized Outfit Recommendation. In SIGIR 2020.

## Introduction
Hierarchical Fashion Graph Network (HFGN) is a new recommendation framework for personalized outfit recommendation task based on hierarchical graph structure.

## Citation
If you want to use our codes and datasets in your research, please cite:
```
@inproceedings{HFGN20,
  author    = {Xingchen Li and
               Xiang Wang and
               Xiangnan He and
               Long Chen and
               Jun Xiao and
               Tat{-}Seng Chua},
  title     = {Hierarchical Fashion Graph Network for Personalized Outfit Recommendation},
  booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} Conference on
               Research and Development in Information Retrieval, {SIGIR} 2020.},
  year      = {2020},
}
```

## Dataset

> Our experiment are based on [POG dataset](https://github.com/wenyuer/POG). We reprocess the data and save the files, and the file format is listed in [Data/pog](https://github.com/xcppy/hierarchical_fashion_graph_network/blob/master/Data/pog/Data.md). 


## Environment
> tensorflow == 1.10.1
> python == 3.6

## Run the Codes
```
python model.py -regs 1e-5 --embed_size 64 --batch_size 1024
```

## Train the model

> For Fill in the Blank (FLTB) task, we only optimize the compatibility loss: L_{com}.

> For Personalized outfit Recommendation task, we use the pretrained FLTB model to intialized the whole model to obtain better performance.


