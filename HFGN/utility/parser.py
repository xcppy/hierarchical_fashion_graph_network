'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run HFGN.")
    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='pog',
                        help='Choose a dataset')
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', type=float, default=0.00001,
                        help='Regularizations.')
    parser.add_argument('--r_view', type=int, default=8,
                        help='R view nums.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--fltb_lr', type=float, default=0.01,
                        help='Learning rate for FLTB.')

    parser.add_argument('--recom_lr', type=float, default=0.0001,
                        help='Learning rate for recommendation.')

    parser.add_argument('--model_type', nargs='?', default='HFGN',
                        help='Specify the name of model (HFGN).')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=1,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10, 20, 30, 40, 50]',
                        help='Top-K evaluation.')
    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')
    parser.add_argument('--train_mode', type=int, default=0,
                        help='0: optimize one loss; 1: optimize two loss')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='parameter for fltb loss.')
    parser.add_argument('--pretrain_path',nargs='?', default='',
                        help='pretrain data load path.')

    return parser.parse_args()
