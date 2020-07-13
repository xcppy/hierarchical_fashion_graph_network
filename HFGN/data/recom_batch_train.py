'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import multiprocessing
import numpy as np
import random as rd
import math

_data_generator = None
_n_users = None
_n_outfits = None
_n_trains = None
_batch_size = None
_n_batch = None

_cores = multiprocessing.cpu_count() // 2
_max_ol = None

def sample(data_generator, batch_size):
    global _data_generator
    global _n_users
    global _n_outfits
    global _max_ol
    global _n_batch
    global _batch_size

    _data_generator = data_generator
    _batch_size = batch_size
    _max_ol = _data_generator.max_ol

    _n_users, _n_outfits, = _data_generator.n_users, _data_generator.n_train_outfits
    _n_trains = _data_generator.n_recom_trains

    np.random.shuffle(_data_generator.pos_list)
    _n_batch = math.ceil(len(_data_generator.pos_list) / _batch_size)

    u_list, po_list, plen_list, no_list, nlen_list= [], [], [], [], []
    # num_task = 8  # multiprocessing.cpu_count()
    if _cores == 1:
        for i in range(0, _n_batch):
            u_batch, po_batch, plen_batch, no_batch, nlen_batch= get_train_batch(i)
            u_list.append(u_batch)
            po_list.append(po_batch)
            plen_list.append(plen_batch)
            no_list.append(no_batch)
            nlen_list.append(nlen_batch)


    else:

        pool = multiprocessing.Pool(_cores)
        # t = math.ceil(len(_Dataset.pos)/_batch_size)
        res = pool.map(get_train_batch, range(_n_batch))
        pool.close()
        pool.join()
        u_list = [r[0] for r in res]
        po_list = [r[1] for r in res]
        plen_list = [r[2] for r in res]
        no_list = [r[3] for r in res]
        nlen_list = [r[4] for r in res]


    return (u_list, po_list, plen_list, no_list, nlen_list)

def batch_get(batches, i):
    return [(batches[r])[i] for r in range(5)]


def create_adj(o_items, cate_adj, item_cate):
    cate_adj.tolil()
    o_cates = []
    o_graph = np.zeros([_max_ol, _max_ol],dtype=np.float32)

    for i in o_items:
        c = item_cate[i]
        o_cates.append(c)
    for i in range(len(o_cates)):
        for j in range(len(o_cates)):
            c1 = o_cates[i]
            c2 = o_cates[j]
            o_graph[i, j] = cate_adj[c1, c2]

    return o_graph

def get_train_batch(b):

    begin = b * _batch_size
    end = min(len(_data_generator.pos_list), begin + _batch_size)
    u_batch, po_batch, plen_batch, no_batch, nlen_batch= [],[],[],[],[]

    for p in range(begin, end):
        u, pos_o = _data_generator.pos_list[p]
        neg_o = pos_o
        while neg_o in _data_generator.train_u_outfits_dict[u] or neg_o in _data_generator.test_u_outfits_dict[u]:
            neg_o = rd.randrange(_n_outfits)

        u_batch.append(u)
        po_batch.append(pos_o)
        no_batch.append(neg_o)
        plen_batch.append(_data_generator.outfit_len[pos_o])
        nlen_batch.append(_data_generator.outfit_len[neg_o])

    u_batch = np.array(u_batch)
    po_batch = np.array(po_batch)
    plen_batch = np.array(plen_batch)
    no_batch = np.array(no_batch)
    nlen_batch = np.array(nlen_batch)


    return (u_batch, po_batch, plen_batch, no_batch, nlen_batch)


