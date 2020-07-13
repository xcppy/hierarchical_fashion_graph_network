'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import math
import heapq
import numpy as np
from tqdm import tqdm

_cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = [1]

_data_generator = None
_N_TEST = None

def ranklist_by_sorted(rating, Ks):
    max_rat = np.max(rating)
    if rating[0] < max_rat:
        r = [0]
    else:
        r = [1]
    return r


def get_performance(r, Ks):
    Ks = [1]
    auc = []

    for K in Ks:
        auc.append(metrics.hit_at_k(r, K))

    return {'auc': np.array(auc)}

def test_one_user(x):
    # user u's ratings for user u
    rating = x

    r = ranklist_by_sorted(rating, Ks)

    return get_performance(r, Ks)

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


def test(sess, model, data_generator, args, drop_flag=True, batch_test_flag=False):
    global _data_generator
    global _N_TEST
    global _batch_size
    global _max_ol

    _data_generator = data_generator
    _test_indx = data_generator.test_indx
    _test_len = data_generator.test_len
    _test_adj = data_generator.test_adj
    _batch_size = 1024
    _max_ol = data_generator.max_ol

    _N_TEST = _data_generator.n_fltb_tests
    n_test = _N_TEST/4

    result = {'auc': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(_cores)

    count = 0

    num_batch = math.ceil(_N_TEST/_batch_size)

    for idx in range(num_batch):
        start = idx * _batch_size
        end = min((idx + 1) * _batch_size, _N_TEST)

        fltb_batch = np.array(_test_indx[start:end])
        flen_batch = np.squeeze(np.array(_test_len[start:end]))
        fadj_batch = np.array(_test_adj[start:end])

        rate_batch = sess.run(model.fltb_neg_scores, {model.fltb_input:fltb_batch,
                                                    model.flen_input:flen_batch,
                                                      model.fadj_input:fadj_batch,
                                                    model.node_dropout: [0.],
                                                    model.mess_dropout: [0.]})

        a = np.reshape(np.squeeze(rate_batch), [-1, 4])
        batch_result = pool.map(test_one_user, a)

        count += len(batch_result)

        for re in batch_result:

            result['auc'] += re['auc']/n_test

    assert count == n_test
    pool.close()
    return result




