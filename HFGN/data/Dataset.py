'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utility import helper
import math
from sklearn import preprocessing
from tqdm import tqdm


class Data(object):
    def __init__(self, path):
        self.path = path

        outfit_file = path + '/new_fltb/fltb_train_outfit_item.txt'
        fltb_test_file = path + '/new_fltb/fltb_test.txt'
        cate_file = path + '/new_fltb/fltb_item_cate2.txt'
        visual_file = path + '/new_fltb/fltb_feat_resnet152_02.npy'

        recom_train_file = path + '/train_uo.txt'
        recom_test_file = path + '/test_uo.txt'

        self.n_train_outfits, self.n_recom_trains, self.n_users, self.exist_users_list, \
        train_outfit_set, self.pos_list, self.train_u_outfits_dict \
            = self._read_recom_train_file(recom_train_file)

        self.fltb_outfit_list = list(train_outfit_set)

        self.n_fltb_trains = self.n_train_outfits

        self.n_recom_tests, self.test_u_outfits_dict = self._read_recom_test_file(recom_test_file)

        self.n_train_items, self.outfit_items_dict, self.max_ol = self._read_outfit_file(outfit_file)

        self.n_all_items, self.n_cates, self.cate_item_dict, self.item_cate_dict = self._read_category_file(cate_file)

        self.cate_adj = self.get_cate_adj_mat(self.outfit_items_dict, self.item_cate_dict)

        self.outfit_map, self.outfit_adj, self.outfit_len = self.construct_outfit_map(self.outfit_items_dict,
                                                                                      self.cate_adj,
                                                                                      self.item_cate_dict)
        self.test_len, self.test_position, self.test_indx, self.test_adj, \
        self.n_fltb_tests, self.neg_num = self._read_fltb_test_file(fltb_test_file)
        self.n_cands = self.neg_num + 1

        self.R_uo = self.get_R_mat(self.train_u_outfits_dict)

        self.uo_adj, self.norm_uo_adj = self.get_uo_adj_mat(self.R_uo)

        self.visual_feat = preprocessing.normalize(np.load(visual_file), norm='l2', axis=1)
        self.vf_dim = np.shape(self.visual_feat)[1]

        self.print_statistics()


    def get_R_mat(self, uo_dict):
        R_uo = sp.dok_matrix((self.n_users, self.n_train_outfits), dtype=np.float32)
        for u in uo_dict:
            for o in uo_dict[u]:
                R_uo[u, o] = 1.0

        print('create R_uo')

        return R_uo

    def get_cate_adj_mat(self, outfit_items_dict, item_cate_dict):
        try:
            t1 = time()
            cate_adj_mat = sp.load_npz(self.path + '/hiergraph/s_category_adj_mat.npz')
            print('already load cate_adj matrix', cate_adj_mat.shape, time() - t1)
        except Exception:
            t1 = time()
            cate_adj_mat = self.create_cate_adj_mat(item_cate_dict, outfit_items_dict)
            sp.save_npz(self.path + '/hiergraph/s_category_adj_mat.npz', cate_adj_mat)
            print('already create cate_adj matrix', cate_adj_mat.shape, time() - t1)

        return cate_adj_mat

    def create_edge_weights(self, R, n_nodes):

        R = R.tocoo()
        row_sum = np.squeeze(R.sum(axis=0).getA())

        adj = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
        adj = adj.tolil()

        for i, j, v in zip(R.row, R.col, R.data):
            adj[i, j] = v / row_sum[j]

        adj = adj.tocoo()

        row_sum2 = np.squeeze(adj.sum(axis=1).getA())
        norm_adj = sp.dok_matrix((n_nodes, n_nodes), dtype=np.float32)
        norm_adj = norm_adj.tolil()

        for i, j, v in zip(adj.row, adj.col, adj.data):
            norm_adj[i, j] = v / row_sum2[i]
        norm_adj = norm_adj.todok()

        return norm_adj.tocoo().tocsr()

    def create_cate_adj_mat(self, item_cate_dict, outfit_items_dict):
        cate_R = sp.dok_matrix((self.n_cates, self.n_cates), dtype=np.float32)
        cate_R = cate_R.tolil()
        for outfit in outfit_items_dict:
            items = outfit_items_dict[outfit]
            for i in items:
                c1 = item_cate_dict[i]
                for j in items:
                    if i != j:
                        c2 = item_cate_dict[j]
                        cate_R[c1, c2] += 1.0

        return self.create_edge_weights(cate_R, self.n_cates)

    def get_uo_adj_mat(self,R_uo):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/hiergraph/s_uo_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/hiergraph/s_uo_norm_adj_mat.npz')
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat = self.create_adj_mat(R_uo)
            sp.save_npz(self.path + '/hiergraph/s_uo_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/hiergraph/s_uo_norm_adj_mat.npz', norm_adj_mat)
        return adj_mat, norm_adj_mat


    def create_adj_mat(self, R):
        t1 = time()

        R = R.tolil()

        adj_mat = R
        adj_mat = adj_mat.todok()

        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat)  # + sp.eye(adj_mat.shape[0])

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr()

    def create_adj(self, o_items, cate_adj, item_cate):
        cate_adj.tolil()
        o_cates = []
        o_graph = np.zeros([self.max_ol, self.max_ol], dtype=np.float32)

        for i in o_items:
            c = item_cate[i]
            o_cates.append(c)
        for i in range(len(o_cates)):
            for j in range(len(o_cates)):
                c1 = o_cates[i]
                c2 = o_cates[j]
                o_graph[i, j] = cate_adj[c1, c2]

        return o_graph

    def construct_outfit_map(self,outfit_dict, cate_adj, item_cate):


        outfit_map = []
        outfit_len = []
        outfit_adj = []
        for o in range(self.n_train_outfits):
            items = outfit_dict[o]
            o_len = len(items)
            adj = self.create_adj(items, cate_adj, item_cate)
            outfit_adj.append(adj)
            outfit = items + [-1]*(self.max_ol-o_len)
            outfit = np.array(outfit, dtype=np.int)
            outfit_map.append(outfit)
            outfit_len.append(o_len)

        outfit_map = np.array(outfit_map, dtype=np.int)
        outfit_adj = np.array(outfit_adj)

        return outfit_map, outfit_adj, outfit_len

    def _read_fltb_test_file(self, test_file):
        test_len = []
        test_position = []
        test_indx = []
        test_adj = []
        neg_num = 0
        n_tests = 0
        with open(test_file, 'r')as f:
            lines = f.readlines()
            count = 0
            for l in lines:
                meta = l.strip('\n').split(';')
                oid, o_len, position, outfits = meta[0], meta[1], meta[2], meta[3:]

                neg_num = len(outfits) - 1
                for outfit in outfits:
                    test_len.append(int(o_len))
                    test_position.append(int(position))
                    o_items = [int(i) for i in outfit.split(',')]
                    test_map = np.array(o_items)
                    test_indx.append(test_map)

                    adj = self.create_adj(o_items[:int(o_len)], self.cate_adj, self.item_cate_dict)
                    test_adj.append(adj)
                    count += 1
        n_tests = len(test_len)

        return test_len, test_position, test_indx, test_adj, n_tests, neg_num

    def _read_category_file(self, cate_file):

        cate_item = dict()
        item_cate = dict()
        n_items = 0
        n_cates = 0

        with open(cate_file) as f_cate:
            for l in f_cate.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    item = int(l.split(' ')[0])
                    cate = int(l.split(' ')[1])
                except Exception:
                    continue

                cate_item.setdefault(cate, [])
                cate_item[cate].append(item)
                item_cate[item]=cate
                n_items = max(n_items, item)
                n_cates = max(n_cates, cate)

        n_items += 1
        n_cates += 1
        return n_items, n_cates, cate_item, item_cate

    def _read_outfit_file(self, outfit_file):
        """
        get the data from the outfit file
        :param outfit_file:
        :return:
        { outfit_item_dict: the items composed of the outfit.
          max_ol: the maximum length of the outfit.
        """
        outfit_items_dict= dict()
        train_items = set()
        max_ol = 0
        with open(outfit_file)as f:
            for l in f.readlines():
                if len(l) == 0: break
                l = l.strip('\n')
                try:
                    items = [int(i) for i in l.split(' ')]
                    oid, o_items = items[0], items[1:]
                except Exception:
                    continue
                outfit_items_dict[oid] = o_items
                train_items = train_items.union(set(o_items))
                max_ol = max(max_ol, len(o_items))

        return len(train_items),outfit_items_dict, max_ol

    def _read_recom_test_file(self, test_file):
        n_tests = 0
        test_u_outfits = {}
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    try:
                        outfits = [int(i) for i in l.split(' ')]
                    except Exception:
                        continue

                    uid, test_outfits = outfits[0], outfits[1:]
                    test_u_outfits[uid] = test_outfits
                    n_tests += len(test_outfits)

        return n_tests, test_u_outfits


    def _read_recom_train_file(self, train_file):
        n_train_outfits, n_trains, n_users = 0, 0, 0
        exist_users = []
        train_outfit_set = set()
        pos = []
        train_u_outfits = {}
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n')
                    outfits = [int(i) for i in l.split(' ')]
                    uid, train_outfits = outfits[0], outfits[1:]

                    train_u_outfits[uid] = train_outfits
                    exist_users.append(uid)
                    n_trains += len(train_outfits)

                    for i in train_outfits:
                        # self.R[uid, i] = 1.
                        pos.append([uid, i])
                        train_outfit_set.add(i)  #
        n_users = len(exist_users)
        n_train_outfits = len(train_outfit_set)

        return n_train_outfits, n_trains, n_users, exist_users, train_outfit_set, pos, train_u_outfits

    def print_statistics(self):
        print('recommendation data...')
        print('n_users=%d, n_cates=%d' % (self.n_users, self.n_cates))
        print('n_interactions=%d' % (self.n_recom_trains + self.n_recom_tests))
        print('n_train=%d, n_test=%d, sparsity=%.5f' %
              (self.n_recom_trains, self.n_recom_tests,
               (self.n_recom_trains + self.n_recom_tests) / (self.n_users * self.n_train_outfits)))
        print('fltb data...')
        print('n_all_items=%d, n_test_items=%d, n_train_items=%d, vf_dim=%d' % (
        self.n_all_items, (self.n_all_items - self.n_train_items), self.n_train_items, self.vf_dim))
        print('n_trains=%d, n_tests=%d'%(self.n_fltb_trains, self.n_fltb_tests))





