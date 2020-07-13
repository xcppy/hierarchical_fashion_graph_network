'''
Created on June, 2020
Tensorflow Implementation of HFGN model in:
Xingchen Li et al. In SIGIR 2020.
Hierarchical Fashion Graph Network for Personalized Outfit Recommendation.

@author: Xingchen Li (xingchenl@zju.edu.cn)
'''

import tensorflow as tf
import os
import sys
from time import time
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from data import Dataset
from data import recom_batch_train as recom_batch_generator
from data import fltb_batch_train as fltb_batch_generator
from data import batch_train as batch_generator
from utility.helper import *
import inferrence.fltb_test as fltb_test
import inferrence.recommend_test as recom_test
import utility.parser as parser

args = parser.parse_args()

class HFGN(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'HFGN'

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_outfits = data_config['n_outfits']
        self.n_items = data_config['n_items']
        self.n_cates = data_config['n_cates']
        self.max_ol = data_config['max_ol']
        self.save_weights = []

        self.n_fold = 100

        self.uo_adj = data_config['norm_uo_adj']
        self.cate_items_dict = data_config['cate_items']
        self.vf_dim = data_config['vf_dim']
        self.n_nonzero_elems = self.uo_adj.count_nonzero()

        self.lr_fltb = args.fltb_lr
        self.lr_recom = args.recom_lr
        self.lr = args.lr
        self.r_view = args.r_view

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.model_type += '_emb%d' % (self.emb_dim)
        self.model_type += '_train%d' % (args.train_mode)

        self.regs = args.regs
        self.decay = self.regs

        self.loss_weight = [args.alpha, 1-args.alpha]
        self.cate_dim = self.emb_dim

        self.verbose = args.verbose
        self.node_dropout_flag = args.node_dropout_flag


        """
        Create placeholders for input and dropout.
        """
        self.placeholders = self._init_placeholders()
        self.user_input = self.placeholders['user_input']
        self.po_input = self.placeholders['po_input']
        self.pl_input = self.placeholders['pl_input']
        self.no_input = self.placeholders['no_input']
        self.nl_input = self.placeholders['nl_input']
        self.fltb_input = self.placeholders['fltb_input']
        self.flen_input = self.placeholders['flen_input']
        self.fadj_input = self.placeholders['fadj_input']

        self._init_visual_feat = self.placeholders['_init_visual_feat']
        self._init_outfit_map = self.placeholders['_init_outfit_map']

        self._init_outfit_len = self.placeholders['_init_outfit_len']
        self._init_gather_index = self.placeholders['_init_gather_index']
        self._init_outfit_adj = self.placeholders['_init_outfit_adj']

        self.node_dropout = self.placeholders['node_dropout']
        self.mess_dropout = self.placeholders['mess_dropout']
        self._init_cate_index = self.placeholders['_init_cate_index']

        """
        Initialization of model parameters
        """
        self.weights = self._init_weights()
        self.item_visual_feat = self.weights['visual_feat']
        self.outfit_map = self.weights['outfit_map']
        self.outfit_len = self.weights['outfit_len']
        self.outfit_adj = self.weights['outfit_adj']
        self.gather_index = self.weights['gather_index']
        self.W_cate = self.weights['W_cate']
        self.user_embedding = self.weights['user_embedding']
        self.outfit_embedding = self.weights['outfit_embedding']
        self.cate_items = self.weights['cate_index']

        # Assign weights.
        self.assign_feat = tf.assign(self.item_visual_feat, self._init_visual_feat)
        self.assign_map = tf.assign(self.outfit_map, self._init_outfit_map)
        self.assign_adj = tf.assign(self.outfit_adj, self._init_outfit_adj)
        self.assign_length = tf.assign(self.outfit_len, self._init_outfit_len)
        self.assign_gather = tf.assign(self.gather_index, self._init_gather_index)
        self.assign_cate_index = []
        for c in range(self.n_cates):
            assign = tf.assign(self.cate_items[c], self._init_cate_index[c])
            self.assign_cate_index.append(assign)


        """
        Build compatibility model.
        """

        self.item_cate_feats = self._get_cate_feats(self.item_visual_feat, self.cate_items)

        # get the item feats for all outfits.
        self.all_outfit_feat = tf.nn.embedding_lookup(self.item_cate_feats, self.outfit_map)

        # get item graph feats for all all outfits.
        self.all_outfit_graph_feat = self._get_outfit_graph_feat(self.outfit_adj, self.all_outfit_feat, self.outfit_len)

        self.fltb_neg_feat = tf.nn.embedding_lookup(self.item_cate_feats, self.fltb_input)

        self.fltb_pos_graph_feat = tf.nn.embedding_lookup(self.all_outfit_graph_feat, self.po_input)
        self.fltb_neg_graph_feat = self._get_outfit_graph_feat(self.fadj_input, self.fltb_neg_feat, self.flen_input)

        self.fltb_pos_scores = self._compatibility_score(self.fltb_pos_graph_feat, self.pl_input)
        self.fltb_neg_scores = self._compatibility_score(self.fltb_neg_graph_feat, self.flen_input)

        self.outfit_hier_embedding = self._get_outfit_hier_embed(self.all_outfit_graph_feat, self.outfit_embedding)
        self.user_hier_embedding = self._get_user_hier_node(self.uo_adj, self.outfit_hier_embedding, self.user_embedding)

        self.u_g_embeddings = tf.nn.embedding_lookup(self.user_hier_embedding, self.user_input)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.outfit_hier_embedding, self.po_input)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.outfit_hier_embedding, self.no_input)


        # Inference for the testing phase.
        self.batch_ratings = tf.matmul(self.u_g_embeddings, self.pos_i_g_embeddings, transpose_a=False, transpose_b=True)

        """
        Generate Predictions & Optimize.
        """
        # fltb loss
        self.fltb_loss = self._create_fltb_loss(self.fltb_pos_scores, self.fltb_neg_scores)

        self.mf_loss, self.emb_loss, self.reg_loss = self.create_bpr_loss(self.u_g_embeddings,
                                                                          self.pos_i_g_embeddings,
                                                                          self.neg_i_g_embeddings)

        self.recom_loss = self.mf_loss + self.reg_loss

        self.loss = self.loss_weight[0]*self.fltb_loss + self.loss_weight[1]*self.mf_loss + self.reg_loss

        self.opt  = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.opt_fltb = tf.train.AdamOptimizer(learning_rate=self.lr_fltb).minimize(self.fltb_loss)
        self.opt_recom = tf.train.AdamOptimizer(learning_rate=self.lr_recom).minimize(self.recom_loss)

    def _init_placeholders(self):
        placeholder = {}
        placeholder['user_input'] = tf.placeholder(tf.int32, shape=(None,), name='user_input')
        placeholder['po_input'] = tf.placeholder(tf.int32, shape=(None,), name='pos_outfit_index')
        placeholder['pl_input'] = tf.placeholder(tf.float32, shape=(None,), name='pos_length_input')
        placeholder['no_input'] = tf.placeholder(tf.int32, shape=(None,), name='neg_outfit_index')
        placeholder['nl_input'] = tf.placeholder(tf.float32, shape=(None,), name='neg_length_input')
        placeholder['fltb_input'] = tf.placeholder(tf.int32, shape=(None, self.max_ol), name='fltb_neg_outfit_map')
        placeholder['flen_input'] = tf.placeholder(tf.float32, shape=(None,), name='fltb_neg_outfit_length')
        placeholder['fadj_input'] = tf.placeholder(tf.float32, shape=(None,self.max_ol,self.max_ol),
                                                   name='fltb_neg_outfit_adj')

        placeholder['_init_visual_feat'] = tf.placeholder(tf.float32, shape=(self.n_items, self.vf_dim),
                                                name='init_visual_feat')
        placeholder['_init_outfit_map'] = tf.placeholder(tf.int32, shape=(self.n_outfits, self.max_ol),
                                               name='init_outfit_map')
        placeholder['_init_outfit_len'] = tf.placeholder(tf.float32, shape=(self.n_outfits,),
                                               name='init_outfit_length')
        placeholder['_init_gather_index'] = tf.placeholder(tf.int32, shape=(self.n_items,),
                                                 name='init_gather_index')
        placeholder['_init_outfit_adj'] = tf.placeholder(tf.float32, shape=(self.n_outfits, self.max_ol, self.max_ol),
                                                         name='init_outfit_adj')

        placeholder['node_dropout'] = tf.placeholder(tf.float32, shape=[None], name='node_dropout')
        placeholder['mess_dropout'] = tf.placeholder(tf.float32, shape=[None], name='mess_dropout')

        cate_index = []
        for c in range(len(self.cate_items_dict)):
            items = self.cate_items_dict[c]
            holder = tf.placeholder(tf.int32, shape=(len(items),),
                                                 name='init_cate_index_%d' % c)
            cate_index.append(holder)
        placeholder['_init_cate_index'] = cate_index

        return placeholder

    def _get_outfit_graph_feat(self, adj, X, hl):
        b = tf.shape(X)[0]
        m = tf.shape(X)[1]
        d = tf.shape(X)[2]
        ego = tf.reshape(X, [-1,d])
        side_mess = tf.matmul(adj, X)
        com_mess = tf.reshape(tf.multiply(side_mess, X), [-1,d])

        com_mess = tf.nn.leaky_relu(
                tf.matmul(com_mess,  self.weights['Wi_mess']) + self.weights['bi_mess'])  #[b*5,d]

        ego_mess = tf.nn.leaky_relu(
            tf.matmul(ego, self.weights['Wi_ego']) + self.weights['bi_ego'])
        new_node = tf.reshape(com_mess+ ego_mess, [b, m, d]) #[b,5,d]

        mask = tf.expand_dims(tf.sequence_mask(hl, maxlen=self.max_ol, dtype=tf.float32), axis=2) # [b, 5,1]

        new_node = mask*new_node

        return new_node

    def _get_outfit_hier_embed(self, neighs, nodes):

        neigh_ = tf.reduce_sum(neighs, axis=1)  # [N, d]
        neigh_mess = tf.nn.leaky_relu(
                tf.matmul(neigh_,  self.weights['Wo_mess']) + self.weights['bo_mess'])  #[N,d]
        neigh_mess = tf.nn.l2_normalize(neigh_mess, axis=-1)

        new_nodes = tf.add(neigh_mess, nodes)

        return new_nodes

    def _get_user_hier_node(self, adj, Xo, Xu):

        if self.node_dropout_flag:
            # node dropout.
            A_fold_hat = self._split_A_hat_node_dropout(adj)
        else:
            A_fold_hat = self._split_A_hat(adj)


        '''get user layer embedding.'''

        temp_embed = []
        for f in range(self.n_fold):
            temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], Xo))

        # sum messages of neighbors.
        side_embeddings = tf.concat(temp_embed, 0)  # [N, d]

        neigh_mess = tf.nn.leaky_relu(
            tf.matmul(side_embeddings, self.weights['Wu_mess']) + self.weights['bu_mess'])  # [N,d]

        neigh_mess = tf.nn.l2_normalize(neigh_mess, axis=-1)

        new_nodes = tf.add(neigh_mess, Xu)

        return new_nodes

    def _create_fltb_loss(self, pos_score, neg_score):

        fltb_loss = tf.reduce_mean(tf.nn.softplus(tf.negative(pos_score - neg_score)))

        return fltb_loss


    def _compatibility_score(self, ilatents,ilen): # [b,5,dim]

        b = tf.shape(ilatents)[0]
        m = tf.shape(ilatents)[1]
        d = tf.shape(ilatents)[2]
        latents = tf.reshape(ilatents, [-1,d]) # [b*5,d]

        iatt = tf.nn.softmax(tf.matmul(tf.nn.leaky_relu(tf.matmul(latents, self.weights['W_com_att0'])),
                                       self.weights['W_com_att1']), axis=-1)  # [b*5,8]

        iscore = tf.nn.tanh(tf.matmul(tf.nn.leaky_relu(tf.matmul(latents, self.weights['W_com_score0'])),
                                       self.weights['W_com_score1']))  # [b*5,8]

        y = tf.reshape(iatt * iscore, [b, m,self.r_view]) # [b,5]
        y = tf.reduce_sum(y, axis=2) # [b,5]

        score = tf.reduce_mean(y, axis=1)


        return score

    def _get_cate_feats(self, item_feat, cate_items):
        temp_feat = []
        for c in range(len(cate_items)):
            v_feat = tf.nn.embedding_lookup(item_feat, cate_items[c])
            v_feat = tf.nn.dropout(v_feat, keep_prob=0.8)
            c_feat = tf.matmul(v_feat, self.W_cate[c][0])
            c_feat = tf.nn.leaky_relu(c_feat)
            c_feat = tf.nn.dropout(c_feat, keep_prob=0.8)
            c_feat = tf.matmul(c_feat, self.W_cate[c][1])
            c_feat = tf.nn.leaky_relu(c_feat)
            temp_feat.append(c_feat)

        cate_feats = tf.concat(temp_feat, axis=0)
        item_cate_feats = tf.nn.embedding_lookup(cate_feats, self.gather_index)
        pad = tf.zeros([1, self.cate_dim], dtype=tf.float32)
        item_cate_feats = tf.concat([item_cate_feats, pad], axis=0)

        return item_cate_feats

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        """Create embedding for user and outfit node."""
        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
            all_weights['outfit_embedding'] = tf.Variable(initializer([self.n_outfits, self.emb_dim]), name='outfit_embedding')

        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['outfit_embedding'] = tf.Variable(initial_value=self.pretrain_data['outfit_embed'], trainable=True,
                                                          name='outfit_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        """Create untrainable weights"""

        all_weights['visual_feat'] = tf.get_variable(name="visual_feat",
                                                    shape=[self.n_items, self.vf_dim],
                                                    dtype=tf.float32,
                                                    initializer=tf.initializers.constant(0),
                                                    trainable=False)

        all_weights['outfit_map'] = tf.get_variable(name="outfit_map",
                                                    shape=[self.n_outfits, self.max_ol],
                                                    dtype=tf.int32,
                                                    initializer=tf.initializers.constant(0),
                                                    trainable=False)
        all_weights['outfit_adj'] = tf.get_variable(name="outfit_adj",
                                                    shape=[self.n_outfits, self.max_ol, self.max_ol],
                                                    dtype=tf.float32,
                                                    initializer=tf.initializers.constant(0),
                                                    trainable=False)
        all_weights['gather_index'] = tf.get_variable(name="gather_index",
                                                      shape=[self.n_items],
                                                      dtype=tf.int32,
                                                      initializer=tf.initializers.constant(0),
                                                      trainable=False)
        all_weights['outfit_len'] = tf.get_variable(name="outfit_len",
                                                      shape=[self.n_outfits],
                                                      dtype=tf.float32,
                                                      initializer=tf.initializers.constant(0),
                                                      trainable=False)
        cate_index = []
        for c in range(len(self.cate_items_dict)):
            items = self.cate_items_dict[c]
            weight = tf.get_variable(name='cate_index_%d' % c,
                                     shape=[len(items)],
                                     dtype=tf.int32,
                                     initializer=tf.initializers.constant(0),
                                     trainable=False)
            self.save_weights.append(weight)
            cate_index.append(weight)
        all_weights['cate_index'] = cate_index

        """Create parameters for category encoders."""
        W_cate = []
        for c in range(self.n_cates):
            weights = []
            wc0 = tf.Variable(
                initializer([self.vf_dim, self.vf_dim // 2]), name='W_cate_%d_0' % c)
            weights.append(wc0)
            self.save_weights.append(wc0)
            wc1 = tf.Variable(
                initializer([self.vf_dim // 2, self.cate_dim]), name='W_cate_%d_1' % c)
            weights.append(wc1)
            self.save_weights.append(wc1)

            W_cate.append(weights)

        all_weights['W_cate'] = W_cate

        """create parameters for compatibility score."""
        all_weights['W_com_att0'] = tf.Variable(
            initializer([self.cate_dim, self.cate_dim//2]), name='W_com_att0')
        self.save_weights.append(all_weights['W_com_att0'])
        all_weights['W_com_att1'] = tf.Variable(
            initializer([self.cate_dim//2, self.r_view]), name='W_com_att1')
        self.save_weights.append(all_weights['W_com_att1'])
        all_weights['W_com_score0'] = tf.Variable(
            initializer([self.cate_dim, self.cate_dim//2]), name='W_com_score0')
        self.save_weights.append(all_weights['W_com_score0'])
        all_weights['W_com_score1'] = tf.Variable(
            initializer([self.cate_dim//2, self.r_view]), name='W_com_score1')
        self.save_weights.append(all_weights['W_com_score1'])


        """Create parameters for message pass."""
        for l in ['i', 'o', 'u']:
            all_weights['W%s_ego'%l] = tf.Variable(
                initializer([self.cate_dim, self.cate_dim]), name='W%s_ego'%l)
            self.save_weights.append(all_weights['W%s_ego'%l])
            all_weights['b%s_ego'%l] = tf.Variable(
                initializer([1, self.cate_dim]), name='b%s_ego'%l)
            self.save_weights.append(all_weights['b%s_ego'%l])
            all_weights['W%s_mess'%l] = tf.Variable(
                initializer([self.cate_dim, self.cate_dim]), name='W%s_mess'%l)
            self.save_weights.append(all_weights['W%s_mess'%l])
            all_weights['b%s_mess'%l] = tf.Variable(
                initializer([1, self.cate_dim]), name='b%s_mess'%l)
            self.save_weights.append(all_weights['b%s_mess'%l])

        for l in ['o', 'u']:
            all_weights['W%s_inter'%l] = tf.Variable(
                initializer([self.cate_dim, 1]), name='W%s_inter'%l)
            self.save_weights.append(all_weights['W%s_inter'%l])
            all_weights['b%s_inter' % l] = tf.Variable(
                initializer([1, self.cate_dim]), name='b%s_inter' % l)
            self.save_weights.append(all_weights['b%s_inter' % l])

        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []
        fold_len = (self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))

        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def create_bpr_loss(self, users, pos_items, neg_items):
        self.recom_pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        self.recom_neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer/self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(tf.negative(self.recom_pos_scores - self.recom_neg_scores)))

        reg_loss = self.decay * regularizer

        emb_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

def load_pretrained_data():
    pretrain_path = '%spretrain/%s/%s.npz' % (args.proj_path, args.dataset, 'embedding')
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data

def _get_gather_index(cate_items):
    index_list = []
    for c in range(len(cate_items)):
        items = cate_items[c]
        index_list.append(items)

    index_ = np.concatenate(index_list, axis=0)
    gather_index = np.argsort(index_)

    return gather_index


if __name__ == '__main__':
    # args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)  # args defined in bath_test.py

    config = dict()

    data_generator = Dataset.Data(path=args.data_path + args.dataset)
    config['n_users'] = data_generator.n_users
    config['n_outfits'] = data_generator.n_train_outfits
    config['n_items'] = data_generator.n_all_items
    config['n_cates'] = data_generator.n_cates
    config['max_ol'] = data_generator.max_ol
    config['visual_feat'] = data_generator.visual_feat
    config['vf_dim'] = data_generator.vf_dim
    config['cate_items'] = data_generator.cate_item_dict
    config['norm_uo_adj'] = data_generator.norm_uo_adj

    gather_index = _get_gather_index(data_generator.cate_item_dict)

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    t0 = time()

    if args.train_mode == 0:
        print('emb dim', args.embed_size, 'lr', args.lr)
    elif args.train_mode == 1:
        print('emb dim', args.embed_size, 'fltb lr', args.fltb_lr, 'recommend lr', args.recom_lr)

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = HFGN(data_config=config, pretrain_data=pretrain_data)

    """
    *********************************************************
    Save the model parameters.
    """

    if args.save_flag == 1: # layer_size: output sizes of every layer.
        if args.train_mode == 0:
            weights_save_path = '%sweights/%s/l%s' % (
                args.weights_path, model.model_type, str(args.lr))
        elif args.train_mode == 1:
            weights_save_path = '%sweights/%s/fl%s_rl%s' % (
                args.weights_path, model.model_type, str(args.fltb_lr),
            str(args.recom_lr))
        ensureDir(weights_save_path)

        save_saver = tf.train.Saver(model.save_weights, max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    """
    *********************************************************
    Reload the pretrained model parameters.
    """
    if args.pretrain == 1:

        pretrain_path = args.pretrain_path

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            save_saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_hit_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_hit_0 = 0.
        print('without pretraining.')

    sess.run(model.assign_feat, feed_dict={model._init_visual_feat: data_generator.visual_feat})
    sess.run(model.assign_map, feed_dict={model._init_outfit_map: data_generator.outfit_map})
    sess.run(model.assign_length,feed_dict={model._init_outfit_len: data_generator.outfit_len})
    sess.run(model.assign_gather, feed_dict={model._init_gather_index:gather_index})
    sess.run(model.assign_adj, feed_dict={model._init_outfit_adj:data_generator.outfit_adj})
    for c in range(data_generator.n_cates):
        sess.run(model.assign_cate_index[c], feed_dict={model._init_cate_index[c]: data_generator.cate_item_dict[c]})

    """
    *********************************************************
    Train.
    """
    pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], []
    fltb_auc_loger = []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):  # args.epoch
        t1 = time()

        recom_loss, mf_loss, reg_loss, fltb_loss = 0., 0., 0., 0.
        if args.train_mode == 0:
            '''Train.'''
            t1 = time()
            # generate all batches for the current epoch.
            batch_begin = time()
            batches = batch_generator.sample(data_generator=data_generator, batch_size=args.batch_size)
            batch_time = time() - batch_begin
            if epoch == 0:
                print("batch_time", batch_time)

            num_batch = len(batches[1])
            batch_index = list(range(num_batch))
            np.random.shuffle(batch_index)


            for idx in tqdm(batch_index, ascii=True):
                u_batch, po_batch, plen_batch, no_batch, nlen_batch, \
                f_batch, flen_batch, fadj_batch = batch_generator.batch_get(batches, idx)

                _, batch_loss, batch_recom_loss, batch_mf_loss, batch_reg_loss, batch_fltb_loss = sess.run(
                    [model.opt, model.loss, model.recom_loss, model.mf_loss, model.reg_loss, model.fltb_loss],
                    feed_dict={model.user_input: u_batch, model.po_input: po_batch,
                               model.pl_input: plen_batch, model.no_input: no_batch, model.nl_input: nlen_batch,
                               model.fltb_input: f_batch, model.flen_input: flen_batch, model.fadj_input: fadj_batch,
                               model.node_dropout: eval(args.node_dropout),
                               model.mess_dropout: eval(args.mess_dropout)
                               })

                recom_loss += batch_loss
                mf_loss += batch_mf_loss
                reg_loss += batch_reg_loss
                fltb_loss += batch_fltb_loss

            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs] train==[loss=%.5f mf_loss= %.5f, reg_loss=%.5f, fltb_loss=%.5f]' % (
                epoch, time() - t1, recom_loss, mf_loss, reg_loss, fltb_loss)
                print(perf_str)

        elif args.train_mode == 1:
            '''recommendation train.'''
            t1 = time()
            batch_begin = time()
            batches = recom_batch_generator.sample(data_generator=data_generator, batch_size=args.batch_size)
            batch_time = time() - batch_begin
            if epoch == 0:
                print("batch_time", batch_time)

            num_batch = len(batches[1])
            batch_index = list(range(num_batch))
            np.random.shuffle(batch_index)

            for idx in tqdm(batch_index, ascii=True):
                u_batch, po_batch, plen_batch, no_batch, nlen_batch = recom_batch_generator.batch_get(batches, idx)

                _, batch_loss, batch_mf_loss, batch_reg_loss = sess.run(
                    [model.opt_recom, model.recom_loss, model.mf_loss, model.reg_loss],
                    feed_dict={model.user_input: u_batch, model.po_input: po_batch,
                               model.pl_input: plen_batch, model.no_input: no_batch, model.nl_input: nlen_batch,

                               model.node_dropout: eval(args.node_dropout),
                               model.mess_dropout: eval(args.mess_dropout)
                               })

                recom_loss += batch_loss
                mf_loss += batch_mf_loss
                reg_loss += batch_reg_loss

            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs] recom loss: train==[%.5f= %.5f + %.5f]' % (
                    epoch, time() - t1, recom_loss, mf_loss, reg_loss)
                print(perf_str)

            '''fltb train.'''
            t1 = time()
            batch_begin = time()
            batches = fltb_batch_generator.sample(data_generator=data_generator, batch_size=args.batch_size)
            batch_time = time() - batch_begin
            if epoch == 0:
                print("batch_time", batch_time)

            num_batch = len(batches[1])
            batch_index = list(range(num_batch))
            np.random.shuffle(batch_index)

            for idx in tqdm(batch_index, ascii=True):
                po_batch, plen_batch, f_batch, flen_batch, fadj_batch = fltb_batch_generator.batch_get(batches, idx)

                _, batch_fltb_loss = sess.run([model.opt_fltb, model.fltb_loss],
                                              feed_dict={model.po_input: po_batch, model.pl_input: plen_batch,
                                                         model.fltb_input: f_batch, model.flen_input: flen_batch,
                                                         model.fadj_input: fadj_batch,
                                                         model.node_dropout: eval(args.node_dropout),
                                                         model.mess_dropout: eval(args.mess_dropout)
                                                         })
                fltb_loss += batch_fltb_loss

            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs] fltb loss: train==[%.5f]' % (
                    epoch, time() - t1, fltb_loss)
                print(perf_str)


        if (epoch) % 5 != 0:
            continue

        if np.isnan(recom_loss) == True or np.isnan(fltb_loss):
            print('ERROR: loss is nan.')
            sys.exit()

        """
        test ....
        """

        t2 = time()
        users_to_test = list(data_generator.test_u_outfits_dict.keys())
        recom_ret = recom_test.test(sess, model, users_to_test, data_generator, args, drop_flag=True)

        t3 = time()

        rec_loger.append(recom_ret['recall'])
        pre_loger.append(recom_ret['precision'])
        ndcg_loger.append(recom_ret['ndcg'])
        hit_loger.append(recom_ret['hit_ratio'])

        if args.verbose > 0:
            print('recommendation test...')
            perf_str = 'Epoch %d [%.1fs + %.1fs]: recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, recom_ret['recall'][0], recom_ret['recall'][-1],
                        recom_ret['precision'][0], recom_ret['precision'][-1], recom_ret['hit_ratio'][0], recom_ret['hit_ratio'][-1],
                        recom_ret['ndcg'][0], recom_ret['ndcg'][-1])
            print(perf_str)

        fltb_ret = fltb_test.test(sess, model, data_generator, args)
        fltb_auc_loger.append(fltb_ret['auc'])

        if args.verbose > 0:
            print('fltb test...')
            perf_str = 'Epoch %d : accuracy=[%.5f]' % (epoch, fltb_ret['auc'][0])
            print(perf_str)

        cur_best_hit_0, stopping_step, should_stop = early_stopping(recom_ret['hit_ratio'][0], cur_best_hit_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_hit_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & outfit embeddings for pretraining.
        if recom_ret['hit_ratio'][0] == cur_best_hit_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hits = np.array(hit_loger)
    fltbs = np.array(fltb_auc_loger)

    best_rec_0 = max(hits[:, 0])
    idx = list(hits[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hits[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s.result' % (args.proj_path,model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    if args.train_mode == 0:
        f.write(
            'embed_size=%d, lr=%.4f, embed_size=%s, regs=%s \n\t%s\n'
            % (args.embed_size, args.lr, args.embed_size, args.regs, final_perf))
    elif args.train_mode == 1:
        f.write(
            'embed_size=%d, fltb_lr=%.4f, recom_lr=%.4f, embed_size=%s, regs=%s \n\t%s\n'
            % (args.embed_size, args.fltb_lr, args.recom_lr, args.embed_size, args.regs, final_perf))
    f.close()