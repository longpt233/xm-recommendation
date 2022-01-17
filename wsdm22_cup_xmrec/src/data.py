import os
import torch
import random
import resource
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import numpy as np
from time import time

random.seed(0)


class Central_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """
    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.last_user_index = 0
        self.last_item_index = 0
        
    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]
    
    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]
    
    def query_user_id(self, user_index):
        user_index_id = {v:k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            print(f'USER index {user_index} is not valid!')
            return 'xxxxx'
        
    def query_item_id(self, item_index):
        item_index_id = {v:k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid!')
            return 'yyyyy'

    
    

class MetaMarket_DataLoader(object):
    """Data Loader for a few markets, samples task and returns the dataloader for that market"""
    
    def __init__(self, task_list, sample_batch_size, task_batch_size=2, shuffle=True, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        self.num_tasks = len(task_list)
        self.task_list = task_list
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_batch_size = sample_batch_size
        self.task_list_loaders = {
            idx:DataLoader(task_list[idx], batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        self.task_batch_size = min(task_batch_size, self.num_tasks)
    
    def refresh_dataloaders(self):
        self.task_list_loaders = {
            idx:DataLoader(self.task_list[idx], batch_size=self.sample_batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        
    def get_iterator(self, index):
        return self.task_list_iters[index]
        
    def sample_task(self):
        sampled_task_idx = random.randint(0, self.num_tasks-1)
#         print(f'task number {sampled_task_idx} sampled')
        return self.task_list_loaders[sampled_task_idx]
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        return self.task_list_loaders[index]
    

        
        
class MetaMarket_Dataset(object):
    """
    Wrapper around market data (task)
    ratings: {
      0: us_market_gen,
      1: de_market_gen,
      ...
    }
    """
    def __init__(self, task_gen_dict, meta_split, id_index_bank, config):
        self.config =  config
        self.split = self.config['A_split']
        self.folds = self.config['a_fold']
        self.num_tasks = len(task_gen_dict)
        self.id_index_bank = id_index_bank
        if meta_split=='train':
            self.task_gen_dict = {idx:cur_task.instance_a_market_train_task(idx) for idx, cur_task  in task_gen_dict.items()}
        else:
            self.task_gen_dict = {idx:cur_task.instance_a_market_valid_task(idx, split=meta_split) for idx, cur_task  in task_gen_dict.items()}
        
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        return self.task_gen_dict[index]

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce())
        return A_fold

    def getSparseGraph(self, pre_adj_mat_path='./DATA'):
        self.Graph = None
        trainUser, trainItem = [], []
        for idx, cur_dataset in self.task_gen_dict.items():
            for idx in range(len(cur_dataset)):
                user_idx, item_idx, _ = cur_dataset[idx]
                trainUser.append(user_idx.item())
                trainItem.append(item_idx.item())
        self.trainUser = np.array(trainUser)        
        self.trainItem = np.array(trainItem)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        self.path = pre_adj_mat_path
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + f'/s_pre_adj_mat_{self.config["tgt_market"]}_{self.config["src_markets"]}_{self.config["exp_name"]}.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + f'/s_pre_adj_mat_{self.config["tgt_market"]}_{self.config["src_markets"]}_{self.config["exp_name"]}.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce()
                print("don't split the matrix")
        return self.Graph

    @property
    def n_users(self):
        return self.id_index_bank.last_user_index + 1
    
    @property
    def m_items(self):
        return self.id_index_bank.last_item_index + 1


class MarketTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset
    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset
    """
    def __init__(self, task_index, user_tensor, pos_item_tensor, neg_item_tensor=None):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.task_index = task_index
        self.user_tensor = user_tensor
        self.pos_item_tensor = pos_item_tensor
        self.neg_item_tensor = neg_item_tensor
        if not self.neg_item_tensor is None:
            self.neg_item_tensor = neg_item_tensor

        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        if not self.neg_item_tensor is None:
            return self.user_tensor[index], self.pos_item_tensor[index], self.neg_item_tensor[index]
        else:
            return self.user_tensor[index], self.pos_item_tensor[index]


    

class TaskGenerator(object):
    """Construct dataset"""

    def __init__(self, train_data, id_index_bank):
        """
        args:
            train_data: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rating']
            id_index_bank: converts ids to indices 
        """

        self.id_index_bank = id_index_bank
        
        # None for evaluation purposes
        if train_data is not None: 
            self.ratings = train_data

            # get item and user pools
            self.user_pool_ids = set(self.ratings['userId'].unique())
            self.item_pool_ids = set(self.ratings['itemId'].unique())

            # replace ids with corrosponding index for both users and items
            self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.id_index_bank.query_user_index(x) )
            self.ratings['itemId'] = self.ratings['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x) )

            # get item and user pools (indexed version)
            self.user_pool = set(self.ratings['userId'].unique())
            self.item_pool = set(self.ratings['itemId'].unique())

            # create negative item samples
            self.negatives_train = self._sample_negative(self.ratings)
            self.train_ratings = self.ratings
    
    def _sample_negative(self, ratings):
        by_userid_group = self.ratings.groupby("userId")['itemId']
        negatives_train = {}
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids
            neg_itemids_train = neg_itemids
            negatives_train[userid] = neg_itemids_train
        return negatives_train
                    
        
    def instance_a_market_train_task(self, index):
        """instance train task's torch Dataset"""
        users, pos_items, neg_items = [], [], []
        train_ratings = self.train_ratings
        for row in train_ratings.itertuples():
            #user and pos item
            users.append(int(row.userId))
            pos_items.append(int(row.itemId))
            
            #neg item
            cur_negs = self.negatives_train[int(row.userId)]
            cur_neg = random.sample(cur_negs, 1)
            neg_items.append(int(cur_neg[0]))   
            

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                                        pos_item_tensor=torch.LongTensor(pos_items),
                                        neg_item_tensor=torch.FloatTensor(neg_items))
        return dataset
        
    
    def load_market_valid_run(self, valid_run_file):
        users, items = [], []
        with open(valid_run_file, 'r') as f:
            for line in f:
                linetoks = line.split('\t')
                user_id = linetoks[0]
                item_ids = linetoks[1].strip().split(',')
                for cindex, item_id in enumerate(item_ids):
                    users.append(self.id_index_bank.query_user_index(user_id))
                    items.append(self.id_index_bank.query_item_index(item_id))

        dataset = MarketTask(0, user_tensor=torch.LongTensor(users),
                                            pos_item_tensor=torch.LongTensor(items))
        return dataset
    
    def instance_a_market_valid_dataloader(self, valid_run_file, sample_batch_size, shuffle=False, num_workers=0):
        """instance target market's validation data torch Dataloader"""
        dataset = self.load_market_valid_run(valid_run_file)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers)