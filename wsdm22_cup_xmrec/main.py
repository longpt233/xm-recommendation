import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import os
import json
import resource
import sys
import pickle

sys.path.insert(1, 'src')
from model import Model
from utils import *
from data import *




def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser(description="Go lightGCN")
    
    # DATA  Arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t1') 
    parser.add_argument('--src_markets', help='specify none ("none") or a few source markets ("-" seperated) to augment the data for training', type=str, default='s1-s2') 
    
    parser.add_argument('--tgt_market_valid', help='specify validation run file for target market', type=str, default='DATA/t1/valid_run.tsv')
    parser.add_argument('--tgt_market_test', help='specify test run file for target market', type=str, default='DATA/t1/test_run.tsv') 
    parser.add_argument('--valid_path', help='combine validation set when training', type=str, default='')
    
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='baseline_toy')
    
    parser.add_argument('--train_data_file', help='the file name of the train data',type=str, default='train_5core.tsv') #'train.tsv' for the original data loading
    
    
    # MODEL arguments 
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--l2_reg', type=float, default=1e-04, help='learning rate')
    
    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keep_prob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--A_split', action='store_true', help='split or not')
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--num_epoch', type=int,default=200)
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--save_trained', action='store_true')
    
    return parser



def main():
    
    parser = create_arg_parser()
    args = parser.parse_args()
    set_seed(args)

    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Running experiment on device: {args.device}')
    
    ############
    ## Target Market data
    ############
    my_id_bank = Central_ID_Bank()
    
    train_file_names = args.train_data_file # 'train_5core.tsv', 'train.tsv' for the original data loading
    
    tgt_train_data_dir = os.path.join(args.data_dir, args.tgt_market, train_file_names)
    tgt_train_ratings = pd.read_csv(tgt_train_data_dir, sep='\t')

    print(f'Loading target market {args.tgt_market}: {tgt_train_data_dir}')
    tgt_task_generator = TaskGenerator(tgt_train_ratings, my_id_bank)
    print('Loaded target data!\n')

    # task_gen_all: contains data for all training markets, index 0 for target market data
    task_gen_all = {
        0: tgt_task_generator
    }  

    ############
    ## Source Market(s) Data
    ############
    src_market_list = args.src_markets.split('-')
    valid_path = args.valid_path
    cur_task_index = 1
    if 'none' not in src_market_list:
        for cur_src_market in src_market_list:
            cur_src_data_dir = os.path.join(args.data_dir, cur_src_market, train_file_names)
            print(f'Loading {cur_src_market}: {cur_src_data_dir}')
            cur_src_train_ratings = pd.read_csv(cur_src_data_dir, sep='\t')
            cur_src_task_generator = TaskGenerator(cur_src_train_ratings, my_id_bank)
            task_gen_all[cur_task_index] = cur_src_task_generator
            cur_task_index+=1
    if valid_path != '':
        cur_val_data_dir = os.path.join(args.data_dir, args.tgt_market, valid_path)
        print(f'Loading valid {args.tgt_market}: {cur_val_data_dir}')
        cur_val_ratings = pd.read_csv(cur_val_data_dir, sep='\t')
        cur_val_generator = TaskGenerator(cur_val_ratings, my_id_bank)
        task_gen_all[cur_task_index] = cur_val_generator
        cur_task_index+=1
            

    train_tasksets = MetaMarket_Dataset(task_gen_dict=task_gen_all, meta_split='train', id_index_bank=my_id_bank, config=vars(args))
    train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)

    ############
    ## Validation and Test Run
    ############
    tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_valid, args.batch_size)
    tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_test, args.batch_size)
    
    
    ############
    ## Model  
    ############
    mymodel = Model(vars(args), my_id_bank, train_tasksets)
    mymodel.fit(train_dataloader)
    
    print('Run output files:')
    # validation data prediction
    valid_run_mf = mymodel.predict(tgt_valid_dataloader)
    valid_output_file = f'valid_{args.tgt_market}_{args.src_markets}_{args.exp_name}.tsv'
    print(f'--validation: {valid_output_file}')
    write_run_file(valid_run_mf, valid_output_file)
    
    # test data prediction
    test_run_mf = mymodel.predict(tgt_test_dataloader)
    test_output_file = f'test_{args.tgt_market}_{args.src_markets}_{args.exp_name}.tsv'
    print(f'--test: {test_output_file}')
    write_run_file(test_run_mf, test_output_file)
    
    print('Experiment finished successfully!')
    
if __name__=="__main__":
    main()