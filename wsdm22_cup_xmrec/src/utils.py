"""
    Some handy functions for pytroch model training ...
"""
import torch
import sys
import math
import pandas as pd
import random
from evaluation import Evaluator


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)


def write_run_file(rankings, model_output_run):
    with open(model_output_run, 'w') as f:
        f.write(f'userId\titemId\tscore\n')
        for userid, cranks in rankings.items():
            for itemid, score in cranks.items():
                f.write(f'{userid}\t{itemid}\t{score}\n')

def get_run_mf(rec_list, unq_users, my_id_bank):
    ranking = {}    
    for cuser in unq_users:
        user_ratings = [x for x in rec_list if x[0]==cuser]
        user_ratings.sort(key=lambda x:x[2], reverse=True)
        ranking[cuser] = user_ratings

    run_mf = {}
    for k, v in ranking.items():
        cur_rank = {}
        for item in v:
            citem_ind = int(item[1])
            citem_id = my_id_bank.query_item_id(citem_ind)
            cur_rank[citem_id]= 2+item[2]
        cuser_ind = int(k)
        cuser_id = my_id_bank.query_user_id(cuser_ind)
        run_mf[cuser_id] = cur_rank
    return run_mf







