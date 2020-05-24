import pandas as pd
from implicit.als import AlternatingLeastSquares
import scipy
import numpy as np
import time
from datetime import timedelta
from multiprocessing import Pool


NUM_THREADS = 8
RUN_LIMIT_HOURS = 12

def paralelize_ndcg(uid):
    top = alg.recommend(userid=uid, user_items=user_items, N=100, filter_already_liked_items=True)
    rec_list = {'user_id': uid, 'top100': [_[0] for _ in top]}

    # NDCG@100 per user in validation set
    rec_list_relevance = [1 if x in val_truth[uid] else 0 for x in rec_list['top100']]
    dcg = 0
    for i in range(len(rec_list_relevance)):
        dcg += rec_list_relevance[i]/np.log2(2+i)        
    idcg = 0
    for i in range(len(val_truth[uid])):
        idcg += 1/np.log2(2+i)
    return dcg/idcg

def paralelize_recommend(uid):
    top = alg.recommend(userid=uid, user_items=user_items, N=100, filter_already_liked_items=True)
    return {'user_id': uid, 'top100': [_[0] for _ in top]}

train = pd.read_csv('../data/fstore/raw/history.csv')
# set score to 1 for all interactions
train['aux']=train.apply(lambda x: 1, axis=1)

val_truth = pd.read_csv('../data/fstore/raw/future.csv')
val_truth = val_truth.groupby('user_id')['item_id'].apply(list)

user_items = scipy.sparse.csr_matrix((train.aux, (train.user_id, train.item_id)))
data_to_fit = scipy.sparse.csr_matrix((train.aux, (train.item_id, train.user_id)))
performance_list = []
best_so_far = 0
best_model = None

start = time.time()

while (time.time() - start) /60 /60 < RUN_LIMIT_HOURS:
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list)+1, ' >> training starting...')
    aux_time = time.time()
    
    # hyperparameters
    factors = 25 * np.random.randint(1,31) # 25, 50, 75, ... , 750
    regularization = (10 ** (-np.random.randint(2, 5))) * np.random.randint(1, 10)
    iterations = 25 * np.random.randint(1,31) # 25, 50, 75, ... , 750
    
    alg = AlternatingLeastSquares(num_threads=NUM_THREADS,
        factors=factors, regularization=regularization, iterations=iterations)
    alg.fit(data_to_fit)
    
    perf_ndcg_at_100 = []
    rec_list = []
    
    print(' >> took ', str(timedelta(seconds=time.time() - aux_time)))
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list)+1, ' >> evaluation starting...')
    aux_time = time.time()
    
    with Pool(NUM_THREADS) as p:
        perf_ndcg_at_100 = p.map(paralelize_ndcg, set(train[train.user_eval_set == 'val'].user_id.values))
    
    performance_list.append({'performance': scipy.average(perf_ndcg_at_100),
                             'factors': factors,
                             'regularization': regularization,
                             'iterations': iterations
                            })
    performance_val_users = pd.DataFrame(performance_list)
    performance_val_users.to_csv('../data/fstore/base_tuning/als.csv', index=False)
    
    print(' >> took ', str(timedelta(seconds=time.time() - aux_time)))
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list), ' >> results saved...')
    
    # generate recommendations for test users if this model is best so far
    # storing models led to Memory Error
    if scipy.average(perf_ndcg_at_100) > best_so_far:
        best_so_far = scipy.average(perf_ndcg_at_100)
        best_model = alg
        with Pool(NUM_THREADS) as p:
            rec_list = p.map(paralelize_recommend, set(train[train.user_eval_set == 'test'].user_id.values))
        recommended = pd.DataFrame(rec_list)
        recommended.to_csv('../data/fstore/predictions/als.csv', index=False)
        print('New best model found - NDCG@100:', best_so_far)

alg = best_model
with Pool(NUM_THREADS) as p:
    perf_ndcg_at_100 = p.map(paralelize_ndcg, set(train[train.user_eval_set == 'test'].user_id.values))

print('users with relevant: ', (len(perf_ndcg_at_100) - perf_ndcg_at_100.count(0)) / len(perf_ndcg_at_100))
print('users with relevant: ', (len(perf_ndcg_at_100) - perf_ndcg_at_100.count(0)))
