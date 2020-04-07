import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy
import numpy as np
import time
from datetime import timedelta
from multiprocessing import Pool

NUM_THREADS = 8
RUN_LIMIT_HOURS = 16

def paralelize_recommend(uid):
    i_list = user_items[uid]
    sim_items = pd.DataFrame(columns=['iid', 'dist'])
    for i in i_list:
        sim_items = sim_items.append(pd.DataFrame(columns=['iid', 'dist'], data=per_item_nn[i].tolist()))
    # get nn using strategy sorted by similarity
    if agg_strategy == 'mean':
        sim_items = sim_items.groupby('iid')['dist'].mean().sort_values().index.tolist()
    elif agg_strategy == 'max':
        sim_items = sim_items.groupby('iid')['dist'].max().sort_values().index.tolist()
    elif agg_strategy == 'min':
        sim_items = sim_items.groupby('iid')['dist'].min().sort_values().index.tolist()
    sim_items = [item for item in sim_items if item not in i_list] # remove already seen items
    return {'user_id': uid, 'top100': [int(iid) for iid in sim_items[:100]]}

def paralelize_ndcg(uid):
    rec_list = paralelize_recommend(uid)

    # NDCG@100 per user in validation set
    rec_list_relevance = [1 if x in val_truth[uid] else 0 for x in rec_list['top100']]
    dcg = 0
    for i in range(len(rec_list_relevance)):
        dcg += rec_list_relevance[i]/np.log2(2+i)        
    idcg = 0
    for i in range(len(val_truth[uid])):
        idcg += 1/np.log2(2+i)
    return dcg/idcg

train = pd.read_csv('data/ustore/raw/history.csv')
# set score to 1 for all interactions
train['aux']=train.apply(lambda x: 1, axis=1)

val_truth = pd.read_csv('data/ustore/raw/future.csv')
val_truth = val_truth.groupby('user_id')['item_id'].apply(list)

train_sparse = scipy.sparse.csr_matrix((train.aux, (train.item_id, train.user_id)))
user_items = train.groupby('user_id')['item_id'].apply(list)

performance_list = []
best_so_far = 0
best_model = None

start = time.time()
i = 0

while (time.time() - start) /60 /60 < RUN_LIMIT_HOURS:
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list)+1, ' >> training starting...')
    aux_time = time.time()
    
    # hyperparameters
    agg_strategy = ['mean', 'max'][np.random.randint(0, 2)]
    metric = ['euclidean', 'cosine'][np.random.randint(0, 2)]
    n_neighbors = 101 + np.random.randint(0, 100)
    
    # for min strategy only 101 neighbors are relevant for TOP100 rec
    if i < 2:
        agg_strategy = 'min'
        n_neighbors = 101
        metric = ['euclidean', 'cosine'][i]
    
    alg = NearestNeighbors(metric=metric, algorithm='brute', n_neighbors=n_neighbors, n_jobs=NUM_THREADS).fit(train_sparse)
    distances, indices = alg.kneighbors(train_sparse)
    per_item_nn = np.dstack((indices, distances))
    
    perf_ndcg_at_100 = []
    rec_list = []
    
    print(' >> took ', str(timedelta(seconds=time.time() - aux_time)))
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list)+1, ' >> evaluation starting...')
    aux_time = time.time()
    
    with Pool(NUM_THREADS) as p:
        perf_ndcg_at_100 = p.map(paralelize_ndcg, set(train[train.user_eval_set == 'val'].user_id.values))
        
    performance_list.append({'performance': scipy.average(perf_ndcg_at_100),
                             'agg_strategy': agg_strategy,
                             'metric': metric,
                             'n_neighbors': n_neighbors
                            })
    performance_val_users = pd.DataFrame(performance_list)
    performance_val_users.to_csv('data/ustore/base_tuning/itemknn.csv', index=False)
    
    print(' >> took ', str(timedelta(seconds=time.time() - aux_time)))
    print(str(timedelta(seconds=time.time() - start)), ' -- config #', len(performance_list), ' >> results saved...')
    
    # generate recommendations for test users if this model is best so far
    # storing models led to Memory Error
    if scipy.average(perf_ndcg_at_100) > best_so_far:
        best_so_far = scipy.average(perf_ndcg_at_100)
        print('New best model found - NDCG@100:', best_so_far)
        with Pool(NUM_THREADS) as p:
            rec_list = p.map(paralelize_recommend, set(train[train.user_eval_set == 'test'].user_id.values))
        recommended = pd.DataFrame(rec_list)
        recommended.to_csv('data/ustore/predictions/itemknn.csv', index=False)
        
    i += 1
