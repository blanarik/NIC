import pandas as pd
import numpy as np
from multiprocessing import Pool

NUM_THREADS = 16

def paralelize_recommend(uid):
    i_list = user_items[uid]
    items = [item for item in pop_items if item not in i_list] # remove already seen items
    return {'user_id': uid, 'top100': [int(iid) for iid in items[:100]]}


train = pd.read_csv('data/ustore/raw/history.csv')

i_occurences = train.item_id.value_counts()
pop_items = i_occurences.index.tolist()

pop_items = [iid for iid in pop_items]
user_items = train.groupby('user_id')['item_id'].apply(list)

rec_list = []

with Pool(NUM_THREADS) as p:
    rec_list = p.map(paralelize_recommend, set(train[train.user_eval_set == 'test'].user_id.values))
recommended = pd.DataFrame(rec_list)
recommended.to_csv('data/ustore/predictions/popular.csv', index=False)
