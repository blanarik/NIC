import pandas as pd
import numpy as np

train = pd.read_csv('data/ustore/raw/history.csv')

def sort_key(item):
    return (item[1], -item[0])

i_occurences = train.item_id.value_counts()
pop_items = np.dstack((i_occurences.index.tolist(), i_occurences.values.tolist()))[0].tolist()
pop_items.sort(key=sort_key, reverse=True)
print('sorted')

pop_items = [_[0] for _ in pop_items]
user_items = train.groupby('user_id')['item_id'].apply(list)

rec_list = []
i = 0
for uid in set(train[train.user_eval_set == 'test'].user_id.values):
    items = [item for item in pop_items if item not in user_items[uid]] # remove already seen items
    rec_list.append({'user_id': uid, 'top100': items[:100]})
    i += 1
    if i%1000 == 0:
        print(i)
    
recommended = pd.DataFrame(rec_list)

recommended.to_csv('data/ustore/predictions/popular.csv', index=False)
