import json
import numpy as np
import pandas as pd
import pdb

def preprocess(data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []
    
    with open(data_name) as f:
        s = next(f)
        print(s)
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            u = int(e[0])
            i = int(e[1])
            
            
            
            ts = float(e[2])
            label = int(e[3])
            
            feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            
            feat_l.append(feat)

    feat_l = np.array(feat_l)
    if feat_l.shape[1]==1:
        unique_values = np.unique(feat_l)
        one_hot_vectors = []
        for value in feat_l:
            one_hot_vector = (unique_values == value).astype(int)
            one_hot_vectors.append(one_hot_vector)
        feat_l = np.array(one_hot_vectors)

    return pd.DataFrame({'u': u_list, 
                         'i':i_list, 
                         'ts':ts_list, 
                         'label':label_list, 
                         'idx':idx_list}), feat_l



def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print(new_df.u.max())
    print(new_df.i.max())

  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print(new_df.u.max())
    print(new_df.i.max())

  return new_df


def run(data_name, bipartite=True):
    PATH = './processed/{}.csv'.format(data_name)
    OUT_DF = './processed/ml_{}.csv'.format(data_name)
    OUT_FEAT = './processed/ml_{}.npy'.format(data_name)
    OUT_NODE_FEAT = './processed/ml_{}_node.npy'.format(data_name)
    
    df, feat = preprocess(PATH)
    new_df = reindex(df, bipartite)
    
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])
    
    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    

run('USLegis', bipartite=False) 
# run('CanParl', bipartite=False) 
# run('wikipedia')
# run('reddit')