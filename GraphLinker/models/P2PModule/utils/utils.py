from email import header
import pandas as pd
from requests import head

# 建立制品信息的全局唯一标识
def gen_global_identifier(offsets, train_pos_):
  num_issue, num_pr = offsets[2], offsets[3]
  offsets = [offsets[0] + offsets[1], offsets[0] + offsets[1] + offsets[2]]
  
  edge_pair_list = []
  for pos_pair in train_pos_:
    type_ = pos_pair[2]
    if type_==0:
        pair_list_indice = [pos_pair[0]+offsets[0], pos_pair[1]+offsets[1]]
    elif type_==1:
        pair_list_indice = [pos_pair[0]+offsets[0], pos_pair[1]+offsets[0]]
    else:
        pair_list_indice = [pos_pair[0]+offsets[1], pos_pair[1]+offsets[1]]
    edge_pair_list.append(pair_list_indice)
  
  edge_df = pd.DataFrame(edge_pair_list)
  art_adj = {}
  for i in range(num_issue):
      key = i + offsets[0]
      re = edge_df[edge_df.isin([key]).any(axis=1)].index
      value = []
      for row in re:
        adj_key = edge_df.iloc[row]
        v_ = adj_key[0] if adj_key[1]==key else adj_key[1]
        value.append(v_)
      art_adj[key] = value
  
  for i in range(num_pr):
    key = i + offsets[1]
    re = edge_df[edge_df.isin([key]).any(axis=1)].index
    value = []
    for row in re:
      adj_key = edge_df.iloc[row]
      v_ = adj_key[0] if adj_key[1]==key else adj_key[1]
      value.append(v_)
    art_adj[key] = value
  
  return art_adj

def transferArtIdIntoIndex(P2Pdatasets, P2PLabels, mapping):
  dataset = {"pos":[], "neg":[]}
  for idx, pair in enumerate(P2Pdatasets):
    label = P2PLabels[idx]
    temp = []
    type_ls = []
    for p in pair:
        p_, type_ = mapping.loc[p].tolist()
        temp.append(int(p_))
        type_ls.append(type_)
    if type_ls[0]=="issue":
      if type_ls[1]=="issue":
        type_int = 1
      else:
        type_int = 0
    else:
      if type_ls[1]=="issue":
        type_int = 0
        temp = [temp[1], temp[0]]
      else:
        type_int = 2
    
    temp.append(type_int)

    if label==0:
        dataset["neg"].append(temp)
    else:
        dataset["pos"].append(temp)

  return dataset