import argparse
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import utils.metric
from utils import *
from models import *
from tqdm import tqdm
import pc_gender_train
import setproctitle
import scipy.stats as stats
path1='data/Book-Crossing/BX-Users.csv'
df = pd.read_csv(path1,encoding='latin1',sep=';')
user_feature={}
for i in range(278858):
    u_data = []
    if np.isnan(df.Age[i]):
        u_data.append(-1)
    else:
        u_data.append(int(df.Age[i]))
    words=df.Location[i].split(',')
    third_word = words[2] if len(words) > 2 else None
    u_data.append(third_word)
    user_feature[i]=u_data

ISBN2ID={}
item_title=[]
path2='data/Book-Crossing/BX-Books.csv'
with open(path2, 'r',encoding='latin1') as file:
    for i, line in enumerate(file):
        fields = line.split(';')
        if i>=1:
            ISBN2ID[fields[0]]=i-1
            item_title.append(fields[1])
print(1)
u_list=[]
i_list=[]
rat_list=[]
path3='data/Book-Crossing/BX-Book-Ratings.csv'
with open(path3, 'r',encoding='latin1') as file:
    for i, line in enumerate(file):
        fields = line.split(';')
        if i>=1:
            userid=int(fields[0].replace('"', ''))
            try:
                itemid=ISBN2ID[fields[1]]
            except KeyError as e:
                continue
            rating=int(fields[2].replace('"', ''))
            u_list.append(userid)
            i_list.append(itemid)
            rat_list.append(rating)
print(1)
train_u2i={}
for u in range(278858):
    train_u2i[u]=[]
for index,data in enumerate(u_list):
    train_u2i[data].append(i_list[index])
print(1)
for i in range(278858):
    if len(train_u2i[i]) <= 15 or user_feature[i][0]==-1:
        del train_u2i[i]
new_user_feature={}
new_train_u2i={}
index=0
for u in train_u2i.keys():
    new_user_feature[index]=user_feature[u]
    new_train_u2i[index]=train_u2i[u]
    index=index+1
print(1)
index_map={}
item_index=0
new_item_title=[]
for u in new_train_u2i.keys():
    items=[]
    for iid in new_train_u2i[u]:
        if iid in index_map:
           items.append(index_map[iid])
        else:
           index_map[iid]=item_index
           new_item_title.append(item_title[iid])
           item_index=item_index+1
           items.append(index_map[iid])
    new_train_u2i[u]=items

age=[]
region=[]
for u in new_user_feature.keys():
    age.append(new_user_feature[u][0])
    region.append(new_user_feature[u][1])
age=np.array(age)
u_feature={'age':age,'region':region}
test_u2i={}
train_u2i={}
for u in new_train_u2i.keys():
    items=new_train_u2i[u]
    k=int(0.8*len(items))
    train_u2i[u]=items[:k]
    test_u2i[u]=items[k:]
train_set={}
userid=[]
itemid=[]
rat=[]
for u in train_u2i.keys():
    k=len(train_u2i[u])
    userid.extend([u]*k)
    itemid.extend(train_u2i[u])
    rat.extend([1]*k)
userid=np.array(userid)
itemid=np.array(itemid)
rating=np.array(rat)
train_set={'userid':userid,'itemid':itemid,'rating':rating}
userid=[]
itemid=[]
rat=[]
test_set={}
for u in test_u2i.keys():
    k=len(test_u2i[u])
    userid.extend([u]*k)
    itemid.extend(test_u2i[u])
    rat.extend([1]*k)
userid=np.array(userid)
itemid=np.array(itemid)
rating=np.array(rat)
test_set={'userid':userid,'itemid':itemid,'rating':rating}
item_feature={'title':new_item_title}
user_num=len(train_u2i)
item_num=len(new_item_title)
save_path='./data/Book-Crossing/process/process.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(train_u2i, f)
    pickle.dump(test_u2i, f)
    pickle.dump(train_set, f)
    pickle.dump(test_set, f)
    pickle.dump(u_feature, f)
    pickle.dump(item_feature, f)
    pickle.dump((user_num, item_num), f)
print(1)




















# path='./data/ml-100k/u.user'
# user_side_features={}
# userid=[]
# gender=[]
# age=[]
# occ=[]
# with open(path, 'r', encoding='latin-1') as file:
#     for line in file:
#         line = line.strip()
#         fields = line.split("|")
#         userid.append(int(fields[0])-1)
#         age.append(int(fields[1])-1)
#         if fields[2]=='M':
#             gender.append(1)
#         else:
#             gender.append(0)
#         occ.append(fields[3])
# n_users=len(userid)
# userid=np.array(userid)
# age=np.array(age)
# gender=np.array(gender)
# user_side_features['userid']=userid
# user_side_features['gender']=gender
# user_side_features['age']=age
# user_side_features['occ']=occ
#
# path1='./data/ml-100k/u.item'
# path2='./data/ml-100k/u.genre'
# item_side_features={}
# itemid=[]
# title=[]
# genre=[]
# order_genre=[]
# with open(path2, 'r', encoding='latin-1') as file:
#     for line in file:
#         line = line.strip()
#         fields = line.split("|")
#         order_genre.append(fields[0])
#
# item_side_features={}
# itemid=[]
# title=[]
# genre=[]
# with open(path1, 'r', encoding='latin-1') as file:
#     for line in file:
#         line = line.strip()
#         fields = line.split("|")
#         itemid.append(int(fields[0]) - 1)
#         title.append(fields[1])
#         item_genre=[]
#         for i in range(6,24):
#             if fields[i]=='1':
#                 item_genre.append(order_genre[i-5])
#         genre.append(item_genre)
# n_items=len(itemid)
# item_id=np.array(itemid)
# item_side_features['itemid']=item_id
# item_side_features['title']=title
# item_side_features['genre']=genre
#
# path3='./data/ml-100k/u.data'
# whole_u2i={}
# for u in range(n_users):
#     whole_u2i[u]=[]
# with open(path3, 'r', encoding='latin-1') as file:
#     for line in file:
#         line = line.strip()
#         fields = line.split('\t')
#         uid=int(fields[0])-1
#         iid=int(fields[1])-1
#         # rat=int(fields[2])
#         # userid.append(uid)
#         # itemid.append(iid)
#         # rating.append(rat)
#         whole_u2i[uid].append(iid)
# train_u2i={}
# for u in range(n_users):
#     train_u2i[u]=[]
# train_set={}
# train_userid=[]
# train_itemid=[]
# train_rating=[]
# test_u2i={}
# for u in range(n_users):
#     test_u2i[u]=[]
# test_set={}
# test_userid=[]
# test_itemid=[]
# test_rating=[]
# with open(path3, 'r', encoding='latin-1') as file:
#     for line in file:
#         line = line.strip()
#         fields = line.split('\t')
#         uid=int(fields[0])-1
#         iid=int(fields[1])-1
#         rat=int(fields[2])
#         if len(train_u2i[uid])<int(0.8*len(whole_u2i[uid])):
#             train_userid.append(uid)
#             train_itemid.append(iid)
#             train_rating.append(rat)
#             train_u2i[uid].append(iid)
#         else:
#             test_userid.append(uid)
#             test_itemid.append(iid)
#             test_rating.append(rat)
#             test_u2i[uid].append(iid)
#
#
# train_userid=np.array(train_userid)
# train_itemid=np.array(train_itemid)
# train_rating=np.array(train_rating)
# train_set['userid']=train_userid
# train_set['itemid']=train_itemid
# train_set['rating']=train_rating
#
# test_userid=np.array(test_userid)
# test_itemid=np.array(test_itemid)
# test_rating=np.array(test_rating)
# test_set['userid']=test_userid
# test_set['itemid']=test_itemid
# test_set['rating']=test_rating
# # path4='./data/ml-100k/u1.test'
# # test_u2i={}
# # for u in range(n_users):
# #     test_u2i[u]=[]
# # test_set={}
# # userid=[]
# # itemid=[]
# # rating=[]
# # with open(path4, 'r', encoding='latin-1') as file:
# #     for line in file:
# #         line = line.strip()
# #         fields = line.split('\t')
# #         uid=int(fields[0])-1
# #         iid=int(fields[1])-1
# #         rat=int(fields[2])
# #         userid.append(uid)
# #         itemid.append(iid)
# #         rating.append(rat)
# #         test_u2i[uid].append(iid)
#
#
# save_path='./data/ml-100k/process/process.pkl'
# with open(save_path, 'wb') as f:
#     pickle.dump(train_u2i, f)
#     pickle.dump(test_u2i, f)
#     pickle.dump(train_set, f)
#     pickle.dump(test_set, f)
#     pickle.dump(user_side_features, f)
#     pickle.dump(item_side_features, f)
#     pickle.dump((n_users, n_items), f)