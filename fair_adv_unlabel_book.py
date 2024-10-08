import torch
# print(torch.__version__)
import torch.nn as nn

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# print('0000')
import pickle
from torch.utils.data import DataLoader
import torch.nn.functional as F
import copy
import time
import pc_gender_train
import setproctitle
import utils.metric
from utils import *
from models import *
import pc_gender_train_ml100k
setproctitle.setproctitle('python')  # 更改当前进程名
parser = argparse.ArgumentParser(
    description='ml_gcn_fairmi',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--bakcbone', type=str, default='gcn')
parser.add_argument('--dataset', type=str, default='./data/Book-Crossing/process/process.pkl')
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.001)
parser.add_argument('--n_layers', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--log_path', type=str, default='logs/gcn_fairmi.txt')
parser.add_argument('--param_path', type=str, default='param/unlabel_semi_model_19.pth')
parser.add_argument('--pretrain_path', type=str, default='param/book_gcn_base.pth')
parser.add_argument('--lreg', type=float, default=0.1)
parser.add_argument('--ureg', type=float, default=0.08)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--train_step', type=int, default=50)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

print(args)

##movieLens-1M
user_num = 6040  # user_size
item_num = 3952  # item_size
factor_num = 64
batch_size = 1024
with open(args.dataset, 'rb') as f:
    train_u2i = pickle.load(f)
    test_u2i = pickle.load(f)
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    user_side_features = pickle.load(f)
    item_side_features = pickle.load(f)
    n_users, n_items = pickle.load(f)
# l=n_users-int(0.1*n_users)
# new_train_u2i={}
# new_test_u2i={}
# for u in range(n_users):
#     if u<l:
#         new_train_u2i[u]=train_u2i[u]
#         new_test_u2i[u]=test_u2i[u]
# userid=[]
# itemid=[]
# rating=[]
# for index,u in enumerate(train_set['userid']):
#     if u<l:
#         userid.append(u)
#         itemid.append(train_set['itemid'][index])
#         rating.append(train_set['rating'][index])
# new_train_set={}
# new_train_set['userid']=np.array(userid)
# new_train_set['itemid']=np.array(itemid)
# new_train_set['rating']=np.array(rating)
# train_u2i=new_train_u2i
# test_u2i=new_test_u2i
# train_set=new_train_set
# user_num = l  # user_size
# n_users=l
# item_num = n_items
user_num=n_users
item_num=n_items
# training_user_set,training_item_set,training_set_count = np.load(dataset_base_path+'/datanpy/training_set.npy',allow_pickle=True)
# testing_user_set,testing_item_set,testing_set_count = np.load(dataset_base_path+'/datanpy/testing_set.npy',allow_pickle=True)
# val_user_set,val_item_set,val_set_count = np.load(dataset_base_path+'/datanpy/val_set.npy',allow_pickle=True)
# user_rating_set_all,_,_ = np.load(dataset_base_path+'/datanpy/user_rating_set_all.npy',allow_pickle=True)

# training_ratings_dict,train_dict_count = np.load(dataset_base_path+'/data1t5/training_ratings_dict.npy',allow_pickle=True)
# testing_ratings_dict,test_dict_count = np.load(dataset_base_path+'/data1t5/testing_ratings_dict.npy',allow_pickle=True)

# training_u_i_set,training_i_u_set = np.load(dataset_base_path+'/data1t5/training_adj_set.npy',allow_pickle=True)

gcn = torch.load(args.pretrain_path,map_location=args.device)
users_emb_gcn, items_emb_gcn = gcn.forward()
users_emb_gcn = users_emb_gcn.detach().cpu().numpy()
items_emb_gcn = items_emb_gcn.detach().cpu().numpy()
# users_emb_gcn = np.load('./gcnModel/user_emb_epoch79.npy',allow_pickle=True)
# items_emb_gcn = np.load('./gcnModel/item_emb_epoch79.npy',allow_pickle=True)
#u_sens = user_side_features['gender'].astype(np.int32)
user_age = user_side_features['age'].astype(np.int32)
u_age = []
for age in user_age:
    if age >= 35:
        u_age.append(1)
    else:
        u_age.append(0)
u_age = np.array(u_age)
u_sens =u_age
# u_sens=u_sens[:user_num]
class InforMax(nn.Module):
    def __init__(self, user_num, item_num, factor_num,  gcn_user_embs, gcn_item_embs,sens,samp,args):
        super(InforMax, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        three module: LineGCN, AvgReadout, Discriminator
        """
        # self.gcn = filter_layer.AttributeLineGCN(user_num, item_num, factor_num,users_features,items_features)
        # pdb.set_trace()
        self.gcn_users_embedding0 = torch.FloatTensor(gcn_user_embs).to(args.device)
        self.gcn_items_embedding0 = torch.FloatTensor(gcn_item_embs).to(args.device)
        self.user_num = user_num
        mean = 0
        std = 1
        noise = torch.normal(mean, std, size=(n_users, 64))
        self.noise = noise.to(args.device)
        # 将高斯噪声添加到向量上
        self.sigm = nn.Sigmoid()
        self.mse_loss = nn.MSELoss()
        self.model_d1 = SemiMLP(n_users, n_items, args.emb_size, args.device)
        #self.model_d1 = Discriminator(args.emb_size)

        self.model_f1 = AttributeFilter(factor_num, attribute='gender')
        self.CLloss = Constrastive_loss_limited_sen(n_users, args.tau, sens, samp, sample_ratio=0.1,lim_ratio=0.1)


    def forward_bpr(self,  user, pos, neg):
        # format of pos_seq or neg_seq:user_item_matrix,item_user_matrix,d_i_train,d_j_train
        gcn_users_embedding0 = self.gcn_users_embedding0
        gcn_items_embedding0 = self.gcn_items_embedding0

        # filter gender, age,occupation
        user_f1_tmp = self.model_f1(gcn_users_embedding0)

        user_f_tmp = user_f1_tmp
        # #local attribute
        item_f1_tmp = self.model_f1(gcn_items_embedding0)


        # L_R preference prediction loss.
        user_person_f = user_f1_tmp
        item_person_f = item_f1_tmp # gcn_items_embedding0#item_f2_tmp#gcn_items_embedding0#item_f2_tmp




        # user_b = F.embedding(user_batch,user_person_f)
        # item_b = F.embedding(item_batch,item_person_f)
        # prediction = (user_b * item_b).sum(dim=-1)
        # loss_part = self.mse_loss(prediction,rating_batch)
        # l2_regulization = 0.01*(user_b**2+item_b**2).sum(dim=-1)
        # l=len(att_label)
        # know_user_embed=user_person_f[:l]
        # unknow_user_embed = user_person_f[l:]
        # d_loss1=self.model_d1.forward(know_user_embed,att_label)
        # d_loss2=self.model_d1.hforward(unknow_user_embed)

        bpr_loss, emb_loss = self.calc_bpr_loss(user_person_f, item_person_f, user, pos, neg)
        # loss_part= -((prediction_i - prediction_j).sigmoid().log().mean())
        loss_p_square = bpr_loss + 0.01 * emb_loss

        # d_loss_all = 1 * (
        #             d_loss + 0.5 * d_loss_local)  # +1*d_loss_local #+1*d_loss1_local.cpu().numpy()    +0.5*d_loss_local
        # g_loss_all = 1 * loss_p_square - 0.3 * d_loss_all  # gender 0.3,age occ 0.1
        # g_d_loss_all = - 10 * d_loss_all
        # d_g_loss = [d_loss_all, g_loss_all, g_d_loss_all]
        #f_loss=0.1*loss_p_square+d_loss2+d_loss1

        return loss_p_square

    def forward_cl(self):
        gcn_users_embedding0 = self.gcn_users_embedding0
        # filter gender, age,occupation
        user_f1_tmp = self.model_f1(gcn_users_embedding0)
        vemb=user_f1_tmp+self.noise

        return self.model_d1(user_f1_tmp),self.model_d1(vemb)

    # Detach the return variables
    def embed(self):
        # h_pos: cat gcn_users_embedding and gcn_items_embedding, dim =0
        fliter_u_emb1 = self.model_f1(self.gcn_users_embedding0)

        fliter_i_emb1 = self.model_f1(self.gcn_items_embedding0)

        # fliter_i_emb = self.gcn_items_embedding0
        return fliter_u_emb1.detach(), fliter_i_emb1.detach()

    def save(self, fn):
        torch.save(self.state_dict(), fn)

    def load(self, fn):
        self.loaded = True
        self.load_state_dict(torch.load(fn))

    def bpr_loss(self, user_emb, pos_emb, neg_emb):
        pos_score = torch.sum(user_emb * pos_emb, dim=1)
        neg_score = torch.sum(user_emb * neg_emb, dim=1)
        mf_loss = torch.mean(F.softplus(neg_score - pos_score))
        emb_loss = (1 / 2) * (user_emb.norm(2).pow(2) +
                              pos_emb.norm(2).pow(2) +
                              neg_emb.norm(2).pow(2)) / user_emb.shape[0]
        return mf_loss, emb_loss

    def calc_bpr_loss(self, user_emb, item_emb, u, i, j):
        batch_user_emb = user_emb[u]
        batch_pos_item_emb = item_emb[i]
        batch_neg_item_emb = item_emb[j]

        mf_loss, emb_loss = self.bpr_loss(batch_user_emb, batch_pos_item_emb, batch_neg_item_emb)
        return mf_loss, emb_loss




dataset = BPRTrainLoader(train_set, train_u2i, n_items)
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)



######################################################## TRAINING #####################################
print('--------training processing-------')
count, best_hr = 0, 0
pretrain_anchor=np.load('age_anchor_emb_ml1m.npy',allow_pickle=True)
samp = su_pretrain_anchor_ml100k(n_users,n_items,u_sens,train_u2i,item_side_features,pretrain_anchor)
su=Collaborative_signal_completion(users_emb_gcn,u_sens,50,args,lim_ratio=0.1).get_su()
samp=0.6*su+0.4*samp##协同信息与侧信息融合
samp=torch.tensor(samp)
samp=samp.to(args.device)
model = InforMax(user_num, item_num, factor_num,  users_emb_gcn, items_emb_gcn,u_sens,samp,args)
model = model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # , betas=(0.5, 0.99))
# d1_optimizer = torch.optim.Adam(model.model_d1.parameters(), lr=0.005)
# f1_optimizer = torch.optim.Adam(model.model_f1.parameters(), lr=0.005)
# gcn_optimizer = torch.optim.Adam(model.gcn.parameters(), lr=0.005)
f_optimizer = torch.optim.Adam(list(model.model_f1.parameters()), lr=0.001)
d_optimizer = torch.optim.Adam(list(model.model_d1.parameters()), lr=0.003)


# f_optimizer = torch.optim.Adam(model.model_f1.parameters(),lr=0.001)
# d_optimizer = torch.optim.Adam(model.model_d1.parameters(),lr=0.001)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


user_feature_n=u_sens.reshape(n_users,1)
user_feature01 = np.eye(2)[u_sens].astype(int)
LACC = utils.metric.FairAndPrivacy(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                   user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)
att = 'gender'

for epoch in range(300):
    model.train()
    start_time = time.time()
    print('train data is  end')

    loss_current = [[], [], []]
    ##############训练判别器############
    for i in range(5):
        vector = list(range(n_users))  # 生成向量
        random.shuffle(vector)
        batch_num = (int)(n_users / batch_size + 1)
        sloss = 0
        for i in range(batch_num):
            if i == batch_num - 1:
                user_batch = vector[i * batch_size:]
            else:
                user_batch = vector[i * batch_size: (i + 1) * batch_size]
            uemb,vemb=model.forward_cl()
            # l2_regulization = 0.01 * (uemb ** 2 + vemb ** 2).sum()
            loss = model.CLloss.CLLOSS(uemb, vemb, user_batch)  # +l2_regulization
            d_optimizer.zero_grad()
            loss.backward()
            d_optimizer.step()
            loss_current[1].append(loss.item())
    ##############训练过滤器############
    for uij in train_loader:
        u = uij[0].type(torch.long).to(args.device)
        i = uij[1].type(torch.long).to(args.device)
        j = uij[2].type(torch.long).to(args.device)
        bpr = model.forward_bpr( u, i, j)
        loss_current[0].append(bpr.item())
        f_optimizer.zero_grad()
        bpr.backward()
        f_optimizer.step()
    ##############判别器-loss训练过滤器############
    for i in range(6):
        vector = list(range(n_users))  # 生成向量
        random.shuffle(vector)
        batch_num = (int)(n_users / batch_size + 1)
        sloss = 0
        for i in range(batch_num):
            if i == batch_num - 1:
                user_batch = vector[i * batch_size:]
            else:
                user_batch = vector[i * batch_size: (i + 1) * batch_size]
            uemb,vemb=model.forward_cl()
            # l2_regulization = 0.01 * (uemb ** 2 + vemb ** 2).sum()
            loss = -1*model.CLloss.CLLOSS(uemb, vemb, user_batch)  # +l2_regulization
            f_optimizer.zero_grad()
            loss.backward()
            f_optimizer.step()
            loss_current[2].append(loss.item())



    loss_current = np.array(loss_current, dtype=object)
    elapsed_time = time.time() - start_time
    # pdb.set_trace()
    train_loss_f = round(np.mean(loss_current[0]), 4)  #
    train_loss_f_d = round(np.mean(loss_current[2]), 4)  #
    # train_loss_f_g=round(train_loss_f+train_loss_f_d,4)#
    train_loss_d = round(np.mean(loss_current[1]), 4)  #
    str_print_train = "epoch:" + str(epoch) + ' time:' + str(
        round(elapsed_time, 1))  # +' train loss:'+str(train_loss)+'='+str(train_loss_part)+'+'

    str_d_g_str = ' loss'
    # str_d_g_str+=' f:'+str(train_loss_f)+'='+str(train_loss_f_g)+' - '+str(train_loss_f_d)
    str_d_g_str += ' f:' + str(train_loss_f) + 'fd:' + str(train_loss_f_d)
    str_d_g_str += '\td:' + str(train_loss_d)  #
    str_print_train += str_d_g_str  # '  d_1:'+str()
    print('--train--', elapsed_time)
    print(str_print_train)

    model.eval()

    f1_users_embedding, f1_i_emb= model.embed()
    user_e_f = f1_users_embedding.cpu().numpy()
    item_e_f = f1_i_emb.cpu().numpy()
    with torch.no_grad():

        #################推荐准确性指标################
        test_res = fair_evaluate.ranking_evaluate(
            user_emb=user_e_f,
            item_emb=item_e_f,
            n_users=n_users,
            n_items=n_items,
            train_u2i=train_u2i,
            test_u2i=test_u2i,
            sens=u_sens,
            num_workers=16)

        precision_Att, _ = LACC.get_Att_precision(user_e_f,
                                                  item_e_f, 'gender', 5)
        if precision_Att < 0.5:
            precision_Att = 1 - precision_Att
        precision_Att = round(precision_Att, 4)
        test_res['LACC@5-Gen'] = precision_Att
        precision_Att, _ = LACC.get_Att_precision(user_e_f,
                                                  item_e_f, 'gender', 10)
        if precision_Att < 0.5:
            precision_Att = 1 - precision_Att
        precision_Att = round(precision_Att, 4)
        test_res['LACC@10-Gen'] = precision_Att
        precision_Att, _ = LACC.get_Att_precision(user_e_f,
                                                  item_e_f, 'gender', 20)
        if precision_Att < 0.5:
            precision_Att = 1 - precision_Att
        precision_Att = round(precision_Att, 4)
        test_res['LACC@20-Gen'] = precision_Att


    ####AUC\F1

    auc_one, auc_res = pc_gender_train_ml100k.clf_gender_all_pre('auc', epoch, user_e_f,
                                                                 args.emb_size, args.device, user_feature01)
    test_res['Gen-AUC'] = round(np.mean(auc_one), 4)

    p_eval = ''
    for keys, values in test_res.items():
        p_eval += keys + ':' + '[%.6f]' % values + ' '
    print(p_eval)