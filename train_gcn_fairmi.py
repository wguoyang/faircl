# -*- coding: utf-8 -*-

import argparse
import copy
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
#import pandas as pd
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

import pdb
import sys


# def train_semigcn(gcn, sens, n_users, lr=0.001, num_epochs=1000, device='cpu'):
#     sens = torch.tensor(sens).to(torch.long).to(device)
#     optimizer = optim.Adam(gcn.parameters(), lr=lr)
#
#     final_loss = 0.0
#     for _ in tqdm(range(num_epochs)):
#         _, _, su, _ = gcn()
#         shuffle_idx = torch.randperm(n_users)
#         classify_loss = F.cross_entropy(su[shuffle_idx].squeeze(), sens[shuffle_idx].squeeze())
#         optimizer.zero_grad()
#         classify_loss.backward()
#         optimizer.step()
#         final_loss = classify_loss.item()
#
#     print('epoch: %d, classify_loss: %.6f' % (num_epochs, final_loss))
def pca_visual(embed,label,filename):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(embed)

    # 绘制可视化图像
    colors = ['r', 'b','y']  # 标签为0的用红色，标签为1的用蓝色
    for i in range(len(embed)):
         plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c=colors[label[i]])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()
def train_semigcn(model, useremb, sens, n_users, args,samp,lr=0.001, num_epochs=500, batch_size=500,device='cpu'):
    sens = torch.tensor(sens).to(torch.long).to(device)
    test_sens = copy.deepcopy(sens)
    l = int(0.6 * n_users)
    miu = torch.mean(samp)
    for i in range(l, n_users, 1):
        if test_sens[i] == 1:
            if samp[i] < miu:
                test_sens[i] = 2
        else:
            if samp[i] > miu:
                test_sens[i] = 2
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.001)
    # samp=samp.double()
    # s=samp*5
    # probability_distribution = F.softmax(s, dim=0)
    #CLloss=Constrastive_loss(n_users,args.tau,sens,samp,sample_ratio=0.1)
    CLloss=Constrastive_loss_limited_sen(n_users,args.tau,u_sens,samp,sample_ratio=0.1,lim_ratio=0.6)
    final_loss = 0.0
    # # 生成与向量相同大小的随机高斯噪声
    mean=0
    std=1
    noise = torch.normal(mean, std, size=(n_users, 64))
    noise=noise.to(args.device)
    # 将高斯噪声添加到向量上
    noisy_useremb = useremb + noise
    bestloss=9999
    #noisy_useremb = useremb.flip(dims=[1])
    for j in tqdm(range(num_epochs)):
        # samples = torch.multinomial(probability_distribution, num_samples=int(0.717*n_users), replacement=False)
        # label=torch.zeros(n_users,)
        # label[samples]=1
        vector = list(range(n_users))  # 生成向量
        random.shuffle(vector)
        batch_num=(int)(n_users/batch_size+1)
        sloss=0
        for i in range(batch_num):
            if i==batch_num-1:
                user_batch=vector[i*batch_size:]
            else:
                user_batch = vector[i * batch_size : (i+1) * batch_size]
            uemb=model(useremb)
            vemb=model(noisy_useremb)
            #l2_regulization = 0.01 * (uemb ** 2 + vemb ** 2).sum()
            loss=CLloss.CLLOSS(uemb,vemb,user_batch)#+l2_regulization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sloss = sloss+loss.item()
        print(sloss)
        final_loss=sloss
        if j!=0 and j%10==0:
            test_uemb = model(useremb)
            test_uemb=test_uemb.detach().cpu().numpy()
            filename = 'figure/ml1m/age_0.6/'+str(j)+'.pdf'
            pca_visual(test_uemb, test_sens,filename)
            filename_emb = 'figure/ml1m/age_0.6/' + str(j) + '.npy'
            np.save(filename_emb,test_uemb)
        #e_su = model.forward(useremb)
        ####可视化
        #e_su_np = e_su.detach().cpu().numpy()
        #data = np.random.randn(100, 10)  # 100个10维向量
        #labels = np.random.randint(0, 2, 100)  # 100个随机标签，0或1
        #pca_visual(e_su_np,sens)
        # if final_loss < bestloss:
        #     bestloss= final_loss
        #     torch.save(model, args.param_path)
        #     print('save successful')
    print('epoch: %d, classify_loss: %.6f' % (num_epochs, final_loss))
def train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args, LACC,adj,user_feature01):
    optimizer_G = optim.Adam(inter_enc.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(club.parameters(), lr=args.lr)

    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    #fairmi原来的敏感属性编码器
    # e_su, e_si, _, _ = sens_enc.forward()
    # e_su = e_su.detach().to(args.device)
    # e_si = e_si.detach().to(args.device)
    # p_su = conditional_samples(e_su.detach().cpu().numpy())
    # p_si = conditional_samples(e_si.detach().cpu().numpy())
    # p_su = torch.tensor(p_su).to(args.device)
    # p_si = torch.tensor(p_si).to(args.device)

    ex_enc = torch.load(args.pretrain_path)
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)


    sens_enc=sens_enc.to(args.device)
    e_su = sens_enc.forward(e_xu)
    e_su = e_su.detach().to(args.device)
    e_si=item_sens_emb(adj,copy.deepcopy(e_su),n_users,n_items,args)
    #e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    p_su = conditional_samples(e_su.detach().cpu().numpy())
    p_si = conditional_samples(e_si.detach().cpu().numpy())
    p_su = torch.tensor(p_su).to(args.device)
    p_si = torch.tensor(p_si).to(args.device)

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'lb': 0.0,
            'ub': 0.0,
            'mi': 0.0,
        }

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)
            main_user_emb, main_item_emb = inter_enc.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg

            e_zu, e_zi = inter_enc.forward()
            lb1 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
                                                   e_su[torch.unique(u)], p_su[torch.unique(u)])
            lb2 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
                                                   e_si[torch.unique(i)], p_si[torch.unique(i)])
            lb = args.lreg * (lb1 + lb2)
            # our further research found that imposing upper bound constraints on
            # the user-side only gives more stable and better results, so codes has been updated here.
            up = club.forward(e_zu[torch.unique(u)], e_su[torch.unique(u)])
            up = args.ureg * up
            loss = bpr_loss + emb_loss + lb + up
           
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['lb'] += lb.item()
            train_res['ub'] += up.item()

        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['lb'] = train_res['lb'] / len(train_loader)
        train_res['ub'] = train_res['ub'] / len(train_loader)

        e_zu, e_zi = inter_enc.forward()
        
        x_samples = e_zu.detach()
        y_samples = e_su.detach()

        for _ in range(args.train_step):
            mi_loss = club.learning_loss(x_samples, y_samples)
            optimizer_D.zero_grad()
            mi_loss.backward()
            optimizer_D.step()
            train_res['mi'] += mi_loss.item()
        train_res['mi'] = train_res['mi'] / args.train_step

        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = inter_enc.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens,
                num_workers=args.num_workers)

            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 5)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@5-Gen'] = precision_Att
            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 10)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@10-Gen'] = precision_Att
            precision_Att, _ = LACC.get_Att_precision(t_user_emb.detach().cpu().numpy(),
                                                      t_item_emb.detach().cpu().numpy(), 'gender', 20)
            if precision_Att < 0.5:
                precision_Att = 1 - precision_Att
            precision_Att = round(precision_Att, 4)
            test_res['LACC@20-Gen'] = precision_Att

        auc_one, auc_res = pc_gender_train.clf_gender_all_pre('auc', epoch, t_user_emb.detach().cpu().numpy(),
                                                              args.emb_size, args.device,user_feature01)
        test_res['Gen-AUC'] = round(np.mean(auc_one), 4)

        p_eval = ''
        for keys, values in test_res.items():
            p_eval += keys + ':' + '[%.6f]' % values + ' '
        print(p_eval)

            # if best_perf < test_res['ndcg@10']:
            #     best_perf = test_res['ndcg@10']
            #     torch.save(inter_enc, args.param_path)
            #     print('save successful')


if __name__ == '__main__':
    setproctitle.setproctitle('python')  # 更改当前进程名
    parser = argparse.ArgumentParser(
        description='ml_gcn_fairmi',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/gcn_fairmi.txt')
    parser.add_argument('--param_path', type=str, default='param/unlabel_semi_model_19.pth')
    parser.add_argument('--pretrain_path', type=str, default='param/gcn_base.pth')
    parser.add_argument('--lreg', type=float, default=0.1)
    parser.add_argument('--ureg', type=float, default=0.08)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--train_step', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    user_feature_n = np.load('./data/ml-1m/users_features_3num.npy', allow_pickle=True)
    user_age=user_feature_n[:,1]
    u_age=[]
    for age in user_age:
        if age>=3 :
            u_age.append(1)
        else:
            u_age.append(0)
    u_age=np.array(u_age)

    #############基于对比学习的敏感属性编码器################
    #u_sens = user_side_features['gender'].astype(np.int32)
    u_sens = u_age


    pretrain_anchor = np.load('age_anchor_emb_ml1m.npy', allow_pickle=True)
    #print_tiaodai(pretrain_anchor)
    samp = su_pretrain_anchor_ml1m(n_users, n_items, u_sens, train_u2i, 0, pretrain_anchor)
    print_function(u_sens,samp,n_users)
    pretrain_model = torch.load(args.pretrain_path,map_location=args.device)
    pretrain_useremb, pretrain_itememb = pretrain_model.forward()
    pretrain_useremb1=pretrain_useremb.detach().cpu().numpy()
    su=Collaborative_signal_completion(pretrain_useremb1,u_sens,100,args,lim_ratio=0.6).get_su()
    samp = 0.7 * su + 0.3 * samp
    samp = torch.tensor(samp)
    samp = samp.to(args.device)
    # att_label=get_att_predict(n_users,0.5,samp)
    # att_label=np.array(att_label).astype(np.int32)
    sens_enc = SemiMLP(n_users, n_items, args.emb_size, args.device)
    # pretrain_model = torch.load(args.pretrain_path)
    # pretrain_useremb,pretrain_itememb = pretrain_model.forward()
    # # model = torch.load(args.param_path,map_location='cuda:0')
    # # uemb=model(pretrain_useremb)
    # # uemb=uemb.detach().cpu().numpy()
    # # pca_visual(uemb,u_sens)
    pretrain_useremb2 = pretrain_useremb.detach().to(args.device)
    del pretrain_model
    del pretrain_itememb
    train_semigcn(sens_enc, pretrain_useremb2, u_sens, n_users, args, samp,device=args.device)
    ######################################


    #u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()
    #####
    user_feature_n = u_sens.reshape(n_users, 1)
    user_feature01 = np.eye(2)[u_sens].astype(int)
    LACC = utils.metric.FairAndPrivacy(usernum=n_users, itemnum=n_items, user_feature_n=user_feature_n,
                                       user_feature01=user_feature01, trainset=train_set, train_u2i=train_u2i)

    inter_enc = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)
    club = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)

    ####原始的敏感属性编码器#####
    # sens_enc = SemiGCN(n_users, n_items, norm_adj,
    #                    args.emb_size, args.n_layers, args.device,
    #                    nb_classes=2)
    # u_sens_noise=copy.deepcopy(u_sens)
    # p=0.6#百分之多少的用户属性是未知的
    # noise = [random.randint(0, 1) for _ in range(int(n_users*p))]
    # u_sens_noise[0:int(n_users*p)]=noise
    # train_semigcn(sens_enc, att_label, n_users, device=args.device)
    #############################################



    #train_unify_mi(sens_enc, inter_enc, club, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, args,LACC,norm_adj,user_feature01)

