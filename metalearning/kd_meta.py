#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np
import argparse

from utils import get_metalearning_data,get_train_data,set_seed,get_test_data
from metalearning.meta_update import Meta
import copy
# from logger.log import logger
import time

def main(args):
    step = args.step
    set_seed(args.seed,0)
    device = torch.device("cuda:0" if args.gpu >= 0 else "cpu")

    dir = r'embedding_rgcn_gsl_ssl.txt'

    print("start to get data")


    features,labels,test_features,test_labels,train_node_num,test_node_num = get_metalearning_data(dir,args.metatrainlabel,args.metatestlabel,args.neg_label)
    features = features.to(device)
    labels = labels.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    print('finish getting data')


    config = [
        ('linear', [args.hidden, features.size(1)]),
        ('linear', [args.n_way, args.hidden])
    ]

    train_pair = [3,4,2]


    print("setup:Knowledge distillation: {}, train_epoch: {}, test_epoch:{}, spt_size:{}, qry_size:{}, meta-lr: {}, task_lr: {}, train_step:{}".format(
            args.kd,args.epoch,args.step,args.k_spt,args.k_qry,args.meta_lr,args.update_lr,args.update_step_test))


    # run multiple times by changing random seed
    for seed_epoch in range(50):
        st = time.time()
        set_seed(seed_epoch, 0)

        for i in range(len(args.metatestlabel)):

            test_label = [args.metatestlabel[i]]

            print('Train_Label_List: {} '.format(args.metatrainlabel))
            print('Test_Label_List: {} '.format(test_label))
            meta_train_acc,meta_train_prec,meta_train_recall,meta_train_f1 = [],[],[],[]

            maml = Meta(args).to(device)

            for pair_sub in train_pair:

                for j in range(args.epoch):

                    x_spt, y_spt, x_qry, y_qry = get_train_data(features, labels, pair_sub, args.k_spt,args.k_qry,args.batch_num)

                    acc, precision, recall, f1 = maml.forward(x_spt, y_spt, x_qry, y_qry)

                    meta_train_acc.append(acc)
                    meta_train_prec.append(precision)
                    meta_train_recall.append(recall)
                    meta_train_f1.append(f1)
                    print('pairs:',pair_sub)
                    print('Step:', j, '\tMeta_Training_Accuracy:', acc)

            torch.save(maml.state_dict(), 'maml.prams')
            maml_copy = copy.deepcopy(maml).to(device)
            meta_test_acc,meta_test_f1 = [],[]
            for test_sub in args.metatestlabel:
                test_pair = [test_sub] + [0]
                model_meta_trained = Meta(args).to(device)
                model_meta_trained.load_state_dict(torch.load('maml.prams'))
                model_meta_trained.eval()   # 训练好的模型参数展示
                for k in range(step):

                    x_spt, y_spt, x_qry, y_qry = get_test_data(test_features, test_labels, test_pair, args.k_spt, args.k_qry,1)

                    if args.kd == 1:
                        with torch.no_grad():
                            teacher_score = maml_copy.predict(x_qry)
                    else:
                        teacher_score = 0

                    acc,f1 = model_meta_trained.forward_kd(x_spt, y_spt, x_qry, y_qry,teacher_score,kd=args.kd)
                    print('Step:', k, '\tMeta_Testing_Accuracy:', acc)
                    meta_test_acc.append(acc)
                    meta_test_f1.append(f1)
                end = time.time()
                print("time {:05d} | Test Acc: {:.4f} |Train F1: {:.4f} | Time: {:.4f}".format(
                            seed_epoch, np.average(meta_test_acc), np.average(meta_test_f1),np.average(end-st)))
            del maml


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpu", type=int, default=1, help="gpu")
    argparser.add_argument('--epoch', type=int, help='training epoch number', default=10)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.05)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.08)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--batch_num', type=int, help='meta batch size', default=10)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=20)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=250)
    argparser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    # argparser.add_argument('--dataset', type=str, default='citeseer', help='Dataset to use.')
    argparser.add_argument('--kd', type=int, default=1, help='Use knowledge distillation')
    # argparser.add_argument('--att', type=int, default=0, help='Use task attention')
    argparser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
    argparser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    argparser.add_argument('--step', type=int, default=50, help='How many times to random select node to test')
    argparser.add_argument('--neg_label', type=int, default=[0], help='label of negative data')
    argparser.add_argument('--metatrainlabel', type=list, default=[2,3,4], help='meta train task labels')
    argparser.add_argument('--metatestlabel', type=list, default=[1,5], help='meta test task labels')

    args = argparser.parse_args()

    main(args)
