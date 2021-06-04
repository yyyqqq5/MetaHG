#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from utils_jl import *
from metalearning.meta_update import Meta
import copy
from MetaLearningGetData import get_attention
from utils import get_train_data, get_test_data
from utils import get_metalearndata, print_performance
import torch

def train_meta(args,logits, labels, fs_label,neg_label):

    meta_trainfeat, meta_trainlabel, meta_testfeat, meta_testlabel = get_metalearndata(logits, labels, fs_label,
                                                                                       neg_label)
    maml = Meta(args)

    if args.att != 0:
        task_weight = get_attention([args.fs_label])
    else:
        task_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    for train_task in args.metatrainlabel:

        for j in range(args.metatrainepoch):  # 每一次只选一个task 进行训练  每个task 5个batch 数据  每个batch 10个samples 正负各5个

            x_spt, y_spt, x_qry, y_qry = get_train_data(meta_trainfeat, meta_trainlabel, train_task, args.k_spt,
                                                        args.k_qry, args.batch_num)

            accs, precision, recall, f1, acc = maml.forward(x_spt, y_spt, x_qry, y_qry, task_weight[train_task])

            print('Step:', j, '\tMeta_Training_Accuracy2:', acc)

    torch.save(maml.state_dict(), 'maml.prams')
    maml_copy = copy.deepcopy(maml)
    meta_test_acc = []
    model_meta_trained = Meta(args)
    model_meta_trained.load_state_dict(torch.load('maml.prams'))
    model_meta_trained.eval()  # 训练好的模型参数

    for k in range(args.metateststep):
        x_spt, y_spt, x_qry, y_qry = get_test_data(meta_testfeat, meta_testlabel, args.fs_label, args.k_spt, args.k_qry,
                                                   args.batch_num)
        with torch.no_grad():
            teacher_score = maml_copy.predict(x_spt)

        loss_meta, acc1, precision, recall, f1, acc2, p, r, F1, acc, TP_ave, TN_ave, FN_ave, FP_ave = model_meta_trained.forward_kd(
            x_spt, y_spt, x_qry, y_qry, teacher_score, kd=args.kd)

        meta_test_acc.append(accs)  # 每一个epoch 有5个batch 每个batch 有10个test samples 这里是每个epoch在在每次参数更新后的accs(一共10次更新)

        print_performance(k, p, r, F1, acc)

    return loss_meta