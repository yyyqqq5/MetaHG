#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
import time
import torch as th
import torch.nn.functional as F

from representationlearning.rgcn_model import RGCN
import torch
from representationlearning.pairwise_attrsim import SSL
from representationlearning.bg_gsl import StructureLearning, build_relation, build_graph
from utils import get_gcnlearndata,print_performance
from metalearning.meta_train import *

def main(args):

    device = torch.device("cuda:0" if args.gpu >= 0 else "cpu")

    post_vector_array = np.delete(np.loadtxt(args.postvectorpath), 0, 1)
    user_vector_array = np.delete(np.loadtxt(args.uservectorpath), 0, 1)


    gsl = StructureLearning(thresholdup = args.thresholdup,thresholduu=args.thresholduu,useridpath = args.useridpath,postidpath=args.postidpath)

    print("start training...")
    dur = []
    gsl.train()

    loss_meta = torch.tensor(0.0)

    relation_list, user_label = build_relation()

    for epoch in range(1,args.n_epochs):

        relation_list_copy = relation_list.copy()
        edge_uu, edge_up = gsl(user_vector_array, post_vector_array)
        relation_list_copy[1] += edge_uu
        relation_list_copy[2] += edge_up

        g, labels,edge_count = build_graph(relation_list_copy, user_label)

        features = g.ndata['h']
        all_features = features['user']

        labels = torch.tensor(labels)

        train_idx, val_idx, test_idx = get_gcnlearndata(labels, args.fs_label)

        rgcn = RGCN(g, args.n_hidden, out_dim=200, num_bases=args.n_bases,  num_hidden_layers=args.n_layers - 2, dropout=args.dropout,  use_self_loop=args.use_self_loop)
        ssl = SSL(all_features)

        optimizer = torch.optim.Adam(list(rgcn.parameters()) + list(gsl.parameters()) + list(ssl.parameters()), lr=args.lr,    weight_decay=5e-4)

        print("Epoch : {}".format(epoch))

        for g_epoch in range(0,args.gcn_epoch):
            t0 = time.time()

            logits = rgcn()[args.target_ent]

            loss = F.nll_loss(F.log_softmax(logits[train_idx], dim=1), labels[train_idx])


            if args.ssl:

                loss_ssl = ssl().to(torch.float32)
                loss_total = (args.ldargcn*loss + args.ldassl*loss_ssl + loss_meta+ args.ldagsl *edge_count ).to(torch.float32)
            else:
                loss_total = loss + loss_meta


            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            #     loss_total.backward(retain_graph=True)
            loss_total.backward(retain_graph=True)
            optimizer.step()
            t1 = time.time()

            dur.append(t1 - t0)

            train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
            train_f1 = F1(logits[train_idx], labels[train_idx])
            val_loss = F.nll_loss(F.log_softmax(logits[val_idx], dim=1), labels[val_idx])
            val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
            val_f1 = F1(logits[val_idx], labels[val_idx])

            print( "Epoch {:05d} | Train Acc: {:.4f} |Train F1: {:.4f} | Train Total Loss: {:.4f} |  Valid Acc: {:.4f} | Valid F1: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}". format(g_epoch, train_acc,train_f1, loss.item(), val_acc,val_f1, val_loss.item(), np.average(dur)))
        if args.model_path is not None:
            th.save(rgcn.state_dict(), args.model_path)

        embed = torch.cat([labels.unsqueeze(dim=1), logits], dim=1)
        embedding = embed.detach().numpy()

        np.savetxt(r'embedding_rgcn_gsl_ssl_0.98.txt', embedding, fmt="%.8e", delimiter=' ')

        rgcn.eval()
        logits_test = rgcn.forward()[args.target_ent]
        test_loss = F.nll_loss(F.log_softmax(logits_test[test_idx], dim=1), labels[test_idx])
        test_acc = th.sum(logits_test[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        test_f1 = F1(logits[test_idx], labels[test_idx])

        print("Test Acc: {:.4f} |Test F1: {:.4f} |Test loss: {:.4f}".format(test_acc, test_f1,test_loss.item()))
        print()


        for param in list(rgcn.parameters()) + list(gsl.parameters()) + list(ssl.parameters()):
            param.requires_grad = False

        del rgcn,gsl,ssl,optimizer
        loss_meta = train_meta(args, logits, labels, args.fs_label, args.neg_label)


        meta_trainfeat, meta_trainlabel, meta_testfeat, meta_testlabel = get_metalearndata(logits, labels, args.fs_label,args.neg_label)
        maml = Meta(args)
        optimizer_meta = torch.optim.Adam(maml.parameters(),lr=args.lr, weight_decay=5e-4)

        for train_task in args.metatrainlabel:

            for j in range(args.metatrainepoch):

                x_spt, y_spt, x_qry, y_qry = get_train_data(meta_trainfeat, meta_trainlabel, train_task, args.k_spt, args.k_qry, args.batch_num)

                accs, precision, recall, f1, acc = maml.forward(x_spt, y_spt, x_qry, y_qry,optimizer_meta)

                print('Step:', j, '\tMeta_Training_Accuracy:', acc)


        torch.save(maml.state_dict(), 'maml.prams')
        maml_copy = copy.deepcopy(maml)
        meta_test_acc = []
        model_meta_trained = Meta(args)
        model_meta_trained.load_state_dict(torch.load('maml.prams'))
        model_meta_trained.eval()

        for k in range(args.metateststep):

            x_spt, y_spt, x_qry, y_qry = get_test_data(meta_testfeat, meta_testlabel, args.fs_label, args.k_spt,args.k_qry, args.batch_num)
            with torch.no_grad():
                teacher_score = maml_copy.predict(x_spt)

            loss_meta, acc1, precision, recall, f1, acc2, p, r, F1, acc, TP_ave, TN_ave, FN_ave, FP_ave = model_meta_trained.forward_kd(x_spt, y_spt, x_qry, y_qry, teacher_score, kd=args.kd)


            meta_test_acc.append( accs)

            print_performance(k, F1, acc)

            print(meta_test_acc)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='kdMeta')

    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
                        help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("-e", "--n_epochs", type=int, default=10,
                        help="number of training epochs")

    ################ self-supervised learning#####################
    parser.add_argument('--ssl', default=True, help='whether add self-supervised learning')
    parser.add_argument('--ldassl', type=int, default=0.8, help='hyparameters of ssl loss')

    ################ graph structure learning#####################
    parser.add_argument('--ldagsl', type=int, default=0.0001, help='hyparameters of gsl loss')
    parser.add_argument('--thresholdup', type=float, default=0.98, help='similarity threshold between user and post')
    parser.add_argument('--thresholduu', type=float, default=0.98, help='similarity threshold between user and user')
    parser.add_argument('--useridpath', type=str, default='user_id_add.json', help='user-id match json')
    parser.add_argument('--postidpath', type=str, default='post_id_add.json', help='post-id match json')
    parser.add_argument('--keywordidpath', type=str, default='keyword_id_add_content.json',
                        help='keyword-id match json')
    parser.add_argument('--uservectorpath', type=str, default='userid_merged_vector_add.txt', help='user-id match json')
    parser.add_argument('--postvectorpath', type=str, default='postid_merged_vector_add.txt', help='post-id match json')
    parser.add_argument('--keywordvectorpath', type=str, default='keywordid_merged_vector_add.txt',
                        help='keyword-id match json')

    ################ gcn#####################
    parser.add_argument('--target_ent', type=str, default='user', help='user entity to train gcn')
    parser.add_argument('--gcn_epoch', type=int, default=100, help='gcn learning epochs')
    parser.add_argument("--model_path", type=str, default=None, help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0, help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=True, action='store_true',
                        help="include self feature as a special relation")
    parser.add_argument('--ldargcn', type=int, default=1, help='hyparameters of rgcn loss')


    ################ meta-learning#####################
    parser.add_argument('--metatrainepoch', type=int, help='train epoch for meta learning', default=20)
    parser.add_argument('--n_way', type=int, help='number of classification', default=2)
    parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.05)
    parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.08)
    parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    parser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    parser.add_argument('--batch_num', type=int, help='meta batch size', default=10)
    parser.add_argument('--k_spt', type=int, help='k shot for support set', default=20)
    parser.add_argument('--k_qry', type=int, help='k shot for query set', default=250)
    parser.add_argument('--hidden', type=int, help='Number of hidden units', default=16)
    parser.add_argument('--kd', type=int, default=0, help='Use knowledge distillation')
    parser.add_argument('--att', type=int, default=1, help='Use task attention or not (0 or 1)')
    parser.add_argument('--normalization', type=str, default='AugNormAdj', help='Normalization method for the adjacency matrix.')
    parser.add_argument('--metaseed', type=int, default=3, help='Random seed.')
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')
    parser.add_argument('--metateststep', type=int, default=50, help='How many times to random select node to test')
    parser.add_argument('--fs_label', type=int, default=1, help='label of few shot')
    parser.add_argument('--neg_label', type=int, default=0, help='label of negative data')
    parser.add_argument('--metatrainlabel', type=list, default=[2,3,4,5], help='meta train task labels')
    parser.add_argument('--embed_dim', type=int, default=200, help='node embedding dimension')


    args = parser.parse_args()
    print(args)
    main(args)
