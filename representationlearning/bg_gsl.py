#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import json
import dgl
import torch
import torch.nn as nn

import numpy as np


torch.set_default_tensor_type(torch.FloatTensor)


def extra_same_elem(lst, *lsts):
    iset = set(lst)
    for li in lsts:
        s = set(li)
        iset = iset.intersection(s)
    return list(iset)


def build_relation():

    fp = open('post_id_add.json','r')
    post_id_dict = json.load(fp)

    fu = open('user_id_add.json', 'r')
    name_id_dict = json.load(fu)


    fk = open('keyword_id_add_content.json', 'r')
    keyword_id_dict = json.load(fk)


    fc = open('userid_categ_add.json', 'r')
    userid_cate_dict = json.load(fc)


    similar_relation,similarity_relation,alike_relation,comment_relation,tagger_relation,have_relation,mention_relation,reply_relation,include_relation,tag_relation,profile_relation = [],[],[],[],[],[],[],[],[],[],[]
    relation_dir = 'relation_add_content.txt'

    fr = open(relation_dir, 'r', encoding='utf8')
    for line in fr.readlines():
        parts = line.replace('\n','').split('\t')
        relation_type = parts[2]

        try:
            if relation_type == 'comment':
                comment_relation.append((name_id_dict[parts[0]],name_id_dict[parts[1]]))
            elif relation_type == 'similarity':
                similarity_relation.append((name_id_dict[parts[0]],name_id_dict[parts[1]]))
            elif relation_type == 'similar':
                similar_relation.append((name_id_dict[parts[0]],post_id_dict[parts[1]]))
            elif relation_type == 'alike':
                alike_relation.append((post_id_dict[parts[0]],post_id_dict[parts[1]]))
            elif relation_type == 'tagger':
                tagger_relation.append((name_id_dict[parts[0]],name_id_dict[parts[1]]))
            elif relation_type == 'have':
                have_relation.append((name_id_dict[parts[0]],post_id_dict[parts[1]]))
            elif relation_type == 'mention':
                mention_relation.append((name_id_dict[parts[0]], post_id_dict[parts[1]]))
            elif relation_type == 'reply':
                reply_relation.append((name_id_dict[parts[0]], post_id_dict[parts[1]]))
            elif relation_type == 'include':
                include_relation.append((post_id_dict[parts[0]], keyword_id_dict[parts[1]]))
            elif relation_type == 'tag':
                tag_relation.append((post_id_dict[parts[0]], keyword_id_dict[parts[1]]))
            elif relation_type == 'profile':
                profile_relation.append((name_id_dict[parts[0]], keyword_id_dict[parts[1]]))
        except:
            continue
    relation_list = [comment_relation, similarity_relation, similar_relation, alike_relation, tagger_relation, have_relation, mention_relation, reply_relation, include_relation, tag_relation, profile_relation]
    user_label = list(userid_cate_dict.values())

    return relation_list,user_label

def build_graph(relation_list,user_label):
    g = dgl.heterograph({
        ('user', 'comment', 'user'): relation_list[0],
        ('user', 'similarity', 'user'):relation_list[1],
        ('user', 'similar', 'post'): relation_list[2],
        ('post', 'alike', 'post'): relation_list[3],
        ('user', 'tagger', 'user'):relation_list[4],
    ('user', 'have', 'post'):relation_list[5],
    ('user', 'mention', 'post'):relation_list[6],
    ('user', 'reply', 'post'):relation_list[7],
    ('post', 'include', 'keyword'):relation_list[8],
    ('post', 'tag', 'keyword'):relation_list[9],
    ('user', 'profile', 'keyword'):relation_list[10],
    })


    post_vector_array = np.delete(np.loadtxt('postid_merged_vector_add.txt'), 0, 1)

    user_vector_array = np.delete(np.loadtxt('userid_merged_vector_add.txt'), 0, 1)

    keyword_vector_array = np.delete(np.loadtxt('keywordid_merged_vector_add.txt'), 0, 1)

    g.nodes['user'].data['h'] = torch.tensor(user_vector_array)
    g.nodes['post'].data['h'] = torch.tensor(post_vector_array)
    g.nodes['keyword'].data['h'] = torch.tensor(keyword_vector_array)

    print("node statistics:")
    print("# of posts:")
    print(g.number_of_nodes('post'))
    print("# of user:")
    print(g.number_of_nodes('user'))
    print("# of keyword:")
    print(g.number_of_nodes('keyword'))
    print("edge statistics:")
    print("# of comment relation:")
    print(g.number_of_edges(('user', 'comment', 'user')))
    print("# of tagger relation:")
    print(g.number_of_edges(('user', 'tagger', 'user')))
    print("# of have relation:")
    print(g.number_of_edges(('user', 'have', 'post')))
    print("# of simiarity relation:")
    try:
        print(g.number_of_edges(('user', 'similarity', 'user')))
    except:
        print(0)
    print("# of similar relation:")
    try:
        print(g.number_of_edges(('user', 'similar', 'post')))
    except:
        print(0)
    print("# of alike relation:")
    try:
        print(g.number_of_edges(('post', 'alike', 'post')))
    except:
        print(0)
    print("# of mention relation:")
    print(g.number_of_edges(('user', 'mention', 'post')))
    print("# of reply relation:")
    print(g.number_of_edges(('user', 'reply', 'post')))
    print("# of include relation:")
    print(g.number_of_edges(('post', 'include', 'keyword')))
    print("# of tag relation:")
    print(g.number_of_edges(('post', 'tag', 'keyword')))
    print("# of profile relation:")
    print(g.number_of_edges(('user', 'profile', 'keyword')))

    print(g.successors(588,etype='include'))
    print(g.canonical_etypes)
    print(g.etypes)

    edge_count = 0
    for item in g.canonical_etypes:
        edge_count += g.number_of_edges(item)

    return g,user_label,edge_count


class EmbeddingWeight(nn.Module):
    def __init__(self,int_feature,out_feature):
        super(EmbeddingWeight,self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(int_feature,out_feature))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self,h):

        return h*self.weight

def cos_sim(a, b, eps=1e-8):
    """
    calculate cosine similarity between matrix a and b
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

class StructureLearning(nn.Module):
    def __init__(self,thresholdup,thresholduu,useridpath,postidpath,duser,dpost,dvector):
        super(StructureLearning,self).__init__()
        self.embed1 = EmbeddingWeight(duser,dvector)
        self.embed2 = EmbeddingWeight(dpost, dvector)
        self.threshold_up = thresholdup
        self.threshold_uu = thresholduu
        self.merge_entity = list(json.load(open(useridpath, 'r')).keys())+list(json.load(open(postidpath, 'r')).keys())
        self.add_relation_up,self.add_relation_uu = [],[]

    def forward(self,attr1,attr2):

        torch_a = self.embed1(torch.from_numpy(attr1))
        torch_b = self.embed2(torch.from_numpy(attr2))

        s = cos_sim(torch_a, torch_b).float()
        similarity = torch.where(s < self.threshold_up, torch.zeros_like(s), s)
        edge_index = torch.nonzero(similarity,as_tuple=False)

        for irow in range(edge_index.shape[0]):
                self.add_relation_up.append([edge_index[irow,0],edge_index[irow,1]])

        s_user = cos_sim(torch_a, torch_a).float()
        similarity_user = torch.where(s_user < self.threshold_uu, torch.zeros_like(s_user), s_user)
        edge_index_user = torch.nonzero(similarity_user, as_tuple=False)

        for irow in range(edge_index_user.shape[0]):
            self.add_relation_uu.append(
                [edge_index_user[irow, 0], edge_index_user[irow, 1]])

        return self.add_relation_uu,self.add_relation_up





