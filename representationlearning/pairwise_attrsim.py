#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np

from torch import nn
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity

torch.set_default_tensor_type(torch.FloatTensor)

use_cuda = 1 >= 0 and torch.cuda.is_available()


class AttrClassifier(nn.Module):
    def __init__(self,input_size,hidden_size1,out_size):
        super(AttrClassifier, self).__init__()

        self.layer1 = nn.Linear(input_size,hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, out_size)

    def forward(self,x):

        h1 = F.relu(self.layer1(x))
        h = self.layer2(h1)

        return h


# get_label()

class SSL(nn.Module):
    def __init__(self,all_feature):
        super(SSL, self).__init__()
        self.feature = all_feature
        self.stackfeature = np.vstack((self.feature, self.feature))
        self.positive_index_array, self.negative_index_array, self.sims = self.get_label()
        self.selected_index = np.vstack((self.positive_index_array, self.negative_index_array))
        self.label = np.vstack(
            (np.ones([self.positive_index_array.shape[0], 1]), np.zeros((self.negative_index_array.shape[0], 1))))
        self.classifier = AttrClassifier(400,200,2)

    def get_label(self):

        print("calculating the similairty")
        sims = cosine_similarity(self.feature)
        print("finishing the similairty calculation")

        # index_array = np.loadtxt(r'attrsim_3_sampled_idx.txt')
        k = 3

        if True:
            positive_index_array = np.ones((sims.shape[0], k))
            negative_index_array = np.ones((sims.shape[0], k))
            sort_index = np.argsort(sims, axis=1)
            for line in range(positive_index_array.shape[0]):
                for col in range(2 * k):
                    if col < k:
                        negative_index_array[line, col] = int(sort_index[line, col])
                    else:
                        positive_index_array[line, col-3] = int(sort_index[line, -col + 1])

            # np.savetxt(f'attrsim_{k}_idx.txt', index_array)
            # # np.savetxt(r'attrsim.txt', sims)
        print("the end of get_label")

        return positive_index_array, negative_index_array, sims

    def forward(self):

        k = 50000
        sample_index_1 = np.random.choice(self.selected_index.shape[0], k, replace=False)
        label = self.label[sample_index_1]
        sample_index_2 = np.random.randint(0, self.selected_index.shape[1], k)

        embeddings0 = self.stackfeature[sample_index_1]
        another_index = self.selected_index[sample_index_1, sample_index_2].astype('int64')
        embeddings1 =self.stackfeature[another_index]
        embeddings = self.classifier(torch.abs(torch.tensor(embeddings0 - embeddings1)).to(torch.float32))

        output = F.log_softmax(embeddings,dim=1)
        loss = F.nll_loss(output,torch.tensor(label).squeeze().long())
        return loss


