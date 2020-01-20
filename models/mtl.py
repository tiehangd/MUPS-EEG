##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Modified from: https://github.com/yaoyao-liu/meta-transfer-learning
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
""" Meta Learner """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.feature_extractor import FeatureExtractor

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.softmax(F.linear(input_x, fc1_w, fc1_b), dim=1)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=4):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 4*2*25
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = FeatureExtractor()  
        else:
            self.encoder = FeatureExtractor(mtl=False)  
            self.pre_fc = nn.Sequential(nn.Linear(4*2*25, num_cls))

    def forward(self, inp):
        if self.mode=='pre' or self.mode=='origval':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward2(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        return F.softmax(self.pre_fc(self.encoder(inp)), dim=1)    

    def preval_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(2):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.005 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q

    def meta_forward(self, data_shot, label_shot, data_query):
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)        
        return logits_q

    def meta_forward2(self, data_shot, label_shot, data_query):
        embedding_shot=self.encoder(data_shot)
        embedding_query = self.encoder(data_query)
        params=self.base_learner.parameters()
        optimizer=optim.Adam(params)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        loss.backward(retain_graph=True)
        optimizer.step()
        logits_q = self.base_learner(embedding_query)

        for _ in range(10):
            optimizer.zero_grad()
            logits = self.base_learner(embedding_shot)
            loss = F.cross_entropy(logits, label_shot)
            loss.backward(retain_graph=True)
            optimizer.step()
            logits_q = self.base_learner(embedding_query)

        return logits_q



