# -*- coding: utf-8 -*-
import sys
import time
import itertools
import psutil
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from subprocess import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score,mean_squared_error

class myDataset(Dataset):
    def __init__(self, x, y):
        super(myDataset, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class myMlp(nn.Module):
    def __init__(self, in_len=1448, out_len=24,gpu_enable=False):
        super(myMlp, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(in_len, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Linear(1024,out_len)
        )
        self.gpu_enable = gpu_enable

    def forward(self,x):
        if self.gpu_enable ==False:
            x = x.cpu()
        x = self.net(x)
        return x

    def fit(self, x, y):
        x = torch.tensor(x,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.float32)
        dataset = myDataset(x, y)
        data_len = len(dataset)
        batchsize = 128
        dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True,num_workers=4)
        epochs = 100
        loss_fn = nn.MSELoss().to(self.device)
        optimizer = torch.optim.SGD(self.parameters(), lr=10e-5, momentum=0.9)
        self.net.to(self.device)
        for epoch in range(epochs):
            i = 0
            for samples, labels in dataloader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                out = self.forward(samples)
                loss = loss_fn(out, labels)
                r2 = r2_score(labels.cpu().detach().numpy(), out.cpu().detach().numpy())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                i = i + len(samples)
                print(
                    'epoch:{}/{}    iter:{}/{}    loss:{}   r2:{} '.format(epoch + 1, epochs, i, data_len,
                                                                  loss, r2))
from torch_geometric.nn import GINConv ,GCNConv,GATConv,GraphConv,FiLMConv
class GIN(nn.Module):
    def __init__(self,modelname, nfeat, dropout,n_layer=5, JK="last", residual=False):
        '''
        nfeat:输入x特征矩阵维度
       
        n_layer: GIN的层数
        dropout：dropout的比例
        '''
        super(GIN, self).__init__()
        self.num_layers = n_layer
        self.JK = JK
        # add residual connection or not
        self.residual = residual
        self.dropout = dropout
         # List of GNNs
         
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(n_layer):
            if modelname=='GIN':
                self.convs.append(GINConv(self.MLP(nfeat,nfeat)))
            elif modelname=='GCN':
                self.convs.append(GCNConv(nfeat,nfeat))
            elif modelname=='GAT':
                self.convs.append(GATConv(nfeat,nfeat))
            elif modelname=='GNN':
                self.convs.append(GraphConv(nfeat,nfeat))
                
            elif modelname=='Flim-GNN':
                self.convs.append(FiLMConv(nfeat,nfeat))
            
            self.batch_norms.append(torch.nn.BatchNorm1d(nfeat))
    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )
    def forward(self, x, adj):
        h_list=[x]
        
        for layer in range(self.num_layers):
            h = self.convs[layer](x, adj)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
    
            if self.residual:
                h += h_list[layer]
    
            h_list.append(h)

        
        return torch.stack(h_list[-self.num_layers:])






class myNetwork(nn.Module):
    def __init__(self, en_size,embed_size,nfeat,hidden_size,tar_len,segment_len,n_layers=2,dropout=0.02,modelname="GIN"):
        '''
        en_size 输入序列的长度
        embed_size  输入样本的特征维度
        nfeat  输入特征矩阵X的特征维度
        hidden_size 隐藏层的维度
        tar_len  输出序列的长度
        segment_len  shapelet的长度
        n_layers  GIN的层数
        '''
        super(myNetwork, self).__init__()
        self.gin_layers=n_layers
        self.k=tar_len//segment_len
        self.gIn = GIN(modelname,hidden_size, dropout,n_layer=n_layers)
       
        
        self.net = nn.Sequential(
            nn.Linear(segment_len, 512),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512*self.k,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,tar_len)
        )
        self.mlp1 =nn.Linear(nfeat,hidden_size)
        self.mlp2 =nn.Linear(self.gin_layers*hidden_size,hidden_size)
    def foroneGraph(self,x,adj):
        '''
        对于单个时间序列，单个图
                
        返回其对应的关键shapelet index
        '''
        a1=(adj >0).nonzero().t()
        
        x=F.relu(self.mlp1(x))#len,hidden_size
        h_=self.gIn(x,a1)#5,num_shapelets,hidden_size   h_[-1]为所有shapelets的表示
        # print('隐藏层的维度：',h_.size())#torch.Size([2, 30, 256])
        #node level Embedding
        node_E=x+h_[-1]


        #compute graph level EMbedding
       # Sum+CONCAT
        graph_Re=torch.sum(h_,dim=1).view(1,-1)#1, n_layer*hidden_size
        # print("%%%%%",graph_Re.size())
        
        graph_Re=self.mlp2(graph_Re)#1, hidden_size
        graph_Re=F.relu(graph_Re)#1, hidden_size
        dis=-torch.mm(graph_Re,node_E.t())#1,num_shapelets
        #取top 
        shapelet_index=torch.topk(dis,self.k,1)[1]#1,k
        return shapelet_index.squeeze(0)
        

    def forward(self,x,adj,embedding):
        '''
        x：batch_size,num_shapelets, nfeat
        adj: batch_size,num_shapelets, num_shapelets
        embedding  前面获取到的shapelet字典   num_shapelets,segments_length
        '''
        key_shapelets=[]#bacth_size,
        for o_x,o_adj in zip(x,adj):
            key_shapelets.append(self.foroneGraph(o_x,o_adj))
        key_shapelets=torch.stack(key_shapelets)   #bacth_size,k
        out=embedding(key_shapelets)  #bacth_size,k,segments_length
#        print("out.size",out.size())
        out=self.net(out)#bacth_size,tar_len
        return out
    
        







class ModelUtils(object):
    """
        model utils for basic classifiers.
        kwargs list:
            lr paras
                penalty: list of str, candidate: l1, l2;
                c: list of float
                inter_scale: list of float
            rf and dts paras:
                criteria: list of str, candidate: gini, entropy
                max_features: list of str(including None), candidate: auto, log2 or None
                max_depth: list of int
                max_split: list of int
                min_leaf: list of int
            xgb paras:
                max_depth: list of int
                learning_rate: list of float
                n_jobs: int
                class_weight: list of int
                booster: list of str, candidate: gblinear, gbtree, dart
            svm paras:
                c: list of float
                svm_kernel: list of str, candidate: rbf, poly, sigmoid
            deepwalk paras:
                num_walks: list of int
                representation_size: list of int
                window_size: list of int
                workers: int
                undirected: bool
    """
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.kwargs = kwargs

    @property
    def clf__(self):
        if self.kernel == 'lr':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression
        elif self.kernel == 'svm':
            from sklearn.svm import SVC
            return SVC
        elif self.kernel == 'dts':
            from sklearn.tree import DecisionTreeClassifier
            return DecisionTreeClassifier
        elif self.kernel == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier
        elif self.kernel == 'mlpReg':
            return myMlp
        
        # elif self.kernel == 'xgb':
        #     from xgboost import XGBClassifier
        #     return XGBClassifier
        else:
            raise NotImplementedError('unsupported kernel {}'.format(self.kernel))

    def para_len(self, balanced):
        cnt = 0
        for args in self.clf_paras(balanced=balanced):
            cnt += 1
        return cnt

    def clf_paras(self, balanced):
        class_weight = 'balanced' if balanced else None
        if self.kernel == 'lr':
            penalty = self.kwargs.get('penalty', ['l1', 'l2'])
            c = self.kwargs.get('c', [pow(5, i) for i in range(-3, 3)])
            intercept_scaling = self.kwargs.get('inter_scale', [pow(5, i) for i in range(-3, 3)])
            for (p1, p2, p3) in itertools.product(penalty, c, intercept_scaling):
                yield {
                    'penalty': p1,
                    'C': p2,
                    'intercept_scaling': p3,
                    'class_weight': class_weight
                }
        elif self.kernel == 'rf' or self.kernel == 'dts':
            criteria = self.kwargs.get('criteria', ['gini', 'entropy'])
            max_features = self.kwargs.get('max_feature', ['auto', 'log2',  None])
            max_depth = self.kwargs.get('max_depth', [10, 25, 50])
            min_samples_split = self.kwargs.get('max_split', [2, 4, 8])
            min_samples_leaf = self.kwargs.get('min_leaf', [1, 3, 5])
            for (p1, p2, p3, p4, p5) in itertools.product(
                    criteria, max_features, max_depth, min_samples_split, min_samples_leaf
            ):
                yield {
                    'criterion': p1,
                    'max_features': p2,
                    'max_depth': p3,
                    'min_samples_split': p4,
                    'min_samples_leaf': p5,
                    'class_weight': class_weight
                }
        elif self.kernel == 'xgb':
            max_depth = self.kwargs.get('max_depth', [1, 4, 8, 12])
            learning_rate = self.kwargs.get('learning_rate', [0.1, 0.2])
            n_jobs = [self.kwargs.get('n_jobs', psutil.cpu_count())]
            class_weight = self.kwargs.get('class_weight', [1, 10, 50])
            booster = self.kwargs.get('booster', ['gblinear', 'gbtree', 'dart'])
            n_estimators = self.kwargs.get('n_estimators', [10, 50, 100, 150])
            for (p1, p2, p3, p4, p5, p6) in itertools.product(
                    max_depth, learning_rate, booster, n_jobs, class_weight, n_estimators
            ):
                yield {
                    'max_depth': p1,
                    'learning_rate': p2,
                    'booster': p3,
                    'n_jobs': p4,
                    'scale_pos_weight': p5,
                    'n_estimators': p6
                }
        elif self.kernel == 'svm':
            c = self.kwargs.get('c', [pow(2, i) for i in range(-2, 2)])
            svm_kernel = self.kwargs.get('svm_kernel', ['rbf', 'poly', 'sigmoid'])
            for (p1, p2) in itertools.product(c, svm_kernel):
                yield {
                    'C': p1,
                    'kernel': p2,
                    'class_weight': class_weight
                    }
        else:
            raise NotImplementedError()

    @staticmethod
    def partition_data__(data, ratio, shuffle=True, multi=True):
        import random
        if not multi:
            size = len(data)
            if shuffle:
                idx = random.sample(range(size), int(size * ratio))
            else:
                idx, step, cnt, init = [], 1.0 / ratio, 0, 0
                while cnt < int(size * ratio):
                    idx.append(int(init))
                    init += step
            return data[idx]
        else:
            num, size = len(data), len(data[0])
            if shuffle:
                idx = random.sample(range(size), int(size * ratio))
            else:
                idx, step, cnt, init = [], 1.0 / ratio, 0, 0
                while cnt < int(size * ratio):
                    idx.append(int(init))
                    init += step
            return [data[k][idx] for k in range(num)]

    def deepwalk_paras(self):
        num_walks = self.kwargs.get('num_walks', [10, 20])
        representation_size = self.kwargs.get('representation_size', [32, 64, 128, 256])
        walk_length = self.kwargs.get('walk_length', [32, 64, 128])
        window_size = self.kwargs.get('window_size', [5, 10])
        workers = self.kwargs.get('workers', psutil.cpu_count())
        undirected = self.kwargs.get('undirected', False)
        for (p1, p2, p3, p4) in itertools.product(
                num_walks, representation_size, walk_length, window_size
        ):
            yield {
                'number-walks': p1,
                'representation-size': p2,
                'walk-length': p3,
                'window-size': p4,
                'workers': workers,
                'undirected': undirected
            }

    def return_metric_method(self, opt_metric):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        if opt_metric == 'accuracy':
            return accuracy_score
        elif opt_metric == 'precision':
            return precision_score
        elif opt_metric == 'recall':
            return recall_score
        elif opt_metric == 'f1':
            return f1_score
        else:
            raise NotImplementedError('unsupported metric {}'.format(opt_metric))

    def save_model(self, fpath):
        pass

    def load_model(self, fpath, map_location='cuda:0'):
        pass

    def save_shapelets(self, fpath):
        pass

    def load_shapelets(self, fpath, map_location='cuda:0'):
        pass


class Debugger(object):
    """
        Class for debugger print
    """
    def __init__(self):
        pass

    @staticmethod
    def error_print(msg, debug=True):
        if debug:
            print('[error]' + msg)

    @staticmethod
    def warn_print(msg, debug=True):
        if debug:
            print('[warning]' + msg)

    @staticmethod
    def debug_print(msg, debug=True):
        if debug:
            print('[debug]' + msg + '\r', end='')
            sys.stdout.flush()

    @staticmethod
    def info_print(msg):
        print('[info]' + msg)

    @staticmethod
    def time_print(msg, begin, profiling=False):
        if profiling:
            assert isinstance(begin, type(time.time())), 'invalid begin time {}'.format(begin)
            print('[info]{}, elapsed for {:.2f}s'.format(msg, time.time() - begin))


class Queue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def enqueue(self, val):
        if self.size() == self.max_size:
            self.dequeue()
        self.queue.insert(0, val)

    def dequeue(self):
        if self.is_empty():
            return None
        else:
            return self.queue.pop()

    def size(self):
        return len(self.queue)

    def is_empty(self):
        return self.size() == 0


def convert_string(string, val, cvt_type='float'):
    """
        Convert a string as given type.
    :param string:  input string
    :param val: default return value if conversion fails
    :param cvt_type: conversion type
    :return: value with given type
    """
    try:
        return eval(cvt_type)(string)
    except NameError as _:
        Debugger.warn_print('invalid convert type {}; use float() by default'.format(cvt_type))
        return float(string)
    except ValueError as _:
        Debugger.warn_print('invalid convert value {}; return {} by default'.format(string, val))
        return val


def syscmd(cmd, encoding=''):
    """
        Runs a command on the system, waits for the command to finish, and then
    returns the text output of the command. If the command produces no text
    output, the command's return code will be returned instead.

    :param cmd: command, str
    :param encoding: encoding method, str(utf8, unicode, etc)
    :return: return code or text output
    """
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
              stderr=STDOUT, close_fds=True)
    p.wait()
    output = p.stdout.read()
    if len(output) > 1:
        if encoding:
            return output.decode(encoding)
        else:
            return output
    return p.returncode


