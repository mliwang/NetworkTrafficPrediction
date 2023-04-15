# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 18:12:54 2021

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:25:35 2021
用存好的训练数据测试数据测试模型
@author: Administrator
"""


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from os import path, cpu_count
import math
from tqdm import tqdm
import random
from time2graph.core.model_gin import Flow2Graph
from time2graph.utils.gat import GAT, accuracy_torch
from pathos.helpers import mp
import logging

from pathos.helpers import mp
import pickle
import warnings
warnings.filterwarnings("ignore")
from torch.nn import MSELoss
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#############model########################

def time2graphnet(K=30):
    '''
    K  生成 shapelets的数目
    seg_length  子片段的长度
    num_segment  一个序列分几段
    gpu_enable  用不用GPU
    optimizer    shapelets学习过程中使用什么优化方法
    device    使用的卡
    dropout  
    lk_relu
    data_size  时间序列的特征维度
    softmax  算距离的时候是否加softmax
    percentile 图构建中的距离阈值（百分位数）  weight小的p%不要
    dataset  数据集名字
    append 算特征的时候是否把片段自身加进去
    diff   是否求一阶方差
    
    '''
    general_options = {
        'init': 0,
        'warp': 2,
        'tflag': True,
        'mode': 'embedding',
        'candidate_method': 'greedy'
        
    }
    model = Flow2Graph(
        K, seg_length=10, num_segment=20, gpu_enable=False, optimizer='Adam', device=device, dropout=0.2, lk_relu=0.2, data_size=7, 
        softmax=False, percentile=10,dataset='Unspecified', append=False, sort=False, feat_flag=True,
        feat_norm=True, aggregate=True, standard_scale=False, diff=False,reg=True,**general_options
    )
    return model        
#########ourmodel###############
        
    
    
#########ourmodel ending###############

def getsingleGroup(pro_data,group,src_len,tar_len,step):
    '''
    pro_data 全部数据
    group 单组 key
    单组生成序列
    
    '''
    curent_df=pro_data.loc[(pro_data['hostname']==group[0]) & (pro_data['series']==group[1])]
    tw=src_len+tar_len#总的采样窗口大小，前面是X,后面部分的Mean是Y
    step=step
    X=[]
    Y=[]
    
    L=len(curent_df)
        #按时间排序
    curent_df['time'] = pd.to_datetime(curent_df['time_window'])
    curent_df.sort_values('time', inplace=True)
    useful_column=[ 'Mean', 'SD', 'Open', 'High','Low', 'Close', 'Volume']#取特征列
        
    for i in range(0,L-tw,step):
#                train_seq = df_tmp[features].values[i:i+tw]
        if i>L-tw:#处理尾巴上的
            train_seq =curent_df[-tw:][useful_column]
            X.append(train_seq.values[:-src_len])
            Y.append(train_seq[-src_len:]['Mean'].values)
            break
        train_seq =curent_df[i:i+tw][useful_column]#
        X.append(train_seq.values[:src_len])
        Y.append(train_seq[src_len:]['Mean'].values)
        
            
        if len(X)>100:#控制内存
            X=X[-50:]
            Y=Y[-50:]
            break
    return np.array(X),np.array(Y)

def get_dataset(inputdir,src_len,tar_len,step=5,train_probility=0.8,sample_pro=10000):
    
    if os.path.exists("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len)):
        train=pickle.load(open("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'rb'))#[:10000,:,:]#生成样本集
        valid=pickle.load(open("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'rb'))#[:1000]
        print("数据：",type(train['X']),train['Y'].shape)
        
        return train['X'][:sample_pro*3],train['Y'][:sample_pro*3],valid['X'][:int(sample_pro*0.1)],valid['Y'][:int(sample_pro*0.1)]
    else:
        pro_data=pd.read_csv(inputdir+'above1900_data.csv')
        all_sample=[]
        for k1,k2 in pro_data.groupby(by=['hostname','series']):
            all_sample.append(k1)
        all_sample=all_sample[:sample_pro]#少搞点试试
        random.shuffle(all_sample)   
        print('总采样点数：',len(all_sample))#19005
        train_all_sample=all_sample[:int(len(all_sample)*train_probility)]
        test_all_sample=list(filter(lambda x: x not in train_all_sample, all_sample))
        print('训练样本',len(train_all_sample),'测试样本:',len(test_all_sample))
        print('生成训练样本...')
        train_x,train_y=[],[]
        for id_ in tqdm(train_all_sample):
            x_i,y_i=getsingleGroup(pro_data,id_,src_len,tar_len,step)#一组样本
            train_x.extend(x_i)
            train_y.extend(y_i)
            
        with open("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'wb') as f:
            pickle.dump({'X':np.array(train_x),'Y':np.array(train_y)},f)
            
        print('生成测试样本...')
        valid_x,valid_y=[],[]
        for id_ in tqdm(test_all_sample):
            x_i,y_i=getsingleGroup(pro_data,id_,src_len,tar_len,step)#一组样本
            valid_x.extend(x_i)
            valid_y.extend(y_i)
            
        with open("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'wb') as f:
            pickle.dump({'X':np.array(valid_x),'Y':np.array(valid_y)},f)
        return np.array(train_x),np.array(train_y),np.array(valid_x),np.array(valid_y)

def get_label_forshapelets(y):
    '''
    if y_n+1-y_n >0 :
        label=1
    else:
        label=0
    y 10000,24
    '''
    #简单的使用差分，后续上升趋势为1，下降为0
    y2=((y[:,1]-y[:,0])>0).astype(int)
    return y2
 
    
def main(Istest):
    hidden_size = 512
    embed_size = 7#输入X的
    de_size=24*1#输出的序列长度也就是要预测未来多少个小时的
    en_size=200 #输入的序列长度,采样窗口的长度(采好样的那个就是200)
    epoch=10#训练轮数
    train_batch_size=512#训练集一批的大小
    K=70
    model = time2graphnet(K)
    inputdir='../'
    train_x,train_y,valid_x,valid_y_1=get_dataset(inputdir,en_size,de_size)#拿到所有的序列
    # print(np.isnan(train_x).all())
    # print(np.isnan(train_y).all())
    # print(np.isnan(valid_x).all())
    # print(np.isnan(valid_y).all())
    print(train_x.shape)
    #数据归一化
    train_x=(train_x-train_x.mean())/train_x.std()
    train_y=(train_y-train_y.mean())/train_y.std()
    #    #进行数据归一化处理
    valid_x=(valid_x-valid_x.mean())/valid_x.std()
    valid_y=(valid_y_1-valid_y_1.mean())/valid_y_1.std()
    all_X=np.vstack((train_x,valid_x))
    for_rescale=(valid_x.mean(),valid_x.std(),valid_y_1)
    print("“*****",for_rescale[0],for_rescale[1])
    model.data_size = embed_size
    shapelets_path = './cache/shapelets_%d_%d.cache'%(K,de_size/24)
    if path.isfile(shapelets_path):
        model.load_shapelets(shapelets_path)
        print('shapelets加载完成')
    else:
        print('开始提取shapelets ...')
        model.learn_shapelets(all_X, get_label_forshapelets(np.vstack((train_y,valid_y))), 20, 7)
        model.save_shapelets(shapelets_path)
        print('shapelets已保存')
    s=open('Flim_flow2graph_traing_log_shapelets_num%d_%d_%d.txt'% (K,en_size,de_size),'w')
    model.fit(for_rescale,train_x, train_y,valid_x,valid_y,get_label_forshapelets,epoch=epoch,train_batch_size=train_batch_size,de_size=de_size,logprintfile=s)
    s.close()
   
    logger.info("训练结束  "+"*"*20)

if __name__ == "__main__":
    main(False)
    
    
    
    
    
    