# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 16:07:22 2021

@author: Administrator
"""
import numpy as np
import pandas as pd
import random
import pickle
import os
from tqdm import tqdm
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
        train_seq =curent_df[i:i+tw][useful_column]#
        X.append(train_seq.values[i:i+src_len])
        Y.append(train_seq[i+src_len:]['Mean'].values)
        if i>L-tw and i<L:#处理尾巴上的
            train_seq =curent_df[-tw:][useful_column]
            X.append(train_seq.values[-tw:tw-src_len])
            Y.append(train_seq[tw-src_len:]['Mean'].values)
            
        if len(X)>1000:#控制内存
            X=X[-1000:]
            Y=Y[-1000:]
        return np.array(X),np.array(Y)

def get_dataset(inputdir,src_len,tar_len,step=5,train_probility=0.8,sample_pro=10000):
    
    if os.path.exists("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len)):
        train=pickle.load(open("train_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'rb'))#生成样本集
        valid=pickle.load(open("valid_numpy_samplePro%d_%d_%d.pkl"%(sample_pro,src_len,tar_len), 'rb'))
        print("数据：",train['X'].shape)
        return train['X'],train['Y'],valid['X'],valid['Y']
    else:
        pro_data=pd.read_csv(inputdir)
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