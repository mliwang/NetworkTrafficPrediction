# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.nn.functional as F
from config import *
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from scipy.special import softmax
from .shapelet_utils import shapelet_distance, adjacent_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from .time_aware_shapelets import learn_time_aware_shapelets
from .static_shapelets import learn_static_shapelets
from .Optimize import AdamW, get_linear_schedule_with_warmup
from ..utils.base_utils import  myNetwork
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import gc
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
class myDataset(Dataset):
    def __init__(self, x,m, y=None,istrain=True):
        super(myDataset, self).__init__()
        self.x = torch.tensor(x,dtype=torch.float32)
        self.m = torch.tensor(m,dtype=torch.float32)
        if istrain:
            self.y =torch.tensor(y,dtype=torch.float32)
        self.istrain=istrain

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.istrain:
            return self.x[idx],self.m[idx], self.y[idx]
        else:
            return self.x[idx],self.m[idx]
def torch_R2(pre,label):
    '''
    pre  预测的流量序列  b,len
    label 实际的流量序列 b,len
    
    '''
    r2=0
    mse=0
    for i in range(len(pre)):
        RSS= torch.sum((pre-label)** 2)
        TSS=torch.sum((label-torch.mean(label))** 2)
        r2+=1-RSS/TSS
        mse+=torch.mean((pre-label)** 2)
        
    return mse/len(pre),r2/len(pre)

class Flow2Graph(nn.Module):
    """
        Time2GraphGAT model
        Hyper-parameters:
            K: number of learned shapelets
            C: number of candidates
            A: number of shapelets assigned to each segment
            tflag: timing flag
    """
    def __init__(self, K, seg_length, num_segment, warp=2, tflag=True, gpu_enable=False,device='cpu', optimizer='Adam', dropout=0.2, lk_relu=0.2, data_size=7, softmax=False,  percentile=10,
                 dataset='Unspecified', append=False, sort=False, 
                 feat_norm=True, aggregate=True, standard_scale=False, diff=False, **kwargs):
        super(Flow2Graph, self).__init__()
        self.K = K
        self.C = kwargs.pop('C', K * 10)
        self.seg_length = seg_length
        self.num_segment = num_segment
        self.data_size = data_size
        self.device=device
        self.warp = warp
        self.tflag = tflag
        self.gpu_enable = gpu_enable
        self.cuda = self.gpu_enable and torch.cuda.is_available()
        # Debugger.info_print('torch.cuda: {}, self.cuda: {}'.format(torch.cuda.is_available(), self.cuda))
        self.shapelets = None
        self.append = append
        self.percentile = percentile
        self.threshold = None
        self.clf=None
        self.sort = sort
        self.aggregate = aggregate
        self.dropout = dropout
        self.lk_relu = lk_relu
        self.softmax = softmax
        self.dataset = dataset
        self.diff = diff
        self.standard_scale = standard_scale
        
        self.feat_norm = feat_norm
        self.pretrain = kwargs.pop('pretrain', None)

        self.lr = kwargs.pop('lr', 1e-3)
        self.p = kwargs.pop('p', 2)
        self.alpha = kwargs.pop('alpha', 0.1)
        self.beta = kwargs.pop('beta', 0.05)
        self.debug = kwargs.pop('debug', False)
        self.optimizer = optimizer
        self.measurement = kwargs.pop('measurement', 'gdtw')
        self.batch_size = kwargs.pop('batch_size', 200)
        self.init = kwargs.pop('init', 0)
        self.niter = kwargs.pop('niter', 1000)
        self.fastmode = kwargs.pop('fastmode', False)
        self.tol = kwargs.pop('tol', 1e-4)
        self.cuda = self.gpu_enable and torch.cuda.is_available()
        self.kwargs = kwargs
        Debugger.info_print('initialize our Flow2Graph with {}'.format(self.__dict__))

    def learn_shapelets(self, x, y, num_segment, data_size):
        assert x.shape[1] == num_segment * self.seg_length
        Debugger.info_print('basic statistics before learn shapelets: max {:.4f}, min {:.4f}'.format(np.max(x), np.min(x)))
        if self.tflag:
            self.shapelets = learn_time_aware_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, p=self.p,
                num_segment=num_segment, seg_length=self.seg_length, data_size=data_size,
                lr=self.lr, alpha=self.alpha, beta=self.beta, num_batch=int(x.shape[0] / self.batch_size),
                measurement=self.measurement, gpu_enable=self.gpu_enable, **self.kwargs)
        else:
            self.shapelets = learn_static_shapelets(
                time_series_set=x, label=y, K=self.K, C=self.C, warp=self.warp,
                num_segment=num_segment, seg_length=self.seg_length, measurement=self.measurement, **self.kwargs)

    def __gat_features__(self, X, train=False):
        __shapelet_distance = shapelet_distance(
            time_series_set=X, shapelets=self.shapelets, seg_length=self.seg_length,
            tflag=self.tflag, tanh=self.kwargs.get('tanh', False), debug=self.debug,
            init=self.init, warp=self.warp, measurement=self.measurement)
        threshold = None if train else self.threshold
        adj_matrix, self.threshold = adjacent_matrix(
            sdist=__shapelet_distance, num_time_series=X.shape[0], num_segment=int(X.shape[1] / self.seg_length),
            num_shapelet=self.K, percentile=self.percentile, threshold=threshold, debug=self.debug)
        __shapelet_distance = np.transpose(__shapelet_distance, axes=(0, 2, 1))
        if self.sort:
            __shapelet_distance = softmax(-1 * np.sort(__shapelet_distance, axis=1), axis=1)
        if self.softmax and not self.sort:
            __shapelet_distance = softmax(__shapelet_distance, axis=1)
        if self.append:
            origin = np.array([v[0].reshape(-1) for v in self.shapelets], dtype=np.float).reshape(1, self.K, -1)
            return np.concatenate((__shapelet_distance, np.tile(origin, (__shapelet_distance.shape[0], 1, 1))),
                                  axis=2).astype(np.float), adj_matrix
        else:
            return __shapelet_distance.astype(np.float), adj_matrix


    def __preprocess_input_data(self, X):
        X_scale = X.copy()
        if self.diff:
            X_scale[:, : -1, :] = X[:, 1:, :] - X[:, :-1, :]
            X_scale[:, -1, :] = 0
            Debugger.debug_print('conduct time differing...')
        if self.standard_scale:
            for i in range(self.data_size):
                X_std = np.std(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)
                X_std[X_std == 0] = 1.0
                X_scale[:, :, i] = (X_scale[:, :, i] - np.mean(X_scale[:, :, i], axis=1).reshape(X.shape[0], -1)) / X_std
                Debugger.debug_print('conduct standard scaling on data-{}, with mean {:.2f} and var {:.2f}'.format(i, np.mean(X_scale[0, :, i]), np.std(X_scale[0, :, i])))
        return X_scale
    def transfer(self,sps):
        '''
        把shapelets的列表转换成矩阵
        '''
        ss=np.zeros((len(sps),self.seg_length))
        for idx, (pattern, _, _, _) in enumerate(sps):
            ss[idx]=pattern[:,0]#取均值那一列
        return nn.Embedding.from_pretrained(torch.tensor(ss,dtype=torch.float32))#  num_shapelets,seg_length
    def fit(self,for_rescale, X_scale, Y,valid_x_scale,valid_y,clf_func, reset=False,train_batch_size=256,de_size=24*1,epoch=100,
          display_steps=2,eval_steps=2,max_grad_norm=1.0,lr=0.3,l2_alpha=0.01,hidden_size=512,output_dir='model',logprintfile=None,):
        '''
        X_scale,  
        Y,
        valid_x_scale,
        valid_y
        clf_func :获取分类标签的函数
        '''
        num_segment, data_size = int(X_scale.shape[1] / self.seg_length), X_scale.shape[-1]
        assert self.data_size == X_scale.shape[-1]
        X_scale = self.__preprocess_input_data(X_scale)#归一化 啥也没干其实
        valid_x_scale=self.__preprocess_input_data(valid_x_scale)#归一化  这里也是啥也没干其实

        if reset or self.shapelets is None:
            self.learn_shapelets(x=np.vstack((X_scale,valid_x_scale)), y=clf_func(np.vstack((Y,valid_y))), num_segment=num_segment, data_size=data_size)
#        self.__fit_gat(X=X_scale, Y=Y)
        print("获取数据的特征表示:")
        import pickle
        if os.path.exists('feauture_%d_%d.plk'%(self.K,de_size/24)):
             X_savel=pickle.load(open('feauture_%d_%d.plk'%(self.K,de_size/24), 'rb'))
             X_feat, X_adj =X_savel['X_feat'],X_savel['X_adj']
             del X_savel
             gc.collect()
        else:
            X_feat, X_adj = self.__gat_features__(X_scale)#获得初始化的节点特征和邻接矩阵
            with open('feauture_%d_%d.plk'%(self.K,de_size/24), 'wb') as f:
                pickle.dump({'X_feat':np.array(X_feat),'X_adj':X_adj},f)
        
        print("开始训练！")
        #开始训练
        dataset = myDataset(X_feat, X_adj,Y)
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=train_batch_size,num_workers=0)
        loss_fn = nn.MSELoss().to(self.device)
        en_size=X_scale.shape[1]
        nfeat=X_feat.shape[-1]#shapelets初始维度
        tar_len=Y.shape[-1]#预测目标的长度
        
        self.clf=myNetwork(en_size,data_size,nfeat,hidden_size,tar_len,self.seg_length,n_layers=2,dropout=0.1,modelname="Flim-GNN")
        '''
        modelname:使用的模型名称
        nfeat:输入x特征矩阵维度
        nhid：中间层维度
        nclass：输出特征维度
        dropout：dropout的比例
        '''
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        #设置优化器
        
        # optimizer = AdamW(self.parameters(), lr=lr, eps=1e-8,weight_decay=0.00001)
        # optimizer =torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader)*epoch*0.2),num_training_steps=int(len(train_dataloader)*epoch))    
        
        self.clf.to(self.device)
        all_shapelets=self.transfer(self.shapelets)#拿到所有shapelet id到真实片段的映射
        self.clf.zero_grad() 
        tr_loss,best_R2,avg_loss = 0.0, -10,0.0
        global_step=0
        for idx in range(epoch):     
            tr_num=0
            train_loss=0
            self.clf.train()
            for step, batch in enumerate(train_dataloader):
                
                feat,adj,Y =(x.to(self.device) for x in batch)
                out =self.clf(feat,adj,all_shapelets.to(self.device))#state_orginal ,b,n,2
#                print(out.device)
                del batch,feat,adj
                gc.collect()
                loss =loss_fn(out,Y)
                optimizer.zero_grad()
    
                loss.backward()#先写到这里，后续再补充！！
                torch.nn.utils.clip_grad_norm_(self.parameters(),max_grad_norm)         
                tr_loss += loss.item()
                tr_num+=1
                train_loss+=loss.item()
                #输出log
                if avg_loss==0:
                    avg_loss=tr_loss
                avg_loss=round(train_loss/tr_num,8)
                
                if (step+1) % display_steps == 0:
                    Debugger.info_print("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss))
                    print("At Training:  epoch {} step {} loss {}".format(idx,step+1,avg_loss),"\n",file=logprintfile)
               
                #update梯度
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()  
                global_step += 1
                
                #测试验证结果
                if (step+1) % eval_steps == 0:
                    #输出验证集预测的结果
                    out= self.infer(self.clf,valid_x_scale)
                    #输出预测的f1和error distance
                    results=self.eval(out,torch.tensor(valid_y,dtype=torch.float32).to(self.device))
                    with open('flow2graph_casestudy.pkl','wb')as fff:
                        pickle.dump({'Y_label':for_rescale[2],'Y_pre':out.cpu().numpy()},fff)
                        # pickle.dump({'Y_label':valid_y*for_rescale[1]+for_rescale[0],'Y_pre':out.cpu().numpy()*for_rescale[1]+for_rescale[0]},fff)    
                    
                    #打印结果                  
                    for key, value in results.items():
                        logger.info("测试结果  %s = %s", key, round(value,8))      
                    #保存最好的年龄结果和模型
                    if results['eval_R2']>best_R2:
                        best_R2=results['eval_R2']
                        print("  "+"*"*20)  
                        print("  "+"*"*20,"\n",file=logprintfile)
                        for key, value in results.items():
                            logger.info("测试结果  %s = %s", key, round(value,8))      
                            print("测试结果  {} = {}".format(key, round(value,8)),"\n",file=logprintfile)
                        logger.info("  Best R2:%s",round(best_R2,8))
                        logger.info("  Best mse:%s",round(results['eval_loss'],8))
                        print("  Best f1:",round(best_R2,8),"  Best MSE:",round(results['eval_loss'],8),"\n",file=logprintfile)
                        print("  "+"*"*20,"\n",file=logprintfile)
                        logger.info("  "+"*"*20)                          
                        
                        model_to_save = self.clf.module if hasattr(self.clf, 'module') else self.clf  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_time2graph_gcn_{}_{}.bin".format(en_size,tar_len))
                        torch.save(model_to_save.state_dict(), output_model_file)
            print("  Best R2:",round(best_R2,8),"  Best MSE:",round(results['eval_loss'],8))

    def infer(self,model,valid_x_scale,eval_batch_size=32):
        assert self.shapelets is not None, 'shapelets has not been learnt yet...'
        X_feat, X_adj = self.__gat_features__(valid_x_scale)#获得初始化的节点特征和邻接矩阵
        eval_dataset=myDataset(X_feat, X_adj,istrain=False)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_batch_size,num_workers=4)
        all_shapelets=self.transfer(self.shapelets)#拿到所有shapelet id到真实片段的映射
        predict=[]
        for step, batch in enumerate(eval_dataloader):
            feat,adj =(x.to(self.device) for x in batch)
            del batch
            with torch.no_grad():
                pred  =model(feat,adj,all_shapelets.to(self.device))#b,tar_len
#                predict.append(pred.cpu().numpy())# b,tar_len
                predict.append(pred)
            del feat,adj,pred
            gc.collect()
#        predict=np.concatenate(predict,0) 
        predict=torch.cat(predict,dim=0)#torch.stack(predict)
        return predict
    def eval(self,predict,Groudth):
        '''
        predict  sample_len ,de_size,1
        Groudth sample_len ,de_size,1
        '''
        results={}
        m,r2=torch_R2(predict,Groudth)
        results['eval_loss']=m.cpu().item()
        results['eval_R2']=r2.cpu().item()
#        from sklearn.metrics import r2_score,mean_squared_error
#        results={}
#        results['eval_R2']=r2_score(Groudth,predict)
#       
#        results['eval_loss']=mean_squared_error(Groudth,predict)         
        return results
        
    def reload(self,model,output_dir,en_size,tar_len):
            #读取在验证集结果最好的模型
        load_model_path=os.path.join(output_dir, "pytorch_time2graph_gcn_{}_{}.bin".format(en_size,tar_len))
        logger.info("Load model from %s",load_model_path)
        model_to_load = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_load.load_state_dict(torch.load(load_model_path))   
        return model    

    def save_shapelets(self, fpath):
        torch.save(self.shapelets, fpath)

    def load_shapelets(self, fpath, map_location='cuda:0'):
        self.shapelets = torch.load(fpath, map_location=map_location)
