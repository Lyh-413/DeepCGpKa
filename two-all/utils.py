import math
import os, shutil
from tqdm import tqdm, trange
from inspect import isfunction
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator
import constants as constants

def delmkdir(path):
    '''
    remove file and create a new one
    '''
    isexist = os.path.exists(path)
    if isexist == True : 
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def scheduler_lr(step):
    if step <= 2:
        return 1.0
    if step > 2 and step <= 7:
        return 1.0
    if step > 7 and step <= 15:
        return 1.0
    if step > 15 and step <= 25:
        return 1.0
    if step > 25 and step <= 50:
        return 0.5
    if step > 50 and step <= 75:
        return 0.2
    if step > 75 and step <=100:
        return 0.1
    if step>100:
        return 0.05
    if step>150:
        return 0.02
    if step>200:
        return 0.01

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class CGDataSet(Dataset):
    def __init__(self, data_type):
        assert data_type in ['train', 'val', 'test','test2','testall','valall']
        self.dataset_path = constants.DATA_DIR
        self.norm_dic = None
        self.resi_type = constants.RESI_TYPE
        self.workdata_path = self.dataset_path+'/'+data_type
        self.AAdata_path = self.dataset_path+'/'+data_type+'AA'
        self.idx_list = [file[:-4] for file in os.listdir(self.workdata_path) if '.npz' in file]

        if data_type=='train' and not os.path.exists('./norm.npz'):
            self.myinit_norm()
        self.myload_norm()
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, idx):
        pdb_i = self.idx_list[idx]
        input_data = self.get_data(pdb_i)
        return input_data
    
    def get_data(self, pdb_idx):
        npz_path = self.workdata_path+'/{}.npz'.format(pdb_idx)
        npz_file = np.load(npz_path, allow_pickle=True)
        seq = npz_file['seq']
        #positionemb = npz_file['positionemb']
        label = npz_file['label']
        #addseq=npz_file['addseq']
        #addseq=torch.from_numpy(addseq).float()
        #position=np.zeros(3)
        #res_id=npz_file['res_id']
        #res_id=int(res_id)
        pos=npz_file['relpos']
        #square=npz_file['square']
        #ctheta=npz_file['ctheta']
        dismap=npz_file['dismap']
        seq = self.onehot_encode(seq, self.resi_type)
        #labels = self.trans_label(lables)
        #dismap /= self.norm_dic['dismap_max']
        #square/=self.norm_dic['square_max']
        label=float(label)
        #label /= self.norm_dic['label_99th']
        seq = torch.from_numpy(seq).float()
        '''
        for i in range(len(seq[0])):
            if (2 * i + 1) == len(seq[0]):
                t = i
                break
        if seq[3][t] == 1:
            label = label - 3.67
        if seq[6][t] == 1:
            label = label - 4.25
        if seq[8][t] == 1:
            label = label - 6.54
        if seq[11][t] == 1:
            label = label - 10.40
        '''
        addseq=torch.zeros((4))
        for i in range(dismap.shape[0]):
            if dismap[i][i][0]==0.:
                resid=i
        if seq[3][resid] == 1:
            addseq[0]=1.0
        if seq[6][resid] == 1:
            addseq[1]=1.0
        if seq[8][resid] == 1:
            addseq[2]=1.0
        if seq[11][resid] == 1:
            addseq[3]=1.0
        relpos=np.zeros((16,dismap.shape[0],dismap.shape[0]))
        for i in range(dismap.shape[0]):
            for j in range(dismap.shape[0]):
                dij=abs(int(pos[i])-int(pos[j]))
                if dij<16:
                    if dij>=0:
                        dij = int(dij)
                        relpos[dij][i][j]=1.0
                else:
                    relpos[15][i][j]=1.0
        #label=self.trans_label(label)
        label /= self.norm_dic['label_max']
        '''
        x = torch.from_numpy(positionemb[0]/self.norm_dic['x_max']).float()
        x=torch.unsqueeze(x,dim=0)
        y = torch.from_numpy(positionemb[1] / self.norm_dic['y_max']).float()
        y = torch.unsqueeze(y, dim=0)
        z = torch.from_numpy(positionemb[2] / self.norm_dic['z_max']).float()
        z = torch.unsqueeze(z, dim=0)
        position=torch.cat((x,y,z),dim=0)
        if np.max(ctheta)>=1.0:
            ctheta/=np.max(ctheta)
        ctheta=np.arccos(ctheta)
        '''
        dismap = torch.from_numpy(dismap).float()
        #square=torch.from_numpy(square).float()
        #ctheta=torch.from_numpy(ctheta).float()
        relpos=torch.from_numpy(relpos).float()
        res_ids=np.zeros((dismap.shape[0],dismap.shape[0]))
        res_ids[resid][resid]=1.0
        res_ids=torch.from_numpy(res_ids).float()
        lables=np.zeros([1])
        lables[0]=label
        lables=torch.from_numpy(lables).float()
        return (pdb_idx, seq,dismap, addseq,relpos,res_ids, lables)

    def onehot_encode(self, x, dictset):# after encode s(20,h)
        #if x not in allowset:
            #x = allowset[0]  # UNK
        bool_list = list(map(lambda s: x == s, dictset))
        x = np.array(bool_list).astype(np.float32)
        return x

    def trans_label(self, label):
        if label>0:
            t=np.log(0.2+label)-np.log(0.2)
            return t
        if label<0:
            t=-np.log(0.2-label)+np.log(0.2)
            return t
        #return np.where(label > 0, np.log1p(label), -np.log1p(-label))  # show warning, dont know why
        #return np.array([np.log1p(x) if x>0 else -np.log1p(-x) for x in label])

    def detrans_label(self, label):
        if label > 0:
            t=label+np.log(0.2)
            t=np.exp(t)-0.2
            return t
        if label < 0:
            t = label - np.log(0.2)
            t=-1.0*t
            t = -np.exp(t) + 0.2
            return t
        if label==0:
            return label
        #label = np.clip(label, -np.log(1e+6), np.log(1e+6)) # for inf in pred
        #label =  np.where(label > 0, np.expm1(label), -np.expm1(-label))  # show warning, dont know why
        #label =  np.array([np.expm1(x) if x>0 else -np.expm1(-x) for x in label])
        #return label

    def myinit_norm(self):
        dismap_list = []
        label_list = []
        maxl=0.0
        maxs=0.0
        #maxys=0.0
        #maxzs=0.0
        for idx in tqdm(self.idx_list):
            npz_path = self.workdata_path + '/{}.npz'.format(idx)
            npz_file = np.load(npz_path, allow_pickle=True)
            square = npz_file['square']
            dismap=npz_file['dismap']
            label = npz_file['label']
            seq=npz_file['seq']
            label=float(label)
            '''
            res_id=npz_file['res_id']
            res_id=int(res_id)
            for i in range(len(seq)):
                if (2*i+1)==len(seq):
                    t=i
                    break
            if seq[t] == 'ASP':
                label = label - 3.67
            if seq[t] == 'GLU':
                label = label - 4.25
            if seq[t] == 'HIS':
                label = label - 6.54
            if seq[t] == 'LYS':
                label = label - 10.40
                '''
            maxx=np.max(square)
            maxy=np.max(dismap)
            #maxz=np.max(np.abs(positionemb[2]))
            if maxx > maxl:
                maxl=maxx
            if maxy>maxs:
                maxs=maxy
                '''
            if maxz > maxzs:
                maxzs = maxz
                '''
            #label=self.trans_label(label)
            label_list.append(label)

            #dismap_list += list(dismap.reshape(-1))
        #dismap_list = np.array(dismap_list)
        #label_list = self.trans_label(label_list)
        label_list = np.abs(label_list)
        self.norm_dic = {
            #'dismap_mean': np.mean(dismap_list),
            #'x_max': maxxs,
            #'y_max':maxys,
            #'z_max':maxzs,
            #'dismap_std': np.std(dismap_list),
            #'dismap_99th': np.percentile(dismap_list, 99),
         #   'label_mean': np.mean(label_list),
            'dismap_max':maxs,
            'label_max': np.max(label_list),
            'square_max':maxl,
           # 'label_std': np.std(label_list),
            #'label_99th': maxl,
        }
        np.savez('./norm.npz', **self.norm_dic)
    def myload_norm(self):
        npz_file = np.load('./norm.npz',allow_pickle=True)
        self.norm_dic = {k:npz_file[k] for k in npz_file.keys()}
    
class TrainRecorder:
    def __init__(self, name):
        self.name = name
        self.record_dic = {}
        self.batch_dic = {}

    def __call__(self, k, x):
        self.batch_dic[k].append(x)

    def reset_recorder(self, k):
        self.record_dic[k] = []
        self.batch_dic[k] = []

    def update_recorder(self, batch_size=1):
        for k in self.record_dic.keys():
            self.record_dic[k].append(np.mean(self.batch_dic[k])/batch_size)
            self.batch_dic[k]=[]

    def save_record(self):
        np.savez('./{}.npz'.format(self.name), **self.record_dic)

    def load_record(self, restart=None):
        recorded = np.load('./{}.npz'.format(self.name))
        if restart==None:
            for k in recorded.keys():
                self.record_dic[k] = list(recorded[k])
        else:
            for k in recorded.keys():
                self.record_dic[k] = list(recorded[k])[:restart]

    def return_record(self, k):
        return self.record_dic[k]

class DataWindow:
    def __init__(self):
        pass

    def cal_pearsonr(self, pred_arr, label_arr):
        result = scipy.stats.pearsonr(pred_arr, label_arr)
        return result[0]

    def cal_kendall(self, pred_arr, label_arr):
        result = scipy.stats.kendalltau(pred_arr, label_arr)
        return result[0]

    def loss_plot(self, x_train, y_train, y_val):
        with plt.style.context(['science', 'no-latex']):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(x_train, y_train, color='red', label='train')
            for tl in ax1.get_yticklabels():
                tl.set_color("red")
            ax2.plot(x_train, y_val, color='blue', label='val')
            for tl in ax2.get_yticklabels():
                tl.set_color("blue")
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Loss')
            ax2.set_ylabel('Val Loss')
            
            lines = []
            labels = []
            for ax in fig.axes:
                axLine, axLabel = ax.get_legend_handles_labels()
                lines.extend(axLine)
                labels.extend(axLabel)
            fig.legend(lines, labels, loc = 'lower right')
            
            plt.title('Loss')

            plt.savefig('Loss.png', dpi=300)
            plt.close()
    
    def pearson_plot(self, x_train, y_train, y_val):
        with plt.style.context(['science', 'no-latex']):
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(x_train, y_train, color='red', label='train')
            for tl in ax1.get_yticklabels():
                tl.set_color("red")
            ax2.plot(x_train, y_val, color='blue', label='val')
            for tl in ax2.get_yticklabels():
                tl.set_color("blue")

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Train Cor')
            ax2.set_ylabel('Val Cor')
            lines = []
            labels = []
            for ax in fig.axes:
                axLine, axLabel = ax.get_legend_handles_labels()
                lines.extend(axLine)
                labels.extend(axLabel)
            fig.legend(lines, labels, loc = 'lower right')
            plt.title('Pearson Correlation Coefficient')

            plt.savefig('Correlation.png', dpi=300)
            plt.close()

    def test_hex_plot(self, pred_list, label_list, pearsonr):
        x = np.linspace(min(pred_list.min(), label_list.min()) ,max(pred_list.max(), label_list.max()), num=100)
        y = x
        with plt.style.context(['science', 'no-latex']):
            fig = plt.figure()
            plt.plot(x, y, c='k', linewidth=0.5)
            plt.hexbin(label_list, pred_list, cmap='jet', gridsize=100, bins='log', norm=LogNorm(vmin=1,vmax=1e3))
            plt.axis("equal")
            plt.xlabel("label data")
            plt.ylabel("pred data")
            plt.title("Pearsonr: {:.4f}".format(pearsonr))
            plt.savefig("Result_hexbin0.png", dpi=300)
            plt.close()
    
    def test_scatter_plot(self, pred_list, label_list, pearsonr):
        x = np.linspace(min(pred_list.min(), label_list.min()) ,max(pred_list.max(), label_list.max()), num=100)
        y = x
        with plt.style.context(['science', 'no-latex']):
            fig = plt.figure()
            plt.plot(x, y, c='k', linewidth=0.5)
            plt.scatter(label_list, pred_list, s=5)
            plt.axis("equal")
            plt.xlabel("label data")
            plt.ylabel("pred data")
            plt.title("Pearsonr: {:.4f}".format(pearsonr))
            plt.savefig("Result_scatter_line0.png", dpi=300)
            plt.close()

    def print_record(self, name, loss, relation):
        print('{} Loss'.format(name).ljust(20, '.')+'{:.6f}'.format(loss).rjust(20, '.'))
        print('{} Correlation'.format(name).ljust(20, '.')+'{:.4f}'.format(relation).rjust(20, '.'))
