import os

import torchvision.transforms
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler as DS
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange

import cgpbcoul_pytorch.constants as constants
from cgpbcoul_pytorch.model import *
from cgpbcoul_pytorch.utils import *
import scienceplots

torch.set_default_tensor_type(torch.FloatTensor)

def setup_ddp(rank, world_size, port, device):
    if device == 'cuda':
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = port
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        #dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
    else:
        rank = device

def train(rank, world_size, port, device):
    setup_ddp(rank, world_size, port, device)
    n_epoch = constants.TOTAL_EPOCH
    lr = constants.LR_START
    batch_size = constants.BATCH_SIZE
    weight_path = constants.SAVE_WEIGHT_DIR
    thread_num = 5
    model = CGSolvElec()
    restart = 0

    model = model.to(rank)
    if device == 'cuda':
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if restart!=0:
        model.load_state_dict(torch.load(weight_path+"/model_weight_{}.ckpt".format(restart)))
    elif rank==0:
        delmkdir(weight_path)

    train_dataset = CGDataSet('train')
    val_dataset = CGDataSet('test')
    loss_type = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    lambda1 = lambda i: scheduler_lr(i)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=- 1)

    if rank==0 or rank=='cpu':
        val_loader = DataLoaderX(dataset=val_dataset, batch_size=batch_size, num_workers=thread_num-1)
        LossRecorder = TrainRecorder('LossRecorder')
        LossRecorder.reset_recorder('TrainLoss')
        LossRecorder.reset_recorder('ValLoss')
        CorRecorder = TrainRecorder('CorrelationRecorder')
        CorRecorder.reset_recorder('TrainR')
        CorRecorder.reset_recorder('ValR')
        if restart != 0:
                LossRecorder.load_record(restart)
                CorRecorder.load_record(restart)
    
    if device=='cuda':
        train_sampler = DS(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoaderX(dataset=train_dataset, batch_size=batch_size,sampler=train_sampler, num_workers=thread_num-1)
    if device!='cuda':
        train_loader = DataLoaderX(dataset=train_dataset, batch_size=batch_size, num_workers=thread_num-1, shuffle=True)
    
    min_loss = 1e6
    max_relation = 0.0
    for epoch in range(restart, n_epoch):
        #running_loss = 0.0
        val_loss = 0.0
        if device == 'cuda':
            train_sampler.set_epoch(epoch)
        #training
        model.train()
        pred_data=[]
        label_data=[]
        for i,(pdb_idx, seq,dismap ,addseq,relpos,pos, label) in enumerate(train_loader):
            optimizer.zero_grad()
            seq, dismap, addseq, relpos, pos, label = seq.to(rank), dismap.to(rank), addseq.to(rank), relpos.to(rank), pos.to(rank), label.to(rank)
            pred = model(seq, dismap, addseq, relpos, pos).squeeze()
            label=label.squeeze()
            #if seq[0][3][int(res_id[0][0])]==1:
               # label=label-3.67
            #if seq[0][6][int(res_id[0][0])]== 1:
                #label=label-4.25
            #if seq[0][8][int(res_id[0][0])]==1:
              #  label=label-6.54
            #if seq[0][11][int(res_id[0][0])]==1:
                #label=label-10.40
            loss=loss_type(pred, label)
            print('no.%d:loss=%f'%(i,loss),flush=True)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=0.12)
            optimizer.step()
            #running_loss += loss.item()

            if rank==0 or rank=='cpu':
                lossp = loss.detach().cpu().numpy()
                predp = pred.detach().cpu().numpy()#*val_dataset.norm_dic['label_max']
                labelp = label.detach().cpu().numpy()#*val_dataset.norm_dic['label_max']
                LossRecorder('TrainLoss', lossp)
                #predp = val_dataset.detrans_label(predp)
                #labelp = val_dataset.detrans_label(labelp)
                #print(type(predp),type(labelp),flush=True)
                label_data.append(labelp)
                pred_data.append(predp)
        if rank == 0 or rank == 'cpu':
            label_data=np.array(label_data)
            pred_data=np.array(pred_data)
            CorRecorder('TrainR', DataWindow.cal_pearsonr(self=None,pred_arr=pred_data,label_arr=label_data))
            scheduler.step()

        '''
        if device == 'cuda':
            dist.barrier()
        '''
        pred_datav = []
        label_datav = []
        if rank==0 or rank=='cpu':
            torch.save(model.state_dict(), constants.SAVE_WEIGHT_DIR+'/'+'model_weight_{}.ckpt'.format(epoch+1))
            model.eval()
            with torch.no_grad():
                for i,(pdb_idx, seq,dismap ,addseq,relpos,pos, label) in enumerate(val_loader):
                    seq,dismap, addseq,relpos,pos,label = seq.to(rank), dismap.to(rank), addseq.to(rank),relpos.to(rank),pos.to(rank),label.to(rank)
                    pred = model(seq,dismap, addseq,relpos,pos).squeeze()
                    label=label.squeeze()
                        #if seq[0][3][int(res_id[0][0])] == 1:
                            #label = label - 3.67
                        #if seq[0][6][int(res_id[0][0])] == 1:
                           # label = label - 4.25
                        #if seq[0][8][int(res_id[0][0])] == 1:
                           # label = label - 6.54
                        #if seq[0][11][int(res_id[0][0])] == 1:
                           # label = label - 10.40
                    loss_val = loss_type(pred, label)
                    lossp = loss_val.detach().cpu().numpy()
                    val_loss += loss_val.item()
                        #print(val_loss,flush=True)
                    predp = pred.detach().cpu().numpy()#*val_dataset.norm_dic['label_max']
                    labelp = label.detach().cpu().numpy()#*val_dataset.norm_dic['label_max']
                    #predp=val_dataset.detrans_label(predp)
                    #labelp=val_dataset.detrans_label(labelp)
                    LossRecorder('ValLoss', lossp)
                    pred_datav.append(predp)
                    label_datav.append(labelp)
                #pred_datav = val_dataset.detrans_label(pred_datav)
                #label_datav=val_dataset.detrans_label(label_datav)
                label_data = np.array(label_data)
                pred_data = np.array(pred_data)
                CorRecorder('ValR', DataWindow.cal_pearsonr(self=None,pred_arr=pred_datav,label_arr= label_datav))
                LossRecorder.update_recorder(batch_size)
                CorRecorder.update_recorder(batch_size)
                print('Epoch [{}/{}], lr: {:.2e}'.format(epoch+1, n_epoch, optimizer.param_groups[0]['lr']))
                plt.style.use('science')
                DataWindow().print_record('Train', LossRecorder.return_record('TrainLoss')[-1],CorRecorder.return_record('TrainR')[-1])
                DataWindow().print_record('Val', LossRecorder.return_record('ValLoss')[-1],CorRecorder.return_record('ValR')[-1])
                LossRecorder.save_record()
                CorRecorder.save_record()
                DataWindow().loss_plot(np.arange(epoch+1),LossRecorder.return_record('TrainLoss'), LossRecorder.return_record('ValLoss'))
                DataWindow().pearson_plot(np.arange(epoch+1), CorRecorder.return_record('TrainR'), CorRecorder.return_record('ValR'))

                if val_loss < min_loss:
                    torch.save(model.state_dict(), 'minloss_model.pkl')
                    min_loss = val_loss

                if CorRecorder.return_record('ValR')[-1] > max_relation:
                    torch.save(model.state_dict(), 'maxcor_model.pkl')
                    max_relation = CorRecorder.return_record('ValR')[-1]

        '''
        if device == 'cuda':
            dist.barrier()
        '''
        
    if device == 'cuda':
        dist.destroy_process_group()

def main():
    device = constants.DEVICE_NAME
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    world_size = 3
    port = '11200'
    args = (world_size, port, device)
    if device == 'cuda':
        torch.multiprocessing.spawn(train, args=args, nprocs=world_size, join=True)
    else:
        torch.set_num_threads(8)
        train(0, world_size, port, device)

if __name__ == '__main__':
    main()