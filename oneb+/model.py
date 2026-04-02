import numpy as np
import torch
from torch import nn
import constants as constants
from modules import *
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from einops import rearrange, repeat, reduce
import heapq


class ConBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            PreActiveBlock2D(
                in_channels,
                out_channels,
                kernel_size=(1,3),
                stride=1,
                padding='same'
            ),
            PreActiveBlock2D(
                out_channels,
                out_channels,
                kernel_size=(3,1),
                stride=1,
                padding='same'
            )
        )
        self.shortcut_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            stride=1
        )

    def forward(self, x):
        x1 = self.layers(x)
        if self.in_channels == self.out_channels:
            identity = x
        if self.in_channels != self.out_channels:
            identity = self.shortcut_conv(x)
        output = x1 + identity
        return output
class Res_Evo_block(nn.Module):
    def __init__(self,in_channels,out_channels,middle_channels,repeaet_layers,dim,heads,dim_head,attn_dropout,ff_dropout,seq_len):
        super().__init__()
        '''
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=middle_channels,kernel_size=(1,1),stride=1),
            nn.InstanceNorm2d(num_features=middle_channels),
            nn.GELU(),
            nn.Dropout(ff_dropout)
        )
        '''
        conv2 = ConBlock2D(in_channels=out_channels,out_channels=out_channels)
        res_layer = []
        res_layer1=[]
        for _ in range(repeaet_layers):
            res_layer.append(conv2)
            res_layer1.append(conv2)
        self.reslayer = nn.Sequential(*res_layer)
        self.reslayer1=nn.Sequential(*res_layer1)
        '''
        self.c2=nn.Sequential(
            nn.Conv2d(in_channels=middle_channels, out_channels=out_channels, kernel_size=(3, 3), stride=1,
                      padding=1),
            nn.InstanceNorm2d(num_features=middle_channels),
            nn.GELU()
        )
        self.c1=conv1
        self.c3=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(1,1),stride=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Dropout(ff_dropout)
        )
        '''
        self.to_T=nn.Sequential(
            nn.Conv2d(kernel_size=(1,1),in_channels=in_channels,out_channels=dim,stride=1),
            nn.Dropout(ff_dropout),
            Rearrange('b c n m -> b n m c'),
            FeedForward(dim=dim),
            nn.Dropout(ff_dropout)
        )
        self.to_R=nn.Sequential(
            Rearrange('b n m c -> b c n m'),
            nn.Conv2d(kernel_size=(1,1),in_channels=dim,out_channels=out_channels,stride=1),
            nn.GroupNorm(num_channels=out_channels, num_groups=32),
            nn.Dropout(ff_dropout),
            #nn.GroupNorm(num_channels=out_channels,num_groups=32),
            nn.GELU(),
        )
        self.afnet= EvoformerBlock(
                dim=dim,
                seq_len=seq_len,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        self.inchanel=in_channels
        self.middle_chanels=middle_channels
    def forward(self, x):
        if self.middle_chanels==128:
            #x=self.encode(x)
            #xr=self.c1(x)
            xt=self.to_T(x)
            xt=self.afnet(xt)
            #xr=self.c2(xr)
            #xra=self.c3(x)
            xr=self.reslayer(x)
            ttor=self.to_R(xt)
            xr=xr+ttor
            xr=self.reslayer1(xr)
            x={xr,xt}
        else:
            xr,xt=x
            rtot=self.to_T(xr)
            xt=xt+rtot
            xt=self.afnet(xt)
            xr=self.reslayer(xr)
            ttor = self.to_R(xt)
            xr = ttor + xr
            xr = self.reslayer1(xr)
            x = {xr,xt}
        return x

'''
class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(
            in_channels = 48,
            out_channels=constants.LAYER_PARA['2d_BlockChannels'][0],
            kernel_size=(7,7),
            stride=(2,2)
            #padding='same'
        )
        pool1 = nn.MaxPool2d(
            kernel_size = (3,3),
            stride = (2,2),
            padding = (1,1)
        )
        conv2_1 = BottleneckResBlock2D(
            in_channels = 48,
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            stride = 1
        )
        conv2_2 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            stride = 1
        )
        conv3_1 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            stride = (2,2)
        )
        conv3_2 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            stride = 1
        )
        conv4_1 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][4],
            stride = (2,2)
        )
        conv4_2 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][4],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][4],
            stride = 1
        )
        conv5_1 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][4],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][5],
            stride = (2,2)
        )
        conv5_2 = BottleneckResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][5],
            middle_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][5],
            stride = 1
        )
        pool2 = nn.AdaptiveAvgPool2d((1,1))
        res_layer = []
        #res_layer.append(conv1)
        #res_layer.append(pool1)
        res_layer.append(conv2_1)
        for _ in range(constants.LAYER_PARA['2d_BlockLayer'][0]-1):
            res_layer.append(conv2_2)
        res_layer.append(conv3_1)
        for _ in range(constants.LAYER_PARA['2d_BlockLayer'][1]-1):
            res_layer.append(conv3_2)
        res_layer.append(conv4_1)
        for _ in range(constants.LAYER_PARA['2d_BlockLayer'][2]-1):
            res_layer.append(conv4_2)
        res_layer.append(conv5_1)
        for _ in range(constants.LAYER_PARA['2d_BlockLayer'][3]-1):
           res_layer.append(conv5_2)
        res_layer.append(pool2)
        self.reslayer = nn.Sequential(*res_layer)
        self.downchannels=nn.Conv2d(
            in_channels=constants.LAYER_PARA['2d_BlockChannels'][5],
            out_channels=32,
            kernel_size=(1,1)
        )
        conv2=ResBlock2D(
            in_channels = 48,
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            kernel_size=(3,3),
            stride = 1,
            padding=1
        )
        conv2_2=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            kernel_size=(3,3),
            stride = 1,
            padding=1
        )
        cov3_1=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][0],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            kernel_size=(3,3),
            stride = (2,2),
            padding=1
        )
        cov3_2=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            kernel_size=(3,3),
            stride = 1,
            padding=1
        )
        conv4_1=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][1],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            kernel_size=(3,3),
            stride = (2,2),
            padding=1
        )
        conv4_2=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            kernel_size=(3,3),
            stride = 1,
            padding=1
        )
        conv5_1=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][2],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            kernel_size=(3,3),
            stride = (2,2),
            padding=1
        )
        conv5_2=ResBlock2D(
            in_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            out_channels = constants.LAYER_PARA['2d_BlockChannels'][3],
            kernel_size=(3,3),
            stride = 1,
            padding=1
        )
        pool2 = nn.AdaptiveAvgPool2d((1, 1))
        res_layer = []
        #res_layer.append(conv1)
        #res_layer.append(pool1)
        res_layer.append(conv2)
        res_layer.append(conv2_2)
        res_layer.append(cov3_1)
        res_layer.append(cov3_2)
        res_layer.append(conv4_1)
        res_layer.append(conv4_2)
        res_layer.append(conv5_1)
        res_layer.append(conv5_2)
        res_layer.append(pool2)
        self.reslayer = nn.Sequential(*res_layer)
       
    def forward(self, x):
        x = self.reslayer(x)
        x=self.downchannels(x)
        x=rearrange(x,'b c h w -> (b h w) c')
        return x'''
class DeepCGpKa(nn.Module):
    def __init__(
        self,
        *,
        #dim,
        max_seq_len = 1024,
        #depth = 1,
        #heads = 8,
        #dim_head = 32,
        attn_dropout = 0.0,
        ff_dropout = 0.0
    ):
        super().__init__()
        self.para = constants.LAYER_PARA
        self.relposencoding=nn.Sequential(
            #nn.LayerNorm(16),
            nn.Linear(16,64)
        )
        self.spaceencoding=nn.Sequential(
            Rearrange('b n m c -> b c n m'),
            nn.InstanceNorm2d(num_features=9, affine=True, track_running_stats=True, momentum=0.9),
            nn.Conv2d(in_channels=9, out_channels=64, kernel_size=(1, 1), stride=1, padding='same'),
            Rearrange('b c n m -> b n m c')
        )
        '''
        self.spaceencoding2 = nn.Sequential(
            Rearrange('b n m c -> b c n m'),
            nn.InstanceNorm2d(num_features=4,affine=True,track_running_stats=True,momentum=0.5),
            nn.Conv2d(in_channels=4,out_channels=128,kernel_size=(1,1),stride=1,padding='same'),
            Rearrange('b c n m -> b n m c')
        )
        '''
        self.seqencoding=nn.Sequential(
            #nn.LayerNorm(44),
            nn.Linear(44,64)
        )
        self.targetencoding=nn.Linear(1,64)
        self.activate=nn.GELU()

        # 1d seq block
        in_channels = self.para['n_ResiType']
        self.rtnet1 =Res_Evo_block(
            in_channels=64,
            out_channels=64,
            middle_channels=128,
            repeaet_layers=1,
            dim=self.para['AFDim'][0],
            seq_len=max_seq_len,
            heads=self.para['AFHeads'][0],
            dim_head=self.para['AFHeadDim'][0],
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        '''
        self.rtnet2 = Res_Evo_block(
            in_channels=128,
            out_channels=128,
            middle_channels=96,
            repeaet_layers=1,
            dim=self.para['AFDim'][1],
            seq_len=max_seq_len,
            heads=self.para['AFHeads'][1],
            dim_head=self.para['AFHeadDim'][1],
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.rtnet3 =Res_Evo_block(
            in_channels=256,
            out_channels=256,
            middle_channels=96,
            repeaet_layers=3,
            dim=self.para['AFDim'][2],
            seq_len=max_seq_len,
            heads=self.para['AFHeads'][2],
            dim_head=self.para['AFHeadDim'][2],
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        '''
        self.decode=DEvoformerBlock(
            dim=self.para['AFDim'][3],
            seq_len=max_seq_len,
            heads=self.para['AFHeads'][3],
            dim_head=self.para['AFHeadDim'][3],
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.todecode=nn.Sequential(
            nn.GroupNorm(num_channels=64,num_groups=32),
            nn.Conv2d(in_channels=64,out_channels=64,stride=1,kernel_size=(1,1)),
            nn.Dropout(ff_dropout),
            Rearrange('b c n m -> b n m c'),
            #nn.LayerNorm(64),
            FeedForward(dim=64),
            nn.Dropout(ff_dropout)
        )
        self.downchannel =nn.Sequential(
            nn.Linear(64,1),
            nn.Dropout(ff_dropout)
        )
    def forward(
        self, 
        seq_input, 
        dismap_input,
        #ctheta_input,
        addseq,
        relpos,
        pos,
        seq_mask=None, 
        dismap_mask=None 
    ):
        #x_1 = seq_input
        #x_1 = x_1.unsqueeze(-2)
        seq_len = seq_input.shape[-1]
        addseq=repeat(addseq,'b c -> b c n', n=seq_len)
        #addseq = repeat(addseq, 'b c n-> b c n l', l=seq_len)
        #x_1_t = torch.einsum("...ij -> ...ji", [x_1])
        #x_1_p1=repeat(seq_input,"b c n -> b c n l", l=seq_len)
       #x_1_p2=repeat(seq_input,"b c n -> b c l n", l=seq_len)
        xt=rearrange(seq_input,'b c n -> b n c')
        x=torch.cat((seq_input,addseq),dim=1)
        x=rearrange(x,'b c n -> b n c')
        xt=repeat(xt,'b h c -> b h w c',w=seq_len)
        x=repeat(x,'b h c ->b w h c',w=seq_len)
        x=torch.cat((x,xt),dim=-1)
        x0=self.seqencoding(x)
        #x=x+xt
        #x=self.afnet1(x)
        #x=torch.cat((x,addseq),dim=1)
        #x_1 = torch.einsum("...ki, ...jk -> ...ji", [x_1, x_1_t])
        #x_1_p=x_1_p-x_1
        xs = self.spaceencoding(dismap_input)
        #x_3=repeat(square_input,"b h w -> b h w c",c=1)
        #x_4=self.spaceencoding2(ctheta_input)
        pos=repeat(pos,"b h w -> b h w c",c=1)
        '''
        x = rearrange(x,'b c h w -> b h w c') # reshape torch2tf
        #x_1 = checkpoint_sequential(self.afnet, 1, x_1)
        x = self.afnet(x) # use DDP with find_unused_parameters so do not use checkpoint
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        #xs=torch.cat((x,x_2),dim=1)
        #xs = x_1+x_4
        #xs = torch.cat((xs, x_4), dim=-1)
        #xs = rearrange(xs,'b n m c -> b c n m')
        #xress=rearrange(xs,'b h w c -> b c h w')
        #xs = torch.cat((xs,pos),dim=-1)
        #xress = rearrange(xs, 'b h w c -> b c h w')
        #xs=self.spacecoding(xs)+xs
        xp=rearrange(relpos,'b c h w -> b h w c')
        xp=self.relposencoding(xp)
        #xs=self.spaceencoding(xs)
        xm=self.targetencoding(pos)
        x=xp+xs+x0+xm
        x=self.activate(x)
        x=rearrange(x,'b n m c -> b c n m')
        #xp=self.poscoding(xp)+xp
        #x=torch.cat((x,xp),dim=-1)
        #x=self.afnet2(x)
        #x=torch.cat((x,xs),dim=-1)
        #x = rearrange(x, 'b c h w -> b h w c') # reshape torch2tf
        #x=self.afnet1(x)
        #x={x,int(pos)}
        #x=self.decode(x)
        #x=rearrange(x,'b n m c -> b c n m')
        '''
        xres=repeat(seq_input,'b c n -> b c n l',l=seq_len)
        xres_t=repeat(seq_input,'b c n -> b c l n',l=seq_len)
        addseq=repeat(addseq,'b c n -> b c l n',l=seq_len)
        xres=torch.cat((xres,xres_t,addseq,xress),dim=1)
        '''
        x = self.rtnet1(x)
        #x = self.rtnet2(x)
        #x = self.rtnet3(x)
        xr,xt=x
        x=self.todecode(xr)+xt
        #x=torch.cat((xr,xt),dim=-1)
        x=self.decode(x)
        x=self.downchannel(x)
        '''
        rtot=self.rtot1(xres)
        #ttor=self.ttor1(x)
        x=x+rtot
        #xres=xres+ttor
        x = self.afnet2(x)
        xres=self.reslayer2(xres)
        rtot=self.rtot2(xres)
        #ttor=self.ttor2(x)
        x=x+rtot
        #xres=xres+ttor
        x = self.afnet3(x)
        xres=self.reslayer3(xres)
        rtot=self.rtot3(xres)
        #ttor=self.ttor3(x)
        x=x+rtot
        #xres=xres+ttor
        x = self.afnet4(x)
        x = rearrange(x,'b n m c -> b c n m')
        xres=self.reslayer4(xres)
        xres=self.downchannelres(xres)
        x=torch.cat((x,xres),dim=1)
        #x=rearrange(x,'b h c -> b c h')
        #x_2 = repeat(dismap_input, "n h w -> n c h w", c=20)
        #x = x_1*x_2
        #x = checkpoint_sequential(self.reslayer, 1, x)
        #x = self.reslayer(x)
        x = self.downchannel(x)
        #x=torch.flatten(x)
        #x = torch.unsqueeze(x, dim=0)
        '''
        return x # return size (1,N)
