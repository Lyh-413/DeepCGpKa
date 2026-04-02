import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from utils import *

class StdConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StdConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size = kernel_size,
                               stride = 1,
                               padding='same')
        #self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        #x = self.norm(x)
        x = self.relu(x)

        return x

class StdConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(StdConv1d, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size = kernel_size,
                               stride = 1,
                               padding='same')
        #self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        #x = self.norm(x)
        x = self.relu(x)

        return x

class PreActiveBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            )
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x

class PreActiveBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            #nn.BatchNorm2d(in_channels),
            #nn.InstanceNorm2d(num_features=in_channels),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        x = self.layer(x)
        return x

class BottleneckResBlock2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, stride,drop_out):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            PreActiveBlock2D(
                in_channels = self.in_channels,
                out_channels = middle_channels,
                kernel_size = (1,1),
                stride = 1,
                padding = 0
            ),
            nn.Dropout(drop_out),
            PreActiveBlock2D(
                in_channels = middle_channels,
                out_channels = middle_channels,
                kernel_size = (3,3),
                stride = stride,
                padding = (1,1)
            ),
            PreActiveBlock2D(
                in_channels = middle_channels,
                out_channels = self.out_channels,
                kernel_size = (1,1),
                stride = 1,
                padding = 0
            ),
            nn.Dropout(drop_out)
        )
        self.scalelayer = nn.Sequential(nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = (1,1),
            stride = stride
            ),
            nn.Dropout(drop_out)
        )
    def forward(self, x):
        x1 = self.layers(x)
        if self.in_channels == self.out_channels:
            identity = x
        if self.in_channels != self.out_channels:
            identity = self.scalelayer(x)
        output = x1 + identity
        return output

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            PreActiveBlock1D(
                in_channels,
                out_channels,
                kernel_size
            ),
            PreActiveBlock1D(
                out_channels,
                out_channels,
                kernel_size
            )
        )
        self.shortcut_conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = (1,1),
            stride = 1,
            padding='same'
        )
    
    def forward(self, x):
        x1 = self.layers(x)
        if self.in_channels == self.out_channels:
            identity = x
        if self.in_channels != self.out_channels:
            identity = self.shortcut_conv(x)
        output = x1 + identity
        return output

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.Sequential(
            PreActiveBlock2D(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding
            ),
            PreActiveBlock2D(
                out_channels,
                out_channels,
                kernel_size,
                stride=1,
                padding='same'
            )
        )
        self.shortcut_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size = (1,1),
            stride = stride
        )
    
    def forward(self, x):
        x1 = self.layers(x)
        if self.in_channels == self.out_channels:
            identity = x
        if self.in_channels != self.out_channels:
            identity = self.shortcut_conv(x)
        output = x1 + identity
        return output


class PatchMerging(nn.Module):


    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        self.pad=nn.ReplicationPad2d(padding=(1,0,1,0))

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B,H,W, C = x.shape

        if H%2!=0:
            x=rearrange(x,'b h w c -> b c h w ')
            x=self.pad(x)
            x=rearrange(x,'b c h w -> b h w c ')
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x
class poolhalf(nn.Module):
    def __init__(self):
        super().__init__()
        self.poolh=nn.AvgPool2d(
            kernel_size=(2,2),
            stride=2)
        self.pad = nn.ReplicationPad2d(padding=(1, 0, 1, 0))

    def forward(self, x):
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w c -> b c h w ')
        if H % 2 != 0:
            x = self.pad(x)
        x=self.poolh(x)
        x = rearrange(x, 'b c h w -> b h w c ')
        return x

#Modules in AF2, codes from https://github.com/lucidrains/alphafold2
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
        init_zero_(self.net[-1])

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        dots = dots - dots.max(dim = -1, keepdims = True).values
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(x)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        global_query_attn = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn

        self.norm = nn.LayerNorm(dim)

        self.attn = Attention(dim = dim, heads = heads, **kwargs)

        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None

    def forward(self, x, edges = None, mask = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'

        b, h, w, d = x.shape

        x = self.norm(x)

        # axial attention

        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'

        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'

        x = rearrange(x, input_fold_eq)

        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None

        out = self.attn(x, mask = mask, attn_bias = attn_bias, tie_dim = tie_dim)
        out = rearrange(out, output_fold_eq, h = h, w = w)

        return out

class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)

        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        if exists(mask):
            left = left * mask
            right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.,
        global_column_attn = False
    ):
        super().__init__()
        #self.outer_mean = OuterMean(dim)

        self.triangle_attention_outgoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.triangle_attention_ingoing = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = True, global_query_attn = global_column_attn)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')

    def forward(
        self,
        x,
        mask = None,
        msa_repr = None,
        msa_mask = None
    ):
        #if exists(msa_repr):
        #    x = x + self.outer_mean(msa_repr, mask = msa_mask)

        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        x = self.triangle_attention_outgoing(x, edges = x, mask = mask) + x
        x = self.triangle_attention_ingoing(x, edges = x, mask = mask) + x
        return x
'''
class MsaAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout = 0.
    ):
        super().__init__()
        self.row_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True)
        self.col_attn = AxialAttention(dim = dim, heads = heads, dim_head = dim_head, row_attn = False, col_attn = True)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        x = self.col_attn(x, mask = mask) + x
        return x
'''
class EvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            PairwiseAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout, global_column_attn = global_column_attn),
            FeedForward(dim = dim, dropout = ff_dropout),
            #MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            #FeedForward(dim = dim, dropout = ff_dropout),
        ])

    def forward(self, inputs):
        #x, m, mask, msa_mask = inputs
        x = inputs
        #attn, ff, msa_attn, msa_ff = self.layer
        attn, ff = self.layer

        #m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        #m = msa_ff(m) + m

        #x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = attn(x, mask=None, msa_repr=None, msa_mask = None)
        x = ff(x) + x

        #return x, m, mask, msa_mask
        #return x, mask
        return x
class DAttention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len = None,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        gating = True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads= heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(self, x,pos, mask = None, attn_bias = None, context = None, context_mask = None, tie_dim = None):
        device, orig_shape, h, has_context = x.device, x.shape, self.heads, exists(context)

        context = default(context, x)
        qx=rearrange(x,'b n c -> n b c')
        qx=qx[int(pos)]
        qx=repeat(qx,'b c -> b n c',n=1)
        q, k, v = (self.to_q(qx), *self.to_kv(context).chunk(2, dim = -1))

        i, j = q.shape[-2], k.shape[-2]
        q=rearrange(q,'b n (h d) -> b h n d', h = h)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), ( k, v))

        # scale

        q = q * self.scale

        # query / key similarities

        if exists(tie_dim):
            # as in the paper, for the extra MSAs
            # they average the queries along the rows of the MSAs
            # they named this particular module MSAColumnGlobalAttention

            q, k = map(lambda t: rearrange(t, '(b r) ... -> b r ...', r = tie_dim), (q, k))
            q = q.mean(dim = 1)

            dots = einsum('b h i d, b r h j d -> b r h i j', q, k)
            dots = rearrange(dots, 'b r ... -> (b r) ...')
        else:
            dots = einsum('b h i d, b h j d -> b h i j', q, k)

        # add attention bias, if supplied (for pairwise to msa attention communication)

        if exists(attn_bias):
            dots = dots + attn_bias

        # masking

        if exists(mask):
            mask = default(mask, lambda: torch.ones(1, i, device = device).bool())
            context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device = device).bool())
            mask_value = -torch.finfo(dots.dtype).max
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, mask_value)

        # attention

        dots = dots - dots.max(dim = -1, keepdims = True).values
        attn = dots.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # gating

        gates = self.gating(qx)
        out = out * gates.sigmoid()

        # combine to out

        out = self.to_out(out)
        return out
class DEvoformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        seq_len,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False
    ):
        super().__init__()
        '''
        self.layer = nn.ModuleList([
            DAttention(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            FeedForward(dim = dim, dropout = ff_dropout),
            #MsaAttentionBlock(dim = dim, seq_len = seq_len, heads = heads, dim_head = dim_head, dropout = attn_dropout),
            #FeedForward(dim = dim, dropout = ff_dropout),
        ])
        '''
        self.ff1=FeedForward(dim = dim, dropout = ff_dropout)
        self.ff2=FeedForward(dim = dim, dropout = ff_dropout)
        self.norm=nn.LayerNorm(dim)
        '''
        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, heads, bias=False),
            Rearrange('b i j h -> b h i j')
        )
        '''
        self.tok1=nn.Linear(dim,heads,bias=False)
        self.tok2=nn.Linear(dim,heads,bias=False)
        #self.to_out = nn.Linear(dim_head*heads, dim)
        self.heads=heads
    def forward(self, inputs):
        #x, m, mask, msa_mask = inputs
        x = inputs
        x=self.norm(x)
        b,h,w,d=x.shape
        '''
        edges=rearrange(x,'b h w d -> h b w d')[int(pos)]
        edges=repeat(edges,'b w d -> b i w d',i=1)
        x=rearrange(x,'b h w d -> (b h) w d')
        #attn, ff, msa_attn, msa_ff = self.layer
        attn, ff = self.layer
        attn_bias = self.edges_to_attn_bias(edges)
        attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x=h)
        #m = msa_attn(m, mask = msa_mask, pairwise_repr = x)
        #m = msa_ff(m) + m

        #x = attn(x, mask = mask, msa_repr = m, msa_mask = msa_mask)
        x = attn(x,pos,attn_bias=attn_bias)
        x = ff(x) + x
        x =rearrange(x,'(b h) k d ->(b k) h d',h=h,b=b,d=d,k=1)
        x =self.norm(x)
        '''
        k=self.tok1(x)
        x=  rearrange(x, 'b n w (h d) -> b h n w d', h=self.heads)
        k=rearrange(k,'b i j h -> b h i j')
        #k=repeat(k,'b h i -> b h j i',j=1)
        k = k - k.max(dim = -1, keepdims = True).values
        k = k.softmax(dim = -1)
        x=torch.einsum('b h i j,b h j i d -> b h i d',(k,x))
        x=rearrange(x,'b h i d -> b i (h d)')
        x=self.ff1(x)+x
        k=self.tok2(x)
        x=  rearrange(x, 'b n (h d) -> b h n d', h=self.heads)
        k=rearrange(k,'b i h -> b h i')
        #k=repeat(k,'b h i -> b h j i',j=1)
        k = k - k.max(dim = -1, keepdims = True).values
        k = k.softmax(dim = -1)
        x=torch.einsum('b h i ,b h i d -> b h d',(k,x))
        x=rearrange(x,'b h d -> b (h d)')
        #x=rearrange(x,'b h n d -> b n (h d)')
        #x=self.to_out(x)
        #x=self.norm(x)
        x = self.ff2(x) + x
        #return x, m, mask, msa_mask
        #return x, mask
        return x
'''
class Evoformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([EvoformerBlock(**kwargs) for _ in range(depth)])

    def forward(
        self,
        x,
        #m,
        mask = None,
        #msa_mask = None
    ):
        #inp = (x, m, mask, msa_mask)
        inp = (x, mask)
        #x, m, *_ = checkpoint_sequential(self.layers, 1, inp)
        x, *_ = checkpoint_sequential(self.layers, 1, inp)
        #return x, m
        return x
'''
