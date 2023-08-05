import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
# from pdb import set_trace as stx
from torch.nn import init
import numpy as np
import numbers
from einops import rearrange

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)  # 计算注意力权重
        y = self.conv_du(y)
        return x * y  # 加权


def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs


class FFTInteraction_N(nn.Module):
    def __init__(self, in_nc,out_nc):
        super(FFTInteraction_N,self).__init__()
        self.post = nn.Conv2d(2*in_nc,out_nc,1,1,0)
        self.mid = nn.Conv2d(in_nc,in_nc,3,1,1,groups=in_nc)


    def forward(self,x,x_enc,x_dec):
        x_enc = torch.fft.rfft2(x_enc, norm='backward')
        x_dec = torch.fft.rfft2(x_dec, norm='backward')
        x_freq_amp = torch.abs(x_enc)
        x_freq_pha = torch.angle(x_dec)
        x_freq_pha = self.mid(x_freq_pha)
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)
        x_recom = torch.fft.irfft2(x_recom)
    
        out = self.post(torch.cat([x_recom,x],1))


        return out
 

class Encoder(nn.Module):
    def __init__(self, n_feat,use_csff=False,depth=4):
        super(Encoder, self).__init__()
        self.body=nn.ModuleList()#[]
        self.depth=depth
        for i in range(depth-1):
            self.body.append(UNetConvBlock(in_size=n_feat*2**(i), out_size=n_feat*2**(i+1), downsample=True, use_csff=use_csff,depth=i))
        
        self.body.append(UNetConvBlock(in_size=n_feat*2**(depth-1), out_size=n_feat*2**(depth-1), downsample=False,  use_csff=use_csff,depth=depth))
        self.shift_size=4

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res=[]
        if encoder_outs is not None and decoder_outs is not None:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    res.append(x)
                    x = down(x,encoder_outs[i],decoder_outs[-i-1])
                else:
                    x = down(x)
        else:
            for i,down in enumerate(self.body):
                if (i+1) < self.depth:
                    res.append(x)
                    x = down(x)
                else:
                    x = down(x)
        return res,x

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,depth):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.test = nn.Identity()
        self.window_size = [8,8]
        self.depth = depth
        self.shift_size = self.window_size[0]//2


    def forward(self, x):
        if (self.depth)%2:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        b,c,h,w = x.shape
        #print(x.shape)
        w_size = self.window_size
        x = rearrange(x, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=w_size[0], b1=w_size[1])
        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        attn = self.test(attn)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=w_size[1], w=w_size[1])

        out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0)  (w b1)', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])
        out = self.project_out(out)
        if (self.depth)%2:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,depth):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,depth)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, use_csff=False,depth=1):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.use_csff = use_csff      
        
        self.block = TransformerBlock(in_size,num_heads=4,ffn_expansion_factor = 2.66,bias=False, LayerNorm_type='WithBias',depth = depth)
        

        if downsample and use_csff:
            self.stage_int = FFTInteraction_N(in_size,in_size)

        if downsample:
            self.downsample = conv_down(in_size, out_size, bias=False)


    def forward(self, x, enc=None, dec=None):
        out = x
        if enc is not None and dec is not None:
            assert self.use_csff
           
            out = self.stage_int(out,enc,dec)
        out = self.block(out)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size,depth):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv = nn.Conv2d(out_size*2, out_size, 1, bias=False)
        self.conv_block = UNetConvBlock(out_size, out_size, False,depth=depth)

    def forward(self, x, bridge):
        up = self.up(x)
        out = self.conv(torch.cat([up, bridge],dim=1))
        out = self.conv_block(out)
        return out

class Decoder(nn.Module):
    def __init__(self, n_feat,depth=4):
        super(Decoder, self).__init__()
        
        self.body=nn.ModuleList()
        self.skip_conv=nn.ModuleList()#[]
        self.shift_size = 4
        self.depth = depth
        for i in range(depth-1):
            self.body.append(UNetUpBlock(in_size=n_feat*2**(depth-i-1), out_size=n_feat*2**(depth-i-2),depth=depth-i-1))

            #self.skip_conv.append(nn.Conv2d(n_feat*(depth-i), n_feat*(depth-i-i), 1))
            
    def forward(self, x, bridges):

        res=[]
        for i,up in enumerate(self.body):
            x=up(x,bridges[-i-1])
            res.append(x)

        return res,x


def default_conv(in_channels, out_channels, kernel_size,stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2),stride=stride, bias=bias)

class SAM(nn.Module):
    def __init__(self, in_c,bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(in_c, in_c, 3,bias=bias)
        self.conv2 = conv(in_c, in_c, 1, bias=bias)
        self.conv3 = nn.Conv2d(in_c*2, in_c, 1, bias=bias)
        self.ca = CALayer(in_c,reduction=8)

    def forward(self, x, Phi):
        x_phi = self.conv1(Phi)
        img = torch.cat([self.conv2(x), x_phi],dim=1)
        r_value = self.ca(self.conv3(img))
        return r_value      
    
class DegraBlock(nn.Module):
    def __init__(self,in_c=28):
        super(DegraBlock, self).__init__()
        self.attention = SAM(in_c=28)

    def forward(self, x_i,y,Phi,Phi_s):
        # compute r_k
        yb = A(x_i, Phi)
        r = self.attention(x_i,Phi)
        #print(r.shape)
        r_i = x_i + r* At(torch.div(y - yb, Phi_s), Phi)
        return r_i     

class MPRBlock(nn.Module):
    def __init__(self, n_feat=80,n_depth = 3 ):
        super(MPRBlock, self).__init__()

        # Cross Stage Feature Fusion (CSFF)
        self.stage_encoder = Encoder(n_feat, use_csff=True,depth=n_depth)
        self.stage_decoder = Decoder(n_feat,depth=n_depth)

        self.gdm = DegraBlock()

    def forward(self, stage_img, y,f_encoder,f_decoder,Phi,Phi_s):
        b,c,w,h = stage_img.shape
        crop = stage_img.clone()
        
        x_k_1=shift_3d(crop[:,:,:256,:310])
        # compute r_k
        r_k = self.gdm(x_k_1,y,Phi,Phi_s)
        
        r_k = shift_back_3d(r_k)
        b, c, h_inp, w_inp = r_k.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r_k = F.pad(r_k, [0, pad_w, 0, pad_h], mode='reflect')

        feat1, f_encoder = self.stage_encoder(r_k, f_encoder, f_decoder)
        ## Pass features through Decoder of Stage 2
        
        f_decoder,last_out = self.stage_decoder(f_encoder, feat1)

        stage_img = last_out+r_k

        return stage_img,feat1 ,f_decoder

##########################################################################
class PADUT(nn.Module):
    def __init__(self, in_c=3,  n_feat=32,  nums_stages=5,n_depth=3):
        super(PADUT, self).__init__()

        
        self.body = nn.ModuleList()
        self.nums_stages = nums_stages

        self.shallow_feat = nn.Conv2d(in_c, n_feat, 3,1,1 ,bias=True)
        self.stage_model= nn.ModuleList([MPRBlock(
                n_feat=n_feat,n_depth=n_depth
            ) for _ in range(self.nums_stages)])
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)

        # Cross Stage Feature Fusion (CSFF)
        self.stage1_encoder = Encoder(n_feat, use_csff=False,depth=n_depth)
        self.stage1_decoder = Decoder(n_feat,depth=n_depth)

        self.gdm = DegraBlock()

    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        return z
    
    def forward(self, y,input_mask=None):
        output_=[]
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        # y,input_mask = y[0],y[1] 
        # y = y.squeeze(0)
        b, h_inp, w_inp = y.shape
        Phi,Phi_s = input_mask  #28 256 310

        x_0 = self.initial(y,Phi)
        r_0 = self.gdm(x_0,y,Phi,Phi_s)
        r_0 = shift_back_3d(r_0)
        
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        r_0 = F.pad(r_0, [0, pad_w, 0, pad_h], mode='reflect')
        # compute x_k

        x = self.shallow_feat(r_0)

        feat1, f_encoder = self.stage1_encoder(x)
        ## Pass features through Decoder of Stage 1
        f_decoder,last_out = self.stage1_decoder(f_encoder,feat1)

        stage_img= last_out+r_0
        output_.append(stage_img)

        ##-------------------------------------------
        ##-------------- Stage 2_k-1---------------------
        ##-------------------------------------------
        for i in range(self.nums_stages): 
            stage_img, feat1, f_decoder = self.stage_model[i](stage_img, y,feat1,f_decoder,Phi,Phi_s)
            output_.append(stage_img)

        return output_[-1][:,:,:256,:256]
