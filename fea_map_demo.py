import datetime
import os
import time
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import PromptSID.archs.common as common
from ldm.ddpm import DDPM
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import argparse
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class GateModel(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GateModel, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        #print(dim, ffn_expansion_factor)

        self.encode = nn.Conv2d(dim, hidden_features, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True))

        self.sg = SimpleGate()

        self.out = nn.Conv2d(hidden_features // 2, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        sca = self.sca(x)
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        sca1,sca2 = sca.chunk(2,dim=1)
        x = x*k_v1*sca1+k_v2*sca2 
        #print(x.shape)  
        x = self.encode(x)
        x = self.sg(x)
        x = self.out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True))
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x,k_v):
        b,c,h,w = x.shape
        save_tensor_as_images(x,"results/pre")
        sca = self.sca(x)
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        sca1,sca2 = sca.chunk(2,dim=1)
        x = x*k_v1*sca1+k_v2*sca2
        save_tensor_as_images(x,"results/after")
        time.sleep(60)
        #print(k_v1)
        #time.sleep(60)  

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.gate = GateModel(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v=y[1]
        x = x + self.attn(self.norm1(x),k_v)
        x = x + self.gate(self.norm2(x),k_v)

        return [x,k_v]

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class DIRformer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [6,8], 
        num_refinement_blocks = 4,
        n_feats = 64,
        heads = [4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(DIRformer, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img,k_v):
        inp_enc_level1 = self.patch_embed(inp_img)
        #save_tensor_as_images(inp_enc_level1, "./results/inp_enc_level1")
        #save_feature(inp_enc_level1, "./results/inp_enc_level1")
        out_enc_level1,_ = self.encoder_level1([inp_enc_level1,k_v])
        #save_tensor_as_images(out_enc_level1, "./results/out_enc_level1")

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2,_ = self.encoder_level2([inp_enc_level2,k_v])

        up_level2 = self.up2_1(out_enc_level2)
        inp_dec_level1 = torch.cat([up_level2, out_enc_level1], 1)
        out_dec_level1,_ = self.decoder_level1([inp_dec_level1,k_v])

        out_dec_level1,_ = self.refinement([out_dec_level1,k_v])
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1 + inp_img
    
class encoder(nn.Module):
    def __init__(self,in_channels = 3,n_feats = 96, n_encoder_res = 6):
        super(encoder, self).__init__()
        E1=[nn.Conv2d(32*in_channels, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x,gt):
        gt0 = self.pixel_unshuffle(gt)
        x0 = self.pixel_unshuffle(x)
        x = torch.cat([x0, gt0], dim=1)
        #print(x.shape)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        S1_IPR = []
        fea1 = self.mlp(fea)
        S1_IPR.append(fea1)
        return fea1,S1_IPR
    
class CPEN(nn.Module):
    def __init__(self,in_c = 3, n_feats = 96, n_encoder_res = 6):
        super(CPEN, self).__init__()
        E1=[nn.Conv2d(16*in_c, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            common.ResBlock(
                common.default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E=E1+E2+E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
    def forward(self, x):
        x = self.pixel_unshuffle(x)
        print(x.shape)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 96, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea 

#@ARCH_REGISTRY.register()
class attendiffneighbor(nn.Module):
    def __init__(self,         
        n_encoder_res=6,         
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [6,8],
        heads = [4,8], 
        num_refinement_blocks = 4,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        n_feats = 64,
        n_denoise_res = 1, 
        linear_start= 0.1,
        linear_end= 0.99, 
        timesteps = 4 ):
        super(attendiffneighbor, self).__init__()

        # Generator
        self.G = DIRformer(        
        inp_channels=inp_channels, 
        out_channels=out_channels, 
        dim = dim,
        num_blocks = num_blocks,
        heads=heads, 
        n_feats=n_feats,
        ffn_expansion_factor = ffn_expansion_factor,
        num_refinement_blocks = num_refinement_blocks,
        bias = bias,
        LayerNorm_type = LayerNorm_type   ## Other option 'BiasFree'
        )
        self.encoder = encoder(n_feats=n_feats, in_channels=inp_channels,n_encoder_res=n_encoder_res)

        self.condition = CPEN(n_feats=n_feats, in_c=inp_channels,n_encoder_res=n_encoder_res)

        self.denoise= denoise(n_feats=n_feats, n_denoise_res=n_denoise_res,timesteps=timesteps)

        self.diffusion = DDPM(denoise=self.denoise, condition=self.condition ,n_feats=n_feats,linear_start= linear_start,
  linear_end= linear_end, timesteps = timesteps)

    def forward(self, img0, img, IPRS1=None):
        if self.training:
            IPRS2, pred_IPR_list=self.diffusion(img0,IPRS1)
            sr = self.G(img, IPRS2)
            return sr, pred_IPR_list
        else:
            IPRS2=self.diffusion(img)
            sr = self.G(img, IPRS2)
            return sr

def save_feature(tensor,output_dir):
    """
    Save each channel of a tensor as a separate PNG image.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape [1, n, 256, 256].
    output_dir (str): The directory where the images will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Remove the batch dimension
    tensor = tensor.squeeze(0)
    tensor = tensor.mean(dim=0).squeeze(0)

    # Normalize the tensor to [0, 255]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor.to(torch.float32)

    channel = tensor.cpu().detach().numpy()
    fig, ax = plt.subplots()
    
    # Display the image with jet colormap
    cax = ax.imshow(channel, cmap='jet', vmin=0, vmax=1)
    
    # Add a colorbar
    fig.colorbar(cax)
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f'channel.png'))
    
    # Close the figure to free up memory
    plt.close(fig)

def save_tensor_as_images(tensor,output_dir):
    """
    Save each channel of a tensor as a separate PNG image.

    Parameters:
    tensor (torch.Tensor): The input tensor of shape [1, n, 256, 256].
    output_dir (str): The directory where the images will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Remove the batch dimension
    tensor = tensor.squeeze(0)

    # Normalize the tensor to [0, 255]
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = tensor.to(torch.float32)

    # Save each channel as a PNG image
    for i in range(tensor.size(0)):
        channel = tensor[i].cpu().detach().numpy()
        """ image = Image.fromarray(channel)
        image.save(os.path.join(output_dir, f'channel_{i}.png')) """
        fig, ax = plt.subplots()
        
        # Display the image with jet colormap
        cax = ax.imshow(channel, cmap='jet', vmin=0, vmax=1)
        
        # Add a colorbar
        fig.colorbar(cax)
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'channel_{i}.png'))
        
        # Close the figure to free up memory
        plt.close(fig)
        

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./*.pth')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--gpu_devices', default='0', type=str)
opt, _ = parser.parse_known_args()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
operation_seed_counter = 0
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_devices
torch.set_num_threads(8)

network = attendiffneighbor(
  n_encoder_res=4,
  inp_channels = 3,
  out_channels = 3,
  dim = 48,
  num_blocks = [6,8],
  num_refinement_blocks = 4,
  heads = [4,8],
  ffn_expansion_factor = 2,
  bias = False,
  LayerNorm_type = 'WithBias',
  n_feats=64,
  n_denoise_res=1,
  linear_start=0.1,
  linear_end=0.99,
  timesteps=4)

if opt.parallel:
    network = torch.nn.DataParallel(network)

checkpoint = torch.load(opt.checkpoint)
network.load_state_dict(checkpoint['params'])
network = network.cuda()
network.eval()

input_dir = '/data/lhq23/datasets/000-008_noisy.jpg'
im = Image.open(input_dir)
im = np.array(im, dtype=np.float32)
im = im[np.newaxis, :, :, :]
# np.ndarray to torch.tensor
im = torch.from_numpy(im)/255.0
im = im.permute(0,3,1,2).cuda()
print(im.shape)
out = network(im,im)
