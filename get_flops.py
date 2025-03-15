from thop import profile  # 确保只导入 profile 函数
import torch
from PromptSID.archs.sttendiff_ablation_arch import DIRformer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = DIRformer(
  inp_channels=3,
  out_channels=3,
  dim=48,
  num_blocks=[6,8],
  heads=[4,8],
  num_refinement_blocks=4,
  bias=False,
  LayerNorm_type='WithBias',
  ffn_expansion_factor=2,
  )  # 定义好的网络模型
if torch.cuda.is_available():
    net.cuda()

input1 = torch.randn(1, 3, 64, 64)#输入到模型时，模型要加一个维度x = x.unsqueeze(0)
input1 = input1.to(device)

flops, params = profile(net,inputs=(input1,))
print('flops: ', flops, 'params: ', params)


