import torch
from src.lloydmax_cpu import LloydMaxLayerQuantize, LloydMaxQuantizer
import numpy as np

def UniformLayerQuantize(x, bits, device):
	mid_tr = True
	num_repre  = bits
	total_delta = (x.max()-x.min()).item()

	if mid_tr:
		step = ((total_delta)/(num_repre-1))
		repre = np.arange(x.min().item(), x.max().item() + step/2.0, step)
	else:
		step = ((total_delta)/(num_repre))
		repre = np.arange(x.min().item(), x.max().item()+ step/2.0, step)

	thre = np.zeros(len(repre)-1)
	for i in range(len(repre)-1):
		thre[i] = 0.5*(repre[i]+repre[i+1])

	thre = np.insert(thre, 0, repre[0] - abs(thre[0]-repre[0]))
	thre = np.append(thre, repre[-1] + abs(thre[-1]-repre[-1]))
	#add prohibited bins
	repre = np.insert(repre, 0, -abs(repre[0]-repre[1]) + repre[0])
	repre = np.append(repre, abs(repre[-2]-repre[-1])+repre[-1])

	delta = np.zeros(len(thre)-1)
	for i in range(len(thre)-1):
		delta[i] = thre[i+1] - thre[i]

	voronoi = (repre, thre, delta)
	return quant_gpu(x, thre, repre, device), voronoi


def quant_gpu(float_tensor, voronoi_bounds, voronoi_centers, device):
	quant_tensor = torch.zeros(float_tensor.shape, device = device)*voronoi_centers[-1]
	quant_tensor += (float_tensor <= voronoi_centers[0]).type(torch.float) * voronoi_centers[0]
	for idx in np.arange(1, len(voronoi_centers)-1):
		quant_tensor += ((float_tensor > voronoi_bounds[idx-1]) * (float_tensor <= voronoi_bounds[idx])).type(torch.float) * voronoi_centers[idx]
	return quant_tensor

def LM_quantize_model(model, bits, device, lloydmax_flag, new_voronoi=True, voronoi = None):
	flag_vcnew = False
	if voronoi==None:
		voronoi = {}
		flag_vcnew = True
	for n,p in model.named_parameters():
		if p.data.numel() > (bits)*10:
			if new_voronoi:
				if flag_vcnew:
					if lloydmax_flag:
						this_quantized, this_voronoi = LloydMaxLayerQuantize(p.data, bits, device)
					else:
						this_quantized, this_voronoi = UniformLayerQuantize(p.data, bits, device)
				else:
					if lloydmax_flag:
						this_quantized, this_voronoi = LloydMaxLayerQuantize(p.data, bits, device, voronoi = voronoi[n])
					else:
						this_quantized, this_voronoi = UniformLayerQuantize(p.data, bits, device)

				p.data.copy_(this_quantized)
				voronoi[n] = this_voronoi
				del this_quantized
			else:
				this_quantized = quant_gpu(p.data, voronoi[n][1], voronoi[n][0], device)
				p.data.copy_(this_quantized)
				del this_quantized
		else:
			voronoi[n] = None
	return voronoi