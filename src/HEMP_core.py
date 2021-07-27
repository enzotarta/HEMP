import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch.autograd import grad

def LM_entropy_quantization_1st(model, N, voronoi, device):
	totel_model = 0
	bin_num = N + 2
	H = 0.0
	px = torch.zeros(bin_num, device=device)
	for n,p in model.named_parameters():
		if voronoi[n] != None:
			layer_px = torch.zeros((p.data.numel(), bin_num), device=device)
			p_flatten = p.view(-1)
			for i in range(bin_num):
				#first with the previous
				if i != 0:
					rel_dist = voronoi[n][0][i] - voronoi[n][0][i-1]
					mask = ((p_flatten < voronoi[n][0][i]) * (p_flatten >= voronoi[n][0][i-1])).type(torch.float)
					layer_px[:, i] += (torch.clamp(1.0 - ((voronoi[n][0][i] - p_flatten) / rel_dist), min = 1e-20) * mask)
				#then with the next
				if i != bin_num-1:
					rel_dist = voronoi[n][0][i+1] - voronoi[n][0][i]
					mask = ((p_flatten >= voronoi[n][0][i]) * (p_flatten < voronoi[n][0][i+1])).type(torch.float)
					layer_px[:, i] += (torch.clamp(1.0 - ((p_flatten - voronoi[n][0][i]) / rel_dist), min=1e-20) * mask)
			px += torch.sum(layer_px, dim=0)
	px = px/torch.sum(px)
	return -torch.sum(px * torch.log2(px+1e-10))

def LM_entropy_quantization_nth(model, N, voronoi, device, order = 5, mode = 0):###for order > 1!!!
	bin_num = N+2
	px = torch.zeros((bin_num**order), device=device)
	for n,p in model.named_parameters():
		#if 'weight' in n:
		if voronoi[n] != None:
			layer_px = torch.zeros((p.data.numel(), bin_num), device=device)
			if (len(p.data.shape) > 1):
				if ('conv' in n):
					if mode == 1:
						p_flatten = p.transpose(2,3)##entropy computed through neurons
					elif mode == 2:
						p_flatten = p.transpose(1,3)##entropy computed through neurons
					elif mode == 3:
						p_flatten = p.transpose(0,3)##entropy computed through neurons
					else:
						p_flatten = p
					p_flatten = torch.reshape(p_flatten, (-1,))
				elif (len(p.shape) == 2) and (mode == 3):
					p_flatten = p.transpose(0,1)
					p_flatten = torch.reshape(p_flatten, (-1,))
				else:
					p_flatten = torch.reshape(p, (-1,))
			else:
				p_flatten = p
			limit = int(np.floor(p.data.numel()/order))

			for i in range(bin_num):
				#first with the previous
				if i != 0:#I am in the first bin, so no contribution from the prev bin
					rel_dist = voronoi[n][0][i] - voronoi[n][0][i-1]
					mask = ((p_flatten < voronoi[n][0][i]) * (p_flatten >= voronoi[n][0][i-1])).type(torch.float)
					layer_px[:, i] += (1.0 - (voronoi[n][0][i] - p_flatten) / rel_dist) * mask
				#then with the next
				if i != bin_num-1:
					rel_dist = voronoi[n][0][i+1] - voronoi[n][0][i]
					mask = ((p_flatten >= voronoi[n][0][i]) * (p_flatten < voronoi[n][0][i+1])).type(torch.float)
					layer_px[:, i] += (1.0 - (p_flatten - voronoi[n][0][i]) / rel_dist) * mask

			#compute 2nd order
			this_layer_px = layer_px.clone()
			layer_px_1 = torch.as_strided(this_layer_px, [limit, bin_num], [bin_num*order, 1], storage_offset = 0).reshape(limit, bin_num, 1)
			layer_px_2 = torch.as_strided(this_layer_px, [limit, bin_num], [bin_num*order, 1], storage_offset = bin_num).reshape(limit, 1, bin_num)

			px_aux = torch.einsum('bij,bjk->bik', layer_px_1, layer_px_2).reshape(limit,-1,1)

			for ord in np.arange(2, order):#the computed order is ord+1
				this_px_aux = torch.as_strided(this_layer_px, [limit, bin_num], [bin_num *order, 1], storage_offset = bin_num*int(ord)).reshape(limit, 1, bin_num)
				px_aux = torch.einsum('bij,bjk->bik', px_aux, this_px_aux).reshape(limit,-1,1)
			px += torch.sum(px_aux.squeeze(dim=2), dim=0)
	px = px/torch.sum(px)
	return -torch.sum(px * torch.log2(torch.clamp(px, min=1e-10)))

def LM_entropy_grads(args, model, voronoi, device):
	H = 0
	if args.entropy_order==1:
		H = args.lamb_H * LM_entropy_quantization_1st(model, args.N, voronoi,device)
	else:
		H = args.lamb_H * LM_entropy_quantization_nth(model, args.N, voronoi, device, order=args.entropy_order)
	grads = grad(H, model.parameters(), create_graph=False, allow_unused=True)
	del H
	return grads



def p_compute_core(input_data, voronoi, bin_num, order, device):
	bin_num = len(voronoi[0])
	layer_px = torch.zeros(input_data.numel(), bin_num, device=device)
	b_plus = torch.zeros(input_data.numel(), device=device, dtype = torch.long)
	limit = int(np.floor(input_data.numel()/order))
	for i in range(bin_num):
		if i != 0:#I am in the first bin, so no contribution from the prev bin
			rel_dist = voronoi[0][i] - voronoi[0][i-1]
			mask = ((input_data < voronoi[0][i]) * (input_data >= voronoi[0][i-1])).type(torch.float)
			layer_px[:, i] += (torch.clamp(1.0 - ((voronoi[0][i] - input_data) / rel_dist), min = 1e-20) * mask)
			b_plus += (mask * i).type(torch.int)
			del mask
		#then with the next
		if i != bin_num-1:
			rel_dist = voronoi[0][i+1] - voronoi[0][i]
			mask = ((input_data >= voronoi[0][i]) * (input_data < voronoi[0][i+1])).type(torch.float)
			layer_px[:, i] += (torch.clamp(1.0 - ((input_data - voronoi[0][i]) / rel_dist), min=1e-20) * mask)
			del mask
	#compute 2nd order
	layer_px_cloned = layer_px.clone()
	layer_px_1 = torch.as_strided(layer_px_cloned, [limit, bin_num], [bin_num*2, 1], storage_offset = 0).reshape(limit, bin_num, 1)#layer_px[idx,:].reshape(limit, bin_num, 1)###OCCHIO ALL'ORDINE DI QUESTO RESHAPE!
	layer_px_2 = torch.as_strided(layer_px_cloned, [limit, bin_num], [bin_num*2, 1], storage_offset = bin_num).reshape(limit, 1, bin_num)#layer_px[idx2,:].reshape(limit, 1, bin_num)

	px_aux = torch.einsum('bij,bjk->bik', layer_px_1, layer_px_2)
	del layer_px_1
	del layer_px_2
	del layer_px
	del layer_px_cloned

	these_px_ord = torch.sum(px_aux, dim=0, dtype = torch.float)
	del px_aux
	return these_px_ord, b_plus

def p_compute(model, N, voronoi, device, order = 2, mode = 0):
	bin_num = N + 2
	px = torch.zeros((bin_num, bin_num), device=device, dtype = torch.float)
	b_plus={}
	num_pams = 0
	limit_memory_const = int(2**(16))
	for n,p in model.named_parameters():
		#if 'weight' in n:
		if voronoi[n] != None:
			if (len(p.data.shape) > 1):
				if ('conv' in n):
					if mode == 1:
						p_flatten = p.transpose(2,3)##entropy computed through neurons
					elif mode == 2:
						p_flatten = p.transpose(1,3)##entropy computed through neurons
					elif mode == 3:
						p_flatten = p.transpose(0,3)##entropy computed through neurons
					else:
						p_flatten = p
					p_flatten = torch.reshape(p_flatten, (-1,))
				elif (len(p.shape) == 2) and (mode == 3):
					p_flatten = p.transpose(0,1)
					p_flatten = torch.reshape(p_flatten, (-1,))
				else:
					p_flatten = torch.reshape(p, (-1,))
			else:
				p_flatten = p
			num_pams += p_flatten.numel()
			if len(p_flatten) >limit_memory_const:
				global_b_plus = torch.zeros(p_flatten.numel(), device=device, dtype = torch.long)
				idx_ranges = np.append(range(0, len(p_flatten)-2, limit_memory_const), len(p_flatten))
				for idx in range(len(idx_ranges)-1):
					this_px, this_b_plus = p_compute_core(p_flatten[idx_ranges[idx]: idx_ranges[idx+1]], voronoi[n], bin_num, order, device)
					global_b_plus[idx_ranges[idx]: idx_ranges[idx+1]] = this_b_plus
					px += this_px
					del this_px
					del this_b_plus
				b_plus[n] = global_b_plus
			else:
				this_px, this_b_plus = p_compute_core(p_flatten, voronoi[n], bin_num, order, device)
				b_plus[n] = this_b_plus
				px += this_px
				del this_px
				del this_b_plus
			del p_flatten
	px = px/torch.sum(px)
	return px, b_plus, num_pams

def H2_compute(model, N, voronoi, device, mode = 0):
	px, b_plus, num_pams = p_compute(model, N, voronoi, device, order = 2, mode = mode)
	H = -torch.sum(px*torch.log2(px + 1e-30))
	del px
	del b_plus
	return H

def H2_grad_core(px, centers, this_b_plus_1, this_b_plus_2, num_pams):
	b_plus_plus = px[this_b_plus_1, this_b_plus_2]
	b_plus_minus = px[this_b_plus_1, this_b_plus_2-1]
	b_minus_plus = px[this_b_plus_1-1, this_b_plus_2]
	b_minus_minus = px[this_b_plus_1-1, this_b_plus_2-1]
	Delta_1 = centers[this_b_plus_1] - centers[this_b_plus_1 - 1]
	Delta_2 = centers[this_b_plus_2] -centers[this_b_plus_2 - 1]
	log_arg = (b_minus_plus * b_minus_minus)/(b_plus_plus * b_plus_minus)
	log_term = torch.log2(log_arg)
	grad_1 = ((1.0/(2.0*Delta_1*num_pams))*log_term).unsqueeze(dim=1)
	del log_arg
	del log_term
	log_arg = (b_plus_minus * b_minus_minus)/(b_plus_plus * b_minus_plus)
	log_term = torch.log2(log_arg)
	grad_2 = ((1.0/(2.0*Delta_2*num_pams))*log_term).unsqueeze(dim=1)
	del log_arg
	del log_term
	del Delta_1
	del Delta_2
	del b_plus_plus
	del b_plus_minus
	del b_minus_plus
	del b_minus_minus
	total_grad = torch.cat((grad_1, grad_2), 1).reshape(-1)
	del grad_1
	del grad_2
	return total_grad

def H2_grad(model, N, voronoi, device, order = 2, mode = 0):
	px, b_plus, num_pams = p_compute(model, N, voronoi, device, order = order, mode = mode)
	H2_grad = {}
	idx = 0
	limit_memory_const = int(2**(32))
	for n,p in model.named_parameters():
		if voronoi[n] != None:
			centers = torch.from_numpy(voronoi[n][0]).to(device)
			this_b = b_plus[n].clone()
			if p.data.numel() >limit_memory_const:
				layer_grad = torch.zeros(p.data.numel(), device=device, dtype = torch.float)
				idx_ranges = np.append(range(0, p.data.numel()-4,  limit_memory_const), p.data.numel())
				for idx in range(len(idx_ranges)-1):
					this_b_idx = this_b[idx_ranges[idx]: idx_ranges[idx+1]].clone()
					this_b_plus_1 = torch.as_strided(this_b_idx, [int(this_b_idx.numel()/2)], [2], storage_offset = 0)
					this_b_plus_2 = torch.as_strided(this_b_idx, [int(this_b_idx.numel()/2)], [2], storage_offset = 1)
					layer_grad[idx_ranges[idx]: idx_ranges[idx+1]] = H2_grad_core(px, centers, this_b_plus_1, this_b_plus_2, num_pams)
					del this_b_plus_1
					del this_b_plus_2
					del this_b_idx
				H2_grad[n] = layer_grad.reshape(p.data.shape)
			else:
				this_b_plus_1 = torch.as_strided(this_b, [int(this_b.numel()/2)], [2], storage_offset = 0)
				this_b_plus_2 = torch.as_strided(this_b, [int(b_plus[n].numel()/2)], [2], storage_offset = 1)
				H2_grad[n] = (H2_grad_core(px, centers, this_b_plus_1, this_b_plus_2, num_pams)).reshape(p.data.shape)
				del this_b_plus_1
				del this_b_plus_2
			del centers
			del this_b
			H2_grad[n] = torch.clamp(torch.nan_to_num(H2_grad[n], nan=0.0), -0.01,0.01)
		idx += 1
	del px
	del b_plus
	return H2_grad
