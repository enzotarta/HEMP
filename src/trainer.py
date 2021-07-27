from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import time
from src.HEMP_core import *
from torch.autograd import grad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.utilities import *
from src.quantization import *

def train(args, model, device, train_loader, optimizer, epoch, criterion, q_model, voronoi, test_loader, models_dir, quantized_dir, scaler, sensitivity):#(args, model, device, train_loader, optimizer, epoch, N, criterion, q_model, voronoi,entropy_order):
	model.train()
	print("Training epoch #{}...".format(epoch))
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')
	progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))
	tk0 = tqdm(train_loader, total=int(len(train_loader)))
	end = time.time();
	H_grad=None; voronoi_scale=None; grad_RMSE=None; RMSE=None
	for batch_idx, (data, target) in enumerate(tk0):
		if batch_idx % args.newregu_every == 0:
			q_model.load_state_dict(deepcopy(model.state_dict()))
			voronoi = LM_quantize_model(q_model, args.N, device, args.lloydmax, new_voronoi=False, voronoi = voronoi)
			if args.lamb_H != 0.0:
				if args.entropy_order==2:
					with torch.no_grad():
						H_grad = H2_grad(model, args.N, voronoi, device, order = 2, mode = args.mode)
						for key in H_grad:
							H_grad[key] = torch.nan_to_num(H_grad[key], nan=0.0)
				else:
					H_grad = LM_entropy_grads(args, model, voronoi, device)
					for idx in range(len(H_grad)):
						if H_grad[idx] is not None:
							H_grad[idx].nan_to_num(nan=0.0)
			if args.lamb_RMSE != 0.0: 
				grad_RMSE, RMSE, voronoi_scale = qerror_RMSE_update(model, q_model, voronoi, device)


		data_time.update(time.time() - end)
		data, target = data.to(device), target.to(device)
		loss, output = update_step(model, data, target, H_grad, device, criterion, q_model, optimizer, args, voronoi, scaler, grad_RMSE, sensitivity)

		batch_time.update(time.time() - end)
		end = time.time()
		loss = criterion(output, target)
		# measure accuracy and record loss
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), data.size(0))
		top1.update(acc1[0], data.size(0))
		top5.update(acc5[0], data.size(0))
		tk0.set_postfix(loss = losses.avg, RMSE = RMSE, top1 = top1.avg.item(), top5 = top5.avg.item())
	del H_grad
	print("Training epoch #{} done!".format(epoch))
	return voronoi

def test(model, test_loader, device):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = torch.nn.CrossEntropyLoss()(output, target)
			test_loss += loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	return test_loss.item()/ 100.0, 100. * correct / len(test_loader.dataset)

def get_statistics_csv(args, model, test_loader, device, voronoi, epoch, report_name, q_model):
	print(epoch, "\t Generating quantized model...")
	q_model.load_state_dict(deepcopy(model.state_dict()))
	voronoi = LM_quantize_model(q_model, args.N, device, args.lloydmax, new_voronoi=args.bound_recompute, voronoi = voronoi)
	
	if args.histograms:
		print("Generating histograms...")
		plot_w_dist(model, q_model, voronoi, "histograms/" + report_name + "/", epoch)
	f = open(report_name + ".csv", "a")
	print("Computing entropy...")
	with torch.no_grad():
		if args.entropy_order==1:
			H = LM_entropy_quantization_1st(model, args.N, voronoi, device).item()
			H_hat = LM_entropy_quantization_1st(q_model, args.N, voronoi, device).item()
		elif args.entropy_order==2:
			H = (H2_compute(model, args.N, voronoi, device, mode = args.mode)/2).item()
			H_hat = (H2_compute(q_model, args.N, voronoi, device, mode = args.mode)/2).item()
		else:
			H = LM_entropy_quantization_nth(model, args.N, voronoi, device, order=args.entropy_order, mode = args.mode).item()/args.entropy_order
			H_hat = LM_entropy_quantization_nth(q_model, args.N, voronoi, device, order=args.entropy_order, mode = args.mode).item()/args.entropy_order
	print("Evaluating test set performance...")
	tl, ta = test(model, test_loader, device)
	quantized_loss, quantized_acc = test(q_model, test_loader, device)

	f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(epoch, tl, quantized_loss, ta, quantized_acc, H, H_hat))
	f.close()
	print(tl, ta, quantized_acc, H, H_hat)
	return voronoi

def compute_voronoi_scale_RMSE(model, q_sd, voronoi, device):
	voronoi_scale = {}
	with torch.no_grad():
		for n,p in model.named_parameters():
			if voronoi[n] != None:
				centers = voronoi[n][0]
				this_mask = torch.zeros(q_sd[n].shape, device = device)
				for i in np.arange(1, len(centers)-1):
					delta = voronoi[n][2][i-1]
					if delta != 0:###because all the layer's variance is 0
						this_mask += (centers[i] == q_sd[n]).type(torch.float) / voronoi[n][2][i-1]
				voronoi_scale[n] = this_mask
	return voronoi_scale

def qerror_RMSE_update(model, q_model, voronoi, device, voronoi_scale = None, relative = False):
	arg = 0.0
	totel_model = 0
	q_sd = q_model.state_dict()
	deriv_RMSE = {}
	if voronoi_scale == None and (relative):
		voronoi_scale = compute_voronoi_scale_RMSE(model, q_sd, voronoi, device)
	with torch.no_grad():
		for n,p in model.named_parameters():
			if voronoi[n] != None:
				centers = voronoi[n][0]
				this_distance = p - q_sd[n]
				if relative:
					arg += torch.sum(((this_distance)*voronoi_scale[n])**2)
					if n in deriv_RMSE:
						deriv_RMSE[n] = deriv_RMSE[n] + ((this_distance)*(voronoi_scale[n] ** 2))
					else:
						deriv_RMSE[n] = this_distance*(voronoi_scale[n] ** 2)
				else:
					arg += torch.sum(this_distance ** 2)
					if n in deriv_RMSE:
						deriv_RMSE[n] = deriv_RMSE[n] + this_distance
					else:
						deriv_RMSE[n] = this_distance

				totel_model += p.data.numel()
	total = torch.sqrt(arg/totel_model)
	return deriv_RMSE, total.item(), voronoi_scale

def update_step(model, data, target, H_grads, device, loss_metrics, q_model, opt, args, voronoi, scaler,grad_RMSE, sensitivity):
	output = model(data)
	loss = loss_metrics(output, target)
	opt.zero_grad()
	loss.backward()
	opt.step()
	sensitivity.update(model)
	idx = 0
	for n, p in model.named_parameters():
		if voronoi[n] != None:
			S_bar = sensitivity.get_Sbar(n)
			if (args.lamb_H != 0.0):
				if args.entropy_order==2:
					p.data.add_(-S_bar * args.lamb_H * H_grads[n])
				else:
					p.data.add_(-S_bar * args.lamb_H * H_grads[idx])
			if args.lamb_RMSE != 0.0:
				p.data.add_(-args.lamb_RMSE * S_bar * grad_RMSE[n])
		idx += 1
	for n, p in model.named_parameters():
		if voronoi[n] != None:
			p.data.copy_(torch.clamp(p.data, min = voronoi[n][1][0]+voronoi[n][2][0]/2.01, max = voronoi[n][1][-1] - voronoi[n][2][-1]/2.01))
	return loss, output