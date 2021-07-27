import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import torchvision.models as models
from copy import deepcopy
from src.trainer import *
from src.models import resnet32, LeNet5, MobileNetV2
from src.lzma_compress import *

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='HEMP')
	parser.add_argument('--batch_size', type=int, default=100, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=1000, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--lr_quant_scale', type=float, default=0.0)
	parser.add_argument('--lamb_H', type=float, default=1.0)
	parser.add_argument('--lamb_RMSE', type=float, default=0.1)
	parser.add_argument('--lamb_LOBSTER', type=float, default=0.0)
	parser.add_argument('--device', default="cpu")
	parser.add_argument('--entropy_order', type=int, default=3)
	parser.add_argument('--newregu_every', type=int, default=10)
	parser.add_argument('--N', type=int, default=4)
	parser.add_argument('--mode', type=int, default=0)
	parser.add_argument('--momentum', type=float, default=0.0, metavar='M', help='SGD momentum (default: 0.5)')
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--datapath', default=f'{os.path.expanduser("~")}/data/')
	parser.add_argument('--histograms', default=False, action='store_true')
	parser.add_argument('--arch', default='mobilenet_v2')
	parser.add_argument('--dataset', default='mnist')
	parser.add_argument('--beg_epoch', type=int, default=0)
	parser.add_argument('--lloydmax', default=True)
	parser.add_argument('--bound_recompute', default=False, action='store_true')
	parser.add_argument('--store_all', default=True)
	parser.add_argument('--amp', default=False, action='store_true')

	args = parser.parse_args()
	device = torch.device(args.device)
	if args.device != "cpu":
		torch.cuda.set_device(device)
	if args.dataset == 'imagenet':
		model = models.__dict__[args.arch](pretrained=True).to(device)
		report_name = "ImageNet_"+args.arch+"_"

		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
		                                 std=[0.229, 0.224, 0.225])

		train_dataset = datasets.ImageFolder(
		    args.datapath + "train/",
		    transforms.Compose([
		        transforms.RandomResizedCrop(224),
		        transforms.RandomHorizontalFlip(),
		        transforms.ToTensor(),
		        normalize,
		    ]))

		train_sampler = None

		train_loader = torch.utils.data.DataLoader(
		    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
		    num_workers=16, prefetch_factor=4, pin_memory=True, persistent_workers=True, sampler=train_sampler)
		test_loader = torch.utils.data.DataLoader(
		    datasets.ImageFolder(args.datapath + "val/", transforms.Compose([
		        transforms.Resize(256),
		        transforms.CenterCrop(224),
		        transforms.ToTensor(),
		        normalize,
		    ])),
		    batch_size=128, shuffle=False,
		    num_workers=4, pin_memory=True, persistent_workers=True)

	elif args.dataset == 'cifar10':
		if args.arch == 'resnet32':
			model = resnet32("A").to(device)
			model.load_state_dict(torch.load("pretrained_models/resnet32_pruned.pt", map_location=device))
			report_name = "CIFAR10_ResNet32"
		else:
			model = MobileNetV2().to(device)
			model.load_state_dict(torch.load("pretrained_models/MobileNet_CIFAR10.pt", map_location=device))
			report_name = "CIFAR10_MobileNetV2"

		kwargs = {'num_workers': 8, 'pin_memory': True}
		transform= transforms.Compose([
						           transforms.RandomCrop(32, padding=4),
						           transforms.RandomHorizontalFlip(),
						           transforms.ToTensor(),
						           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
						       ])
		train_dataset = datasets.CIFAR10(args.datapath, transform=transform, train=True, download=True)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
		test_loader = torch.utils.data.DataLoader(
		     datasets.CIFAR10(args.datapath, train=False, transform=transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		                   ])),
		    batch_size=args.test_batch_size, shuffle=False, **kwargs)
	elif args.dataset == 'cifar100':
		model = AlexNet().to(device)
		model.load_state_dict(torch.load("pretrained_models/AlexNet.pt", map_location=device))
		report_name = "CIFAR100_AlexNet"
		kwargs = {'num_workers': 8, 'pin_memory': True}
		train_loader = torch.utils.data.DataLoader(
		   datasets.CIFAR100(args.datapath, train=True, download=True,
		                   transform= transforms.Compose([
						           transforms.RandomCrop(32, padding=4),
						           transforms.RandomHorizontalFlip(),
						           transforms.ToTensor(),
						           transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
						       ])),
		    batch_size=args.batch_size, shuffle=True, **kwargs)
		test_loader = torch.utils.data.DataLoader(
		     datasets.CIFAR100(args.datapath, train=False, transform=transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Normalize((0.50707516, 0.48654887, 0.44091784), (0.26733429, 0.25643846, 0.27615047)),
		                   ])),
		    batch_size=args.test_batch_size, shuffle=False, **kwargs)

	elif args.dataset == 'mnist':
		model = LeNet5().to(device)
		report_name = "MNIST_Lenet5_"

		kwargs = {'num_workers': 4, 'pin_memory': True, 'persistent_workers':True}
		train_loader = torch.utils.data.DataLoader(
		    datasets.MNIST(args.datapath + '/MNIST/', train=True, download=True,
		                   transform=transforms.Compose([
		                       transforms.ToTensor(),
		                       transforms.Normalize((0.1307,), (0.3081,))
		                   ])),
		    batch_size=args.batch_size, shuffle=True, **kwargs)
		test_loader = torch.utils.data.DataLoader(
		    datasets.MNIST(args.datapath + 'MNIST/', train=False, transform=transforms.Compose([
		                       transforms.ToTensor(),
		                       transforms.Normalize((0.1307,), (0.3081,))
		                   ])),
		    batch_size=args.test_batch_size, shuffle=False, **kwargs)
	else:
		error()#not implemented
	report_name = report_name + str(args.N)+"_"+str(args.entropy_order)+"_"+str(args.mode)+"_"+str(args.lr)+"_"+str(args.momentum)+"_"+str(args.weight_decay)+"_"+str(args.newregu_every)+"_"+str(args.lamb_H)+"_"+str(args.lamb_RMSE)+"_"+str(args.lr_quant_scale)
	if args.lloydmax:
		report_name = report_name + "_L"
	else:
		report_name = report_name + "_U"

	report_name += 'noS_low'

	optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weight_decay, momentum=args.momentum)
	criterion = nn.CrossEntropyLoss().to(device)

	main_dir = "saved_models"
	models_dir = main_dir+"/"+report_name+"_model"
	quantized_dir = main_dir+"/"+report_name+"_quantized"
	lz_compressed_dir = main_dir+"/"+report_name+"_lz_compressed"
	os.makedirs(main_dir, exist_ok=True)
	os.makedirs(models_dir, exist_ok=True)
	os.makedirs(quantized_dir, exist_ok=True)
	os.makedirs(lz_compressed_dir, exist_ok=True)
	if args.histograms:
		os.makedirs("histograms/"+report_name, exist_ok=True)
	if args.store_all:
		os.makedirs(models_dir, exist_ok=True)
		os.makedirs(quantized_dir, exist_ok=True)

	f = open(report_name + ".csv", "w")
	f.close()
	print("All setup correctly")
	voronoi = None
	q_model = deepcopy(model)
	epoch = 0
	print(epoch, "\t Generating quantized model...")
	q_model.load_state_dict(deepcopy(model.state_dict()))
	voronoi = LM_quantize_model(q_model, args.N, device, args.lloydmax, new_voronoi=True, voronoi = voronoi)
	print("generated")
	sequence = generate_sequence_tocompress(q_model, voronoi, mode = args.mode)
	torch.save(model.state_dict(), models_dir+'.pt')
	torch.save(sequence, quantized_dir+'.pt')
	scaler = torch.cuda.amp.GradScaler(enabled = args.amp)

	sensitivity = S(model, device)

	for epoch in range(args.beg_epoch + 1, args.epochs + 1):
		voronoi = train(args, model, device, train_loader, optimizer, epoch, criterion, q_model, voronoi, test_loader, models_dir, quantized_dir, scaler, sensitivity)
		voronoi = get_statistics_csv(args, model, test_loader, device, voronoi, epoch, report_name, q_model)
		sequence = generate_sequence_tocompress(q_model, voronoi, mode = args.mode)
		if args.store_all:
			torch.save(model.state_dict(), models_dir+'/'+str(epoch)+'.pt')
			torch.save(sequence, quantized_dir+'/'+str(epoch)+'.pt')
			size = lzma_compress(lz_compressed_dir+'/'+str(epoch)+'.7z', quantized_dir+'/'+str(epoch)+'.pt')
			f = open(report_name + ".csv", "a")
			f.write("\t{}\n".format(size))
			f.close()
		else:
			torch.save(model.state_dict(), models_dir+'.pt')
			torch.save(sequence, quantized_dir+'.pt')
			print("stored at ", quantized_dir)

if __name__ == '__main__':
    main()
