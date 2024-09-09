from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
from pathlib import Path
from os.path import join
from torch.autograd import Variable
from cka_utils import MinibatchCKA
import tqdm
import skimage
import pandas as pd
import json
import numpy as np

parser = argparse.ArgumentParser(description='TrojAI Object Detection Jan 2023 CKA')
parser.add_argument('--model_num', default=0, type=int, help='model id number')
parser.add_argument('--clean_poisoned_input', default='clean', type=str, help='options are {clean, local_poisoned, global_poisoned}, clean model has only clean sample input')
parser.add_argument('--layer_types', default='convlinearlayer', type=str, help='options are {convlinearlayer, actlayer, bnlayer, fulllayer}')

args = parser.parse_args()

MODEL_NUM = 117
MODEL_FILEDIR = '/scratch/data/TrojAI/object-detection-feb2023-train/models/'
MODEL_SUMMARY_FILEPATH = '/scratch/data/TrojAI/object-detection-feb2023-train/METADATA.csv'
OUTPUT_FILEDIR = '/scratch/jialin/object-detection-feb2023/weight_analysis/extracted_source/'
METADATA = pd.read_csv(MODEL_SUMMARY_FILEPATH)

def num_to_model_id(num):
    return 'id-' + str(100000000+num)[1:]

def load_model(model_num):
    model_id = num_to_model_id(model_num)
    model_filepath = os.path.join(MODEL_FILEDIR, model_id, 'model.pt')
    # model_info_fp = model_filepath + '.stats.json'
    model = torch.load(model_filepath)
    # with open(model_info_fp, 'r') as f:
    #     model_info = json.load(f)
    return model

if args.model_num not in range(MODEL_NUM):
    print('\n Model Num Provided Invalid \n End the Process')
    exit

model_id = num_to_model_id(args.model_num)
model_dir = os.path.join(MODEL_FILEDIR, model_id)

augmentation_transforms = transforms.Compose([torchvision.transforms.ConvertImageDtype(torch.float)])

IMGS, IMGS_INFO = [], []

if args.clean_poisoned_input.startswith('global'):
    for model_num in range(MODEL_NUM):
            sample_data_dir = os.path.join(MODEL_FILEDIR, num_to_model_id(model_num), 'poisoned-example-data')
            if os.path.exists(sample_data_dir):
                imgs, imgs_info = [], []
                for img_fp in os.listdir(sample_data_dir):
                    if img_fp.endswith('.png'):
                        img = skimage.io.imread(os.path.join(sample_data_dir, img_fp))
                        img = augmentation_transforms(torch.as_tensor(img).permute(2, 0, 1))
                        imgs.append(img)
                        with open(os.path.join(sample_data_dir, f"{img_fp.split('.')[0]}.json")) as jsonfile:
                            imgs_info.append(json.load(jsonfile))
                IMGS += imgs
                IMGS_INFO += imgs_info
elif args.clean_poisoned_input == 'clean':
    sample_data_dir = os.path.join(model_dir, f'{args.clean_poisoned_input}-example-data')
elif args.clean_poisoned_input.startswith('local'):
    sample_data_dir = os.path.join(model_dir, 'poisoned-example-data')
    if not os.path.exists(sample_data_dir):
        print('\n Clean model has no local poisoned input data, using clean data instead')
        sample_data_dir = os.path.join(model_dir, 'clean-example-data')
if len(IMGS) == 0:
    for img_fp in os.listdir(sample_data_dir):
        if img_fp.endswith('.png'):
            img = skimage.io.imread(os.path.join(sample_data_dir, img_fp))
            img = augmentation_transforms(torch.as_tensor(img).permute(2, 0, 1))
            IMGS.append(img)
            with open(os.path.join(sample_data_dir, f"{img_fp.split('.')[0]}.json")) as jsonfile:
                IMGS_INFO.append(json.load(jsonfile))    

model_fp = os.path.join(model_dir, 'model.pt')

# Hyper Parameter settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# best_acc = 0
# start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, args.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
# transform_train = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
# ]) # meanstd transformation

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),

# imgs = torch.stack(imgs, dim=0)
# labels = torch.as_tensor([img_info['label'] for img_info in imgs_info])

# if(args.dataset == 'cifar10'):
#     print("| Preparing CIFAR-10 dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.CIFAR10(root='/data/yefan0726/data/cv/cifar10', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR10(root='/data/yefan0726/data/cv/cifar10', train=False, download=False, transform=transform_test)
#     num_classes = 10
# elif(args.dataset == 'cifar100'):
#     print("| Preparing CIFAR-100 dataset...")
#     sys.stdout.write("| ")
#     trainset = torchvision.datasets.CIFAR100(root='/data/yefan0726/data/cv/cifar100', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR100(root='/data/yefan0726/data/cv/cifar100', train=False, download=False, transform=transform_test)
#     num_classes = 100


# if args.data_samples < 10000:
#     print(f"Use only {args.data_samples} data samples")
#     subset_list = list(range(0, args.data_samples))
#     testset_subset = torch.utils.data.Subset(testset, subset_list)
#     testloader = torch.utils.data.DataLoader(testset_subset, batch_size=args.num_batch, 
#                                                     shuffle=False, num_workers=6)

#     trainset_subset = torch.utils.data.Subset(trainset, subset_list)
#     trainloader = torch.utils.data.DataLoader(trainset_subset, batch_size=args.num_batch, 
#                                                     shuffle=False, num_workers=6)
# else:
#     print("Use full test dataset")
#     testloader = torch.utils.data.DataLoader(testset, batch_size=args.num_batch, 
#                                                         shuffle=False, num_workers=6)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.num_batch, 
#                                                         shuffle=False, num_workers=6)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes, args.widen_factor)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'resnet-init'):
        net = ResNet_init(args.depth, num_classes, args.widen_factor)
        file_name = 'resnet-init'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'wide-resnet-dropout'):
        net = WideResNet(depth=args.depth, 
                        num_classes=num_classes,
                        widen_factor=args.widen_factor,
                        dropRate=args.dropout)
        file_name = 'wide-resnet-dropout'+str(args.depth)+'x'+str(args.widen_factor)
    elif (args.net_type == 'deep-resnet'):
        net = DeepResNet(depth=args.depth, 
                        output_classes=num_classes)
        file_name = 'deep-resnet-'+str(args.depth)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    #print(net)
    return net, file_name

# Model
print('\n[Phase 2] : Model setup')
# if args.resume:
#     # Load checkpoint
#     print(f'| Resuming from checkpoint {args.resume}...')
#     net, file_name = getNetwork(args)
#     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
#     checkpoint = torch.load(args.resume, map_location='cpu')
#     net.load_state_dict(checkpoint['net'])
    
# else:
#     print('| Building net type [' + args.net_type + ']...')
#     net, file_name = getNetwork(args)
#     net.apply(conv_init)
net = load_model(args.model_num).to(device)
net.eval()

# if use_cuda:
#     net.cuda()
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

def test(epoch, net):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * targets.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100.*correct/total
        acc = acc.item()
        test_loss = test_loss/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, acc))
        
        return acc, test_loss
        

print('\n[Phase 3] : Training model')
# print('| Training Epochs = ' + str(num_epochs))
# print('| Initial Learning Rate = ' + str(args.lr))
# print('| Optimizer = ' + str(optim_type))

# use_cuda=True
# device = torch.device('cuda')

net.eval()
# test_acc, test_loss = test(epoch=0, net=net)
layer_name_lst = []
    
def get_activation(name):
    def hook(model, input, output):
        activations.append(output) #.detach()
    return hook

hooks = {} #isinstance(module, nn.ReLU)  or  or isinstance(module, nn.BatchNorm2d)

# print(net)
for name, module in net.backbone.named_modules():
    if len(name) == 0:
        continue
    if args.layer_types == 'convlinearlayer':
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            layer_name_lst.append(name)
            hooks[name] = module.register_forward_hook(get_activation(name))
    elif args.layer_types == 'actlayer':
        if isinstance(module, nn.ReLU) or isinstance(module, nn.Linear):
            layer_name_lst.append(name)
            hooks[name] = module.register_forward_hook(get_activation(name))
    elif args.layer_types == 'bnlayer':
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
            layer_name_lst.append(name)
            hooks[name] = module.register_forward_hook(get_activation(name))
    elif args.layer_types == 'fulllayer':
        # print(f"{args.layer_types}")
        if isinstance(module, nn.ReLU) or isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.Linear):
            hooks[name] = module.register_forward_hook(get_activation(name))
            layer_name_lst.append(name)

print(layer_name_lst)
print(len(layer_name_lst))
print(len(hooks))
# np.save(os.path.join(OUTPUT_FILEDIR, f'layer_name_lst.npy'), layer_name_lst)

cka = MinibatchCKA(len(layer_name_lst), device)
batch_size = 50
# # pbar = tqdm.tqdm(total = len(imgs))

if len(IMGS) == 1120:
    imgs_inds = np.random.permutation(1120)[:]
    # imgs = [IMGS[ind].to(device) for ind in imgs_inds]

with torch.no_grad():
    # activations = []
    # net(imgs)
    for ind in range(int(len(IMGS)//batch_size+1)):
        activations = []
        max_len = (ind+1)*batch_size if (ind+1)*batch_size < len(IMGS) else len(IMGS)
        input_data = [d.to(device) for d in IMGS[ind*batch_size:max_len]]
        net(input_data)
    # # for act in activations:
    # #     print(act.shape, act.sum(), torch.nonzero(act).shape[0] / act.numel())

    #     print(f'activation length on main_cka_debug {len(activations)}')
        cka.update_state(activations)


heatmap = cka.result().detach().cpu().numpy()

print(f"Saving the heatmap name as minicka_model_{model_id}_{args.clean_poisoned_input}_input_{args.layer_types}")
np.save(os.path.join(OUTPUT_FILEDIR, f'minicka_model_{model_id}_{args.clean_poisoned_input}_input_{args.layer_types}.npy'), heatmap)


#MinibatchCKA()
#print(len(layer_name_lst))
# cka = CKA(net, net,
#         model1_name=f"model1", model2_name=f"model1_1",
#          model1_layers=layer_lst, # List of layers to extract features from
#          model2_layers=layer_lst, 
#         device=device)



