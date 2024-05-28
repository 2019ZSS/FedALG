import os
import sys
from tkinter import NO

from model.ResNet import resnet18
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from tqdm import tqdm
import random 
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from model import (
    WideResNet,
    Mnist_2NN,
    Mnist_CNN,
    LinearNet,
    LeNet,
    LeNet5,
    LeNet_EM,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    Cifar_CNN,
    CIFAR10Model,
    VGG11,
)
from utils.clients import (
    ClientsGroup, 
)


def test_mkdir(path):
    if not os.path.isdir(path):
        # os.mkdir(path)
        os.makedirs(path)


def adjust_learning_rate(optimizer, lr):
    '''
    adjust learning rate of optimizer
    '''
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


class BaseServer(object):

    def __init__(self, args):
        # print(args)
        test_mkdir('./logs')
        test_mkdir(args['save_path'])
        os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
        self.args = args
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = None

        self.num_outputs = 10
        if args['dataset'] == 'mnist':
            self.num_outputs = 10
        elif args['dataset'] == 'mnist_v2':
            self.num_outputs = 10
        elif args['dataset'] == 'emnist':
            self.num_outputs = 62
        elif args['dataset'] == 'cifar10':
            self.num_outputs = 10
        elif args['dataset'] == 'cifar100':
            self.num_outputs = 100
        elif args['dataset'] == 'svhn':
            self.num_outputs = 10
        else:
            raise NotImplementedError('{}'.format(args['dataset']))
        
        if args['model_name'] == 'mnist_2nn':
            self.net = Mnist_2NN(num_outputs=self.num_outputs)
        elif args['model_name'] == 'mnist_cnn':
            self.net = Mnist_CNN(num_outputs=self.num_outputs)
        elif args['model_name'] == 'mnist_linear':
            self.net = LinearNet(num_outputs=self.num_outputs)
        elif args['model_name'] == 'wideResNet':
            self.net = WideResNet(depth=28, num_classes=self.num_outputs)
        elif args['model_name'] == 'lenet':
            self.net = LeNet(num_classes=self.num_outputs)
        elif args['model_name'] == 'lenet5':
            self.net = LeNet5(num_classes=self.num_outputs)
        elif args['model_name'] == 'lenet_em':
            self.net = LeNet_EM(num_classes=self.num_outputs)
        elif args['model_name'] == 'resnet18':
            self.net = resnet18(num_classes=self.num_outputs)
        elif args['model_name'] == 'resnet34':
            self.net = resnet34(num_classes=self.num_outputs)
        elif args['model_name'] == 'resnet50':
            self.net = resnet50(num_classes=self.num_outputs)
        elif args['model_name'] == 'resnet101':
            self.net = resnet101(num_classes=self.num_outputs)
        elif args['model_name'] == 'cifar10_cnn':
            self.net = Cifar_CNN(num_classes=self.num_outputs)
        elif args['model_name'] == 'cifar10_model':
            self.net = CIFAR10Model(num_classes=self.num_outputs)
        elif args['model_name'] == 'vgg11':
            self.net = VGG11(num_classes=self.num_outputs)
        else:
            raise NotImplementedError('{}'.format(args['model_name']))

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = torch.nn.DataParallel(self.net)
        
        self.net = self.net.to(self.dev)
        self.loss_func = F.cross_entropy
        self.optimers = {}
        for i in range(args['num_of_clients']):
            self.optimers['client{}'.format(i)] = optim.SGD(self.net.parameters(), lr=args['learning_rate'])
        alpha = args['alpha'] if 'alpha' in args else 1.0
        resize = args['resize'] if 'resize' in args else 224
        split = args["split"] if "split" in args else "letters"
        rearrange = args["rearrange"] if "rearrange" in args else 0
        self.myClients = ClientsGroup(dataSetName=args['dataset'], isIID=args['IID'], numOfClients=args['num_of_clients'], 
                                dev=self.dev, alpha=alpha, resize=resize, split=split, rearrange=rearrange)
        self.testDataLoader = self.myClients.test_data_loader

        self.global_parameters = {}

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()

        self.theta_list = {}
        for i in range(args['num_of_clients']):
            client = 'client{}'.format(i)
            self.theta_list[client] = self.myClients.clients_set[client].theta 

        self.num_comm = args['num_comm']
        self.client_set = [i for i in range(0, args['num_of_clients'])]
        self.num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

        self.log_save_dir = './logs/{}/IID_{}/{}/{}/cli_{}_frac_{}_local_e_{}_alpha_{}_resize_{}'.format(self.args['dataset'], self.args['IID'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args["cfraction"], self.args['epoch'], alpha, resize)
        test_mkdir(self.log_save_dir)
        log_file = os.path.join(self.log_save_dir, "test_accuracy.txt")
        self.test_txt = open(log_file, mode="a")
        self.test_txt.write('{}_{}'.format(args['model_name'], args['IID']))

    def sample(self):
        # order = np.random.permutation(self.args['num_of_clients'])
        # clients_in_comm = ['client{}'.format(i) for i in order[0:self.num_in_comm]]
        clients_sample = random.sample(self.client_set, self.num_in_comm)
        clients_in_comm = ['client{}'.format(i) for i in clients_sample]
        theta_sum = sum([self.theta_list[client] for client in clients_in_comm])
        theta_list = {}
        for client in clients_in_comm:
            theta_list[client] = self.theta_list[client] / theta_sum
        return clients_in_comm, theta_list

    def run(self):
        pass

    def eval(self, t, is_loss=False):
        sum_accu, num = 0, 0
        self.net.load_state_dict(self.global_parameters, strict=True)
        with torch.no_grad():
            loss_list = []
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                # print(data.shape, label.shape)
                preds = self.net(data)
                # print(preds.shape)
                # print(label.shape)
                loss = self.loss_func(preds, label)
                loss_list.append(loss.mean())
                preds = torch.argmax(preds, dim=1)
                # print(preds.shape, label.shape)
                # exit(0)
                sum_accu += (preds == label).float().mean()
                num += 1
            avg_loss = float(sum(loss_list) / num)
            accuracy = float(sum_accu / num)
        if (t + 1) % self.args['save_freq'] == 0:
            torch.save(self.net, os.path.join(self.args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(
                                                            self.args['alg'], self.args['model_name'],
                                                            t, self.args['epoch'], self.args['batchsize'], self.args['learning_rate'], 
                                                            self.args['num_of_clients'],self.args['cfraction'])))
        return accuracy, avg_loss
    
    def local_eval(self, t, local_parameters, client):
        sum_accu, num = 0, 0
        self.net.load_state_dict(local_parameters, strict=True)
        with torch.no_grad():
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            accuracy = float(sum_accu / num)
        # acc_avg_dir = './checkpoints/acc/IID_{}/{}/{}/{}/{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
        #                 self.args['model_name'], self.args['num_of_clients'])
        # os.makedirs(acc_avg_dir, exist_ok=True)
        acc_avg_file = os.path.join(self.log_save_dir, 'loc_acc.txt')
        # os.makedirs(acc_avg_file, exist_ok=True)
        with open(acc_avg_file, 'a') as f:
            f.write('t = {}, client = {}, acc = {}\n'.format(t, client, accuracy))
        
    