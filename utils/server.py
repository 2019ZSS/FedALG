import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from tqdm import tqdm
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
)
from utils.clients import (
    ClientsGroup, 
)


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def adjust_learning_rate(optimizer, lr):
    '''
    adjust learning rate of optimizer
    '''
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


class BaseServer(object):

    def __init__(self, args):
        test_mkdir('./logs')
        test_mkdir(args['save_path'])
        self.test_txt = open("./logs/test_accuracy.txt", mode="a")
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
        elif args['dataset'] == 'cifa10':
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
            self.net = WideResNet(depth=28, num_classes=10).to(self.dev)
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

        self.myClients = ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], self.dev)
        self.testDataLoader = self.myClients.test_data_loader

        self.global_parameters = {}

        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for key, var in self.net.state_dict().items():
            self.global_parameters[key] = var.clone()

        self.theta_list = {}
        for i in range(args['num_of_clients']):
            client = 'client{}'.format(i)
            self.theta_list[client] = self.myClients.clients_set[client].theta 

        self.num_comm = args['num_comm']
        self.num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    def sample(self):
        order = np.random.permutation(self.args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:self.num_in_comm]]
        theta_sum = sum([self.theta_list[client] for client in clients_in_comm])
        theta_list = {}
        for client in clients_in_comm:
            theta_list[client] = self.theta_list[client] / theta_sum
        return clients_in_comm, theta_list

    def run(self):
        pass

    def eval(self, t):
        sum_accu, num = 0, 0
        self.net.load_state_dict(self.global_parameters, strict=True)
        with torch.no_grad():
            for data, label in self.testDataLoader:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = self.net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
            accuracy = float(sum_accu / num)
        if (t + 1) % self.args['save_freq'] == 0:
            torch.save(self.net, os.path.join(self.args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(
                                                            self.args['alg'], self.args['model_name'],
                                                            t, self.args['epoch'], self.args['batchsize'], self.args['learning_rate'], 
                                                            self.args['num_of_clients'],self.args['cfraction'])))
        return accuracy
    
    