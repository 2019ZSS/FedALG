import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils import BaseServer
import matplotlib.pyplot as plt


class FedADLRServer(BaseServer):

    def __init__(self, args):
        super(FedADLRServer, self).__init__(args=args)
        optimers, loss_func, global_parameters = self.init_params()
        self.optimers = optimers
        self.loss_func = loss_func
        self.global_parameters = global_parameters
        self.beta = args['beta']
        self.min_lr = args['min_lr']
        self.weight_decay = args['weight_decay'] if 'weight_decay' in args else 0.99
        self.shuffle = args['shuffle'] if 'shuffle' in args else True
        self.is_approx = args['is_approx'] if 'is_approx' in args else False

    def init_params(self):
        global_parameters = {}
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
            global_parameters[key] = var.clone()
            
        loss_func = F.cross_entropy

        optimers = {}
        for i in range(self.args['num_of_clients']):
            opt_list = []
            for name, params in self.net.named_parameters():
                opt_d = {
                    'params': params,
                    'params_name': name,
                }
                opt_list.append(opt_d)
            optimers['client{}'.format(i)] = optim.SGD(opt_list, lr=self.args['learning_rate'])
        return optimers, loss_func, global_parameters 

    def run(self):
        accuracy_list = []
        phi_list = []
        lr_list = []
        if 'parmas_mode' in self.args:
            model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'])
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)
        accuracy_list.append(self.eval(t=0))
        print(accuracy_list[-1])
        phi_t = 0.0
        for var in self.global_parameters:
            phi_t += float(self.global_parameters[var].reshape(-1).float().dot(self.global_parameters[var].reshape(-1).float()))

        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            clients_lr = {}
            next_phi_t = 0 
            for client in tqdm(clients_in_comm):
                local_parameters = self.myClients.clients_set[client].adplrUpdate(
                                                localEpoch=self.args['epoch'], localBatchSize=self.args['batchsize'], Net=self.net, 
                                                lossFun=self.loss_func, opti=self.optimers[client], global_parameters=self.global_parameters, 
                                                client=client, t=t, phi_t=phi_t, theta_i=theta_list[client],  shuffle=self.shuffle, 
                                                beta=self.beta, min_lr=self.min_lr, weight_decay=self.weight_decay, is_approx=self.is_approx)
                lr = 0.0
                for param in self.optimers[client].param_groups:
                    lr = param['lr']
                    break
                clients_lr[client] = lr
                for var in local_parameters:
                    next_phi_t += theta_list[client] * float(local_parameters[var].reshape(-1).float().dot(local_parameters[var].float().reshape(-1)))

                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = theta_list[client] * var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] +  theta_list[client] * local_parameters[var]

            phi_list.append(next_phi_t)
            lr_list.append(clients_lr)
            for var in self.global_parameters:
                self.global_parameters[var] = sum_parameters[var]
            
            if 'parmas_mode' in self.args:
                torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            
            accuracy = self.eval(t=t)
            accuracy_list.append(accuracy)
            print('accuracy: ' + str(accuracy) + "\n")
            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n") 

        clients_lr = {}
        for dic in lr_list:
            for key, val in dic.items():
                if key not in clients_lr:
                    clients_lr[key] = []
                clients_lr[key].append(val)
        
        for key, val in clients_lr.items():
            t_list = [i for i in range(len(val))]
            plt.cla()
            plt.plot(t_list, val)
            plt.xlabel('local iters')
            plt.ylabel('learning rate')
            plt.savefig('./images/lr/{}_{}_lr.png'.format(key, self.args['IID']), bbox_inches='tight')

        self.test_txt.write('phi_list={}'.format(phi_list))
        self.test_txt.write('accuracy_list={}'.format(accuracy_list))
        self.test_txt.close()
        print('phi_list={}'.format(phi_list))
        print('accuracy_list={}'.format(accuracy_list))
            