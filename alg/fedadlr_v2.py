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


class FedADLRV2Server(BaseServer):

    def __init__(self, args):
        super(FedADLRV2Server, self).__init__(args=args)
        optimers, loss_func, global_parameters = self.init_params()
        self.optimers = optimers
        self.loss_func = loss_func
        self.global_parameters = global_parameters
        self.beta = args['beta']
        self.min_lr = args['min_lr'] if "min_lr" in args else 1e-6
        self.max_lr = args["max_lr"] if "max_lr" in args else 0.1
        self.is_weight_decay= args['is_weight_decay'] if 'is_weight_decay' in args else False
        self.weight_decay = args['weight_decay'] if 'weight_decay' in args else 0.99
        self.shuffle = args['shuffle'] if 'shuffle' in args else True
        self.pre_t = args['pre_t']
        self.eps = args['eps']
        self.is_approx = args['is_approx'] if 'is_approx' in args else False
        self.is_local_acc = args['is_local_acc'] if 'is_local_acc' in args else False

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

    def get_conv_lr_list(self):
        model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                    self.args['model_name'], self.args['num_of_clients'])
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        conv_lr_list = []
        for idx in range(self.pre_t):
            lr_list = []
            self.optimers, self.loss_func, self.global_parameters = self.init_params()
            print('Fix-point round {}'.format(idx))
            eps_list = []
            for t in tqdm([i for i in range(self.num_comm)]):
                clients_in_comm, theta_list = self.sample()
                client_lrs = {}
                sum_parameters = None
                if idx == 0:
                    next_params_path = os.path.join(self.args['init_model'], '{}.pth'.format(t + 1))
                else:
                    next_params_path = os.path.join(model_save_dir, '{}.pth'.format(t + 1))
                next_params = torch.load(next_params_path, map_location=lambda storage, loc: storage.cuda())
                next_phi_t_2 = 0.0 
                for var in next_params:
                    next_phi_t_2  += float(next_params[var].reshape(-1).float().dot(next_params[var].reshape(-1).float()))
                for client in clients_in_comm:
                    # print(idx, t, client)
                    local_parameters = self.myClients.clients_set[client].fixPointUpdate(
                                                        localEpoch=self.args['epoch'], localBatchSize=self.args['batchsize'], Net=self.net, 
                                                        lossFun=self.loss_func, opti=self.optimers[client], global_parameters=self.global_parameters, 
                                                        client=client, t=t, theta_i=theta_list[client], next_phi_t_2=next_phi_t_2, shuffle=self.shuffle, 
                                                        beta=self.beta, min_lr=self.min_lr, max_lr=self.max_lr, is_weight_decay=self.is_weight_decay, weight_decay=self.weight_decay)
                    lr = 0.0
                    for param in self.optimers[client].param_groups:
                        lr = param['lr']
                        break
                    client_lrs[client] = lr
                    if sum_parameters is None:
                        sum_parameters = {}
                        for key, var in local_parameters.items():
                            sum_parameters[key] = theta_list[client] * var.clone()
                    else:
                        for var in sum_parameters:
                            sum_parameters[var] = sum_parameters[var] + theta_list[client] * local_parameters[var]
                
                for var in self.global_parameters:
                    self.global_parameters[var] = sum_parameters[var]
                
                eps = 0
                for var in self.global_parameters:
                    eps += torch.sum(torch.abs(self.global_parameters[var].reshape(-1) - next_params[var].reshape(-1)))
                eps_list.append(eps)

                self.net.load_state_dict(self.global_parameters, strict=True)
                torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
                lr_list.append(client_lrs)
                # print("idx = {}, t \ {}".format(idx, t))

            conv_lr_list = lr_list
            # print('---', eps_list)
            flag = True
            for eps in eps_list:
                if eps > self.eps:
                    flag = False
                    break
            if flag:
                break

        return conv_lr_list

    def adjust_learning_rate(self, optimizer, lr):
        '''
        adjust learning rate of optimizer
        '''
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr)

    def run(self):
        if 'parmas_mode' in self.args:
            model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'])
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
        conv_lr_list = self.get_conv_lr_list()
        # print(conv_lr_list)
        self.optimers, self.loss_func, self.global_parameters = self.init_params()
        loss_list = []
        accuracy_list = []
        accuracy, loss = self.eval(t=0)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            # clients_in_comm, theta_list = self.sample()
            clients_in_comm = conv_lr_list[t]
            theta_sum = sum([self.theta_list[client] for client in clients_in_comm])
            theta_list = {}
            for client in clients_in_comm:
                theta_list[client] = self.theta_list[client] / theta_sum
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            for client in tqdm(clients_in_comm):
                self.adjust_learning_rate(optimizer=self.optimers[client], lr=conv_lr_list[t][client])
                local_parameters = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                            self.loss_func, self.optimers[client], self.global_parameters)
                if self.is_local_acc and t == self.num_comm - 1:
                    self.local_eval(t=t, local_parameters=local_parameters, client=client)
                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = theta_list[client] * var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] +  theta_list[client] * local_parameters[var]
            
            for var in self.global_parameters:
                self.global_parameters[var] = sum_parameters[var]
            
            accuracy, loss = self.eval(t=t)
            accuracy_list.append(accuracy)
            loss_list.append(loss)
            print('accuracy: ' + str(accuracy) + "\n")
            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n") 

        s = 'dataset_{}_IID_{}_{}_{}_cli_{}_frac_{}_local_e_{}'.format(self.args['dataset'], self.args['IID'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args["cfraction"], self.args['epoch'])
        self.test_txt.write(s + " start\n")
        self.test_txt.write("\n")
        self.test_txt.write('accuracy_list={}'.format(accuracy_list))
        self.test_txt.write("\n")
        self.test_txt.write('loss_list={}'.format(loss_list))
        self.test_txt.write("end\n")
        self.test_txt.close()
        print('accuracy_list={}'.format(accuracy_list))
        print('loss_list={}'.format(loss_list))
            