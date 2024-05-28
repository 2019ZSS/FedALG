from cgi import print_arguments
import os
import sys
import copy
import random
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import math
from tokenize import Number
from traceback import print_tb
from turtle import left
from matplotlib.pyplot import flag
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from utils.getData import GetDataSet, MyDataset, PartialDataset
import sympy 
from sympy import Matrix, Symbol, false, log, posify
from collections import Counter
import json

def get_label_distribution(labels):
    label_counts = Counter(labels.cpu().numpy())
    return label_counts

class client(object):
    
    def __init__(self, trainDataSet, dev, theta):
        self.train_ds = trainDataSet
        self.dev = dev
        self.theta = theta
        self.train_dl = None
        self.local_parameters = None
    
    def localPartialUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, max_bound=None, indices=None):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        Net.load_state_dict(global_parameters, strict=True)

        partial_dataset = PartialDataset(self.train_ds, indices)

        self.train_dl = DataLoader(partial_dataset, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            epoch_label = []
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
            # 验证标签分布
            #     epoch_label.extend(label.cpu().numpy())
            # label_distribution = get_label_distribution(torch.tensor(epoch_label))
            # print(label_distribution)      
        return Net.state_dict()

    def localPartialUpdate_prox(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, prox_mu, indices=None):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数group['params'][0]
        '''
        Net.load_state_dict(global_parameters, strict=True)
        
        partial_dataset = PartialDataset(self.train_ds, indices)
        self.train_dl = DataLoader(partial_dataset, batch_size=localBatchSize, shuffle=True)
        prox_mu = 0.001
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                opti.zero_grad()
                proximal_term = 0.0
                # for w, w_t in zip(Net.parameters(), global_parameters.values()):
                #     proximal_term += (w - w_t).norm(2)
                # local_parameters = Net.state_dict()
                # for key, var in local_parameters.items():
                #     proximal_term += (global_parameters[key] - var).norm(2)
                flitered_global_parameters = {}
                for name, param in global_parameters.items():
                    if "weight" in name or "bias" in name:
                        flitered_global_parameters[name] = param
                for w, w_t in zip(Net.parameters(), flitered_global_parameters.values()):
                    proximal_term += (w - w_t).norm(2)
                loss = lossFun(preds, label) + (prox_mu / 2) * proximal_term
                loss.backward()
                opti.step()
        return Net.state_dict()

    def localPartialUpdate_babu(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, max_bound=None, indices=None):
        Net.load_state_dict(global_parameters, strict=True)
        partial_dataset = PartialDataset(self.train_ds, indices)
        for param in Net.fc2.parameters():
            param.requires_grad = False  # 确保仅更新body部分
        self.train_dl = DataLoader(partial_dataset, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        for param in Net.fc2.parameters():
            param.requires_grad = True  # 恢复fc2的梯度计算
        return Net.state_dict()

    def localPartialUpdate_gh(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, max_bound=None, indices=None):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # Load the global parameters (GGH)
        Net.load_state_dict(global_parameters, strict=True)
        partial_dataset = PartialDataset(self.train_ds, indices)
        self.train_dl = DataLoader(partial_dataset, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)  # Adjusted to handle the output format of FedGH
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        return Net.state_dict()

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, max_bound=None, indices=None):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        max_local_grad_bound=0
        max_L_bound=0
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        last_param_w, last_grad = {}, {}
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                # if max_bound is not None:
                #     with torch.no_grad():
                #         local_parameters = Net.state_dict()
                #         for group in opti.param_groups:
                #             var = group['params_name']
                #             grad = group['params'][0].grad.reshape(-1)
                #             max_local_grad_bound = max(max_local_grad_bound, torch.linalg.norm(grad))
                #             if var not in last_param_w:
                #                 last_param_w[var] = copy.deepcopy(local_parameters[var].reshape(-1))
                #                 last_grad[var] = copy.deepcopy(grad)
                #             else:
                #                 param_w = local_parameters[var].reshape(-1)
                #                 delta_grad = torch.linalg.norm(grad - last_grad[var])
                #                 delta_w = torch.linalg.norm(param_w - last_param_w[var])
                #                 real_L_bound =  delta_grad / delta_w
                #                 # if real_L_bound > 1e8:
                #                 #     print(delta_grad, delta_w, real_L_bound)
                #                 max_L_bound = max(max_L_bound, real_L_bound)
                #                 last_param_w[var] = copy.deepcopy(param_w)
                #                 last_grad[var] = copy.deepcopy(grad)
                opti.step()
        # if max_bound is not None:
        #     # print('max_local_grad_bound={}, max_L_bound={}'.format(max_local_grad_bound, max_L_bound))
        #     max_bound['max_local_grad_bound'] = max(max_local_grad_bound, max_bound['max_local_grad_bound'])
        #     max_bound['max_L_bound'] = max(max_L_bound, max_bound['max_L_bound'])
        return Net.state_dict()

    def localUpdate_prox(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, prox_mu):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数group['params'][0]
        '''
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                opti.zero_grad()
                proximal_term = 0.0
                # for w, w_t in zip(Net.parameters(), global_parameters.values()):
                #     proximal_term += (w - w_t).norm(2)
                # local_parameters = Net.state_dict()
                # for key, var in local_parameters.items():
                #     proximal_term += (global_parameters[key] - var).norm(2)
                flitered_global_parameters = {}
                for name, param in global_parameters.items():
                    if "weight" in name or "bias" in name:
                        flitered_global_parameters[name] = param
                for w, w_t in zip(Net.parameters(), flitered_global_parameters.values()):
                    proximal_term += (w - w_t).norm(2)
                loss = lossFun(preds, label) + (prox_mu / 2) * proximal_term
                loss.backward()
                opti.step()
        return Net.state_dict()
    
    def localUpdate_babu(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, loss_list):
        Net.load_state_dict(global_parameters, strict=True)
        for param in Net.fc2.parameters():
            param.requires_grad = False  # 确保仅更新body部分
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        loss_list.append(loss.item())
        for param in Net.fc2.parameters():
            param.requires_grad = True  # 恢复fc2的梯度计算
        return Net.state_dict(), loss_list

    def fine_tune(self, Net, fineTuningEpochs, localBatchSize, lossFun, opti):
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        Net.train()
        for param in Net.fc2.parameters():
            param.requires_grad = True  # 解冻fc2的梯度计算
        for epoch in range(fineTuningEpochs):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        return Net.state_dict()

    def localUpdate_gh(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, loss_list):
        '''
            param: localEpoch 当前Client的迭代次数
            param: localBatchSize 当前Client的batchsize大小
            param: Net Server共享的模型
            param: LossFun 损失函数
            param: opti 优化函数
            param: global_parmeters 当前通讯中最全局参数
            return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # Load the global parameters (GGH)
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)  # Adjusted to handle the output format of FedGH
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        loss_list.append(loss.item())
        return Net.state_dict(), loss_list

    def localUpdate_dyn(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, dyn_alpha):
        def model_parameter_vector(model):
            param = [p.view(-1) for p in model.parameters()]
            return torch.cat(param, dim=0)

        self.global_model_vector = None
        old_grad = copy.deepcopy(Net)
        old_grad = model_parameter_vector(old_grad)
        self.old_grad = torch.zeros_like(old_grad)
        self.global_model_vector = model_parameter_vector(Net).detach().clone()

        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)

                if self.global_model_vector != None:
                    v1 = model_parameter_vector(Net)
                    loss += dyn_alpha / 2 * torch.norm(v1 - self.global_model_vector, 2)
                    loss -= torch.dot(v1, self.old_grad)

                opti.zero_grad()
                loss.backward()
                opti.step()

        if self.global_model_vector != None:
            v1 = model_parameter_vector(Net).detach()
            self.old_grad = self.old_grad - 0.01 * (v1 - self.global_model_vector)
            
        return Net.state_dict()

    def preUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, next_phi, shuffle=False, beta=0.9, min_lr=0.9):
        opti.zero_grad()
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=shuffle)
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            loss.backward()

        x = Symbol('x')
        left_1, left_2 = 0, 0
        left_1_t, left_2_t = 0, 0
        with torch.no_grad():
            for group in opti.param_groups:
                var = group['params_name']
                group['params'][0].grad = group['params'][0].grad / len(self.train_dl)
                grad = group['params'][0].grad.reshape(1, -1)
                wt = global_parameters[var].reshape(1, -1)
                A = wt.cpu().numpy()
                B = grad.cpu().numpy()
                # (ai - x * bi) * bi = ai * bi - bi * bi * x
                # (ai - x * bi) * (ai - x * bi) = (ai * ai - 2 * ai * bi * x + bi * bi * x * x)
                item1 = float(np.sum(np.dot(A, A.T).reshape(1, -1)))
                item2 = float(np.sum(np.dot(A, B.T).reshape(1, -1)))
                item3 = float(np.sum(np.dot(B, B.T).reshape(1, -1)))
                left_1 += item2 - item3 * x
                left_2 += item1 - 2 * item2 * x + item3 * x * x
                left_1_t += item2
                left_2_t += item1

        ans = sympy.solve((beta / (1 - beta)) * (left_1 / next_phi) * log((left_2 / next_phi) + 1), x)
        deleta = -2 * beta * (left_1_t / next_phi) * math.log((left_2_t / next_phi) + 1)
        real_lr = []
        complex_lr = []
        for val in ans:
            if not isinstance(val, sympy.core.add.Add):
                real_lr.append(val)
            else:
                complex_lr.append(val)

        lr = 0.0
        if len(complex_lr) > 0:
            if deleta > 0:
                lr = min(real_lr)
            else:
                lr = max(real_lr)
        else:
            lr = min(real_lr)

        if lr <= 0:
            lr = min_lr

        for param_group in opti.param_groups:
            param_group['lr'] = float(lr)

        opti.step()
        return Net.state_dict()

    def adplrUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, client, t, phi_t, theta_i, 
                    shuffle=True, beta=0.9, min_lr=1e-6, weight_decay=0.99, is_approx=False):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=shuffle)
        local_parameters = {}

        last_lr = 0.0
        for param_group in opti.param_groups:
            last_lr = param_group['lr']
            break
        # lr_list = []
        # cnt1, cnt2 = 0, 0
        lr = 0.0 # min_lr
        flag = True
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                # if lr == min_lr:
                # if lr != 1e6:
                if flag:
                    x = Symbol('x')
                    left_1, left_2 = 0, 0
                    items_sum = [0 for _ in range(3)]
                    with torch.no_grad():
                        # for key, var in Net.state_dict().items():
                        #     local_parameters[key] = var

                        for group in opti.param_groups:
                            var = group['params_name']
                            grad = group['params'][0].grad.reshape(-1)
                            # wt = local_parameters[var].reshape(-1)
                            wt = global_parameters[var].reshape(-1)

                            item1 = float(wt.dot(wt))
                            item2 = float(wt.dot(grad))
                            item3 = float(grad.dot(grad))
                            items_sum[0] += item1
                            items_sum[1] += item2
                            items_sum[2] += item3
                            left_1 += item2 - item3 * x
                            left_2 += item1 - 2 * item2 * x + item3 * x * x

                    # left_1 = (items_sum[1] - items_sum[2] * x)
                    # left_2 = (items_sum[0] - 2 * items_sum[1] * x + items_sum[2] * x * x)
                    # brefoe 1
                    # f = x - ((beta * theta_i * theta_i) / ((1 - beta) * phi_t)) * left_1 * left_2
                    # update phi_t
                    if is_approx:
                        f = ((beta * theta_i * theta_i) / ((1 - beta) * phi_t * phi_t)) * left_1 * left_2 - x
                    else:
                        f = (beta * theta_i * theta_i / (1 - beta)) * (left_1 / phi_t) * log((left_2 / phi_t) + 1) - x
                    # print(f, end=',')
                    # way 1
                    # ans = sympy.solve(f)
                    # way 2
                    ans = sympy.nsolve(f, 1e-3)
                    ans = [ans]
                    # print(ans)
                    # ans = sympy.solve((beta / (1 - beta)) * (left_1 / phi_t) * log((left_2 / phi_t) + 1), x)
                    # ans = sympy.solve(((beta * theta_i * theta_i) / ((1 - beta) * phi_t)) * left_1 * left_2,  x)
                    # if len(ans) == 0:
                    #     print(delta)
                    #     print(left_1, left_2)
                    #     print(beta, phi_t)
                    #     print(theta_i)
                    #     exit(0)
                    real_lr = []
                    for val in ans:
                        if not isinstance(val, sympy.core.add.Add):
                            real_lr.append(val)

                    # print("ans=", ans)
                    # print('real_lr={}, eq={}'.format(real_lr, f))
                    if len(real_lr) > 0:
                        pos_lr, neg_lr = [], []
                        for val in real_lr:
                            if val > 0:
                                pos_lr.append(val)
                            else:
                                neg_lr.append(val)
                        # print('real_lr={}, pos_lr={}'.format(real_lr, pos_lr))
                        cnt_pos, cnt_neg = len(pos_lr), len(neg_lr)
                        if cnt_neg == 1 and cnt_pos == 2:
                            lr = max(pos_lr)
                        elif cnt_pos == 3:
                            lr = min(pos_lr)
                        elif cnt_pos == 1:
                            lr = min(pos_lr)
                        else:
                            lr = min_lr 
                            # lr = max(neg_lr)
                            # cnt2 += 1
                    else:
                        lr = min_lr

                    # cnt1 += 1
                    # if t == 0:
                    #     if lr != min_lr:
                    #         last_lr = 0.0
                    # if t == 0 and lr != min_lr:
                    #     last_lr = lr
                    # print(last_lr, lr)
                    lr = weight_decay * last_lr + (1.0 - weight_decay) * lr
                    # print(ans, lr)
                    flag = False
                    for param_group in opti.param_groups:
                        param_group['lr'] = float(lr)

                opti.step()
        
        # use_lr = 0.0
        # for param_group in opti.param_groups:
        #     use_lr = param_group['lr']
        #     break

        # print('total={}, min_lr={}, min_lr_rate = {}, lr={}, use_lr={}'.format(cnt1, cnt2, cnt2 / cnt1, lr, use_lr))
        # print('last_lr={}, lr={}'.format(last_lr, lr))
        
        # import matplotlib.pyplot as plt
        # t_list = [i for i in range(len(lr_list))]
        # # minn_lr = min(lr_list)
        # # maxx_lr = max(lr_list)
        # # avg_lr = sum(lr_list) / len(lr_list)
        # # print("minn_lr={}, maxx_lr={}, avg_lr={}".format(minn_lr, maxx_lr, avg_lr))
        # plt.plot(t_list, lr_list)
        # plt.xlabel('local iters')
        # plt.ylabel('learning rate')
        # plt.tight_layout()
        # plt.savefig('./images/{}_{}_lr.png'.format(client, t), bbox_inches='tight')
        # plt.show()

        return Net.state_dict()
    
    def fixPointUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,  client,  t, next_phi_t_2, theta_i, 
                    shuffle=True, beta=0.9, min_lr=1e-6, max_lr=1.0, is_weight_decay=False, weight_decay=0.99):
        opti.zero_grad()
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=shuffle)
        init_lr = 0.0
        for param_group in opti.param_groups:
            init_lr = param_group['lr']
            break 
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            loss.backward()

        x = Symbol('x')
        item1, item2, item3 = 0.0, 0.0, 0.0
        left_1, left_2 = 0.0, 0.0
        with torch.no_grad():
            for group in opti.param_groups:
                var = group['params_name']
                grad = group['params'][0].grad.reshape(-1)
                phi_t_1 = global_parameters[var].reshape(-1)
                item1 = float(phi_t_1.dot(phi_t_1))
                item2 = float(phi_t_1.dot(grad))
                item3 = float(grad.dot(grad))
                left_1 += item2 - item3 * x
                left_2 += item1 - 2 * item2 * x + item3 * x * x
        left_eq = (theta_i * beta) / (next_phi_t_2 * (1 - beta)) * left_1 * (1 + log((theta_i * left_2) / next_phi_t_2))
        # f = left_eq - x 
        # mnist 上计算方法  
        # lr = float(sympy.nsolve(f, 1e-3))
        # lr =  sympy.nsolve(left_eq, x)lr = float(lr)
        # lr = float(lr)
        # lr = sympy.solveset(left_eq, x)
        # print(is_weight_decay, type(lr), lr)
        # lr = float(sympy.pretty(lr))
        ans  = sympy.solve(left_eq, x)
        # print(type(ans))
        real_lr = []
        for val in ans:
            if not isinstance(val, sympy.core.add.Add):
                real_lr.append(val)
        if len(real_lr) > 0:
            pos_lr, neg_lr = [], []
            for val in real_lr:
                if val > 0:
                    pos_lr.append(val)
                else:
                    neg_lr.append(val)
            # print('real_lr={}, pos_lr={}'.format(real_lr, pos_lr))
            cnt_pos, cnt_neg = len(pos_lr), len(neg_lr)
            if cnt_neg == 1 and cnt_pos == 2:
                lr = max(pos_lr)
            elif cnt_pos == 3:
                lr = min(pos_lr)
            elif cnt_pos == 1:
                lr = min(pos_lr)
            else:
                lr = min_lr
        else:
            lr = min_lr
        if lr <= 0:
            lr = min_lr
        while lr >= max_lr:
            lr = (1 - weight_decay) * float(lr)
            # lr = min([(1 - weight_decay) * float(lr), max_lr])
        # print(lr)
        if is_weight_decay:
            lr = init_lr * weight_decay  + (1 - weight_decay) * float(lr)
        # print(is_weight_decay, type(lr), lr)
        for param_group in opti.param_groups:
            param_group['lr'] = lr 

        opti.step()
        # print('t={}, {},lr={}'.format(t, client, lr))
        return Net.state_dict()

    def preAdlrUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, client, t, next_params, theta_i, 
                    shuffle=True, beta=0.9, min_lr=1e-6, is_weight_decay=False, weight_decay=0.99):
        
        opti.zero_grad()
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=shuffle)
        init_lr = 0.0
        for param_group in opti.param_groups:
            init_lr = param_group['lr']
            break 
        for data, label in self.train_dl:
            data, label = data.to(self.dev), label.to(self.dev)
            preds = Net(data)
            loss = lossFun(preds, label)
            loss.backward()
        item1 = 0.0
        item2 = 0.0
        with torch.no_grad():
            for group in opti.param_groups:
                var = group['params_name']
                # group['params'][0].grad = group['params'][0].grad / len(self.train_dl)
                grad = group['params'][0].grad.reshape(-1)
                w_t = global_parameters[var].reshape(-1)
                w_next_t = next_params[var].reshape(-1)
                # print(grad.device, w_t.device, w_next_t.device)
                item1 += grad.dot(w_t - w_next_t)
                item2 +=  grad.dot(grad)
        lr = item1 / (1.0 + item2)
        if lr <= 0:
            lr = min_lr
        if is_weight_decay:
            lr = weight_decay * init_lr + (1 - weight_decay) * lr
        for param_group in opti.param_groups:
            param_group['lr'] = float(lr)

        opti.step()
        # print('t={}, {},lr={}'.format(t, client, lr))
        return Net.state_dict()


        '''
        param: localEpoch 当前Client的迭代次数
        param: localBatchSize 当前Client的batchsize大小
        param: Net Server共享的模型
        param: lossFun 损失函数
        param: opti 优化函数
        param: global_parmeters 当前通讯中最全局参数
        return: 返回当前Client基于自己的数据训练得到的新的模型参数
        '''
        # 加载全局参数
        Net.load_state_dict(global_parameters, strict=True)
        
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=False)
        
        local_gradients = []
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                label = label.to(torch.int64)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()

                # 收集本地梯度
                gradients = [param.grad.clone() for param in Net.parameters()]
                local_gradients.append(gradients)
                
                opti.step()
        
        # 返回最后一次迭代的模型参数和梯度
        return Net.state_dict(), local_gradients

    def local_val(self):
        pass


class ClientsGroup(object):

    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端
        param: is_export 表示导出数据分布
    '''
    def __init__(self, dataSetName, isIID, numOfClients, dev, alpha=1.0, resize=224, is_export=False, split="letters", rearrange=1):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.alpha = alpha
        self.clients_set = {}
        self.resize = resize
        self.is_export = is_export
        self.test_data_loader = None
        self.split = split
        self.rearrange = rearrange

        if self.data_set_name in ('mnist'):
            self.dataSetBalanceAllocation()
        else:
            self.dataset_allocation()

    def dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid, resize=self.resize)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        if self.is_export:
            export_dict = {}
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)
            if self.is_export:
                count_label = Counter(local_label)
                print(count_label)
                count_label = {str(k): count_label[k] for k in count_label}
                export_dict['client{}'.format(i)] = count_label
            
            if self.rearrange:
                # 数据重新排序
                sorted_indices = np.argsort(local_label)

                sample = random.randint(0,9)
                start_index = np.where(sorted_indices == sample)[0][0]
                sorted_indices = np.concatenate((sorted_indices[start_index:], sorted_indices[:start_index]))

                local_data = local_data[sorted_indices]
                local_label = local_label[sorted_indices]
                
            # print(local_label)
            
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev, theta=1.0/self.num_of_clients)
            self.clients_set['client{}'.format(i)] = someone
        if self.is_export:
            iid = 1 if self.is_iid else 0
            data_save_dir = "./outs/{}/IID_{}".format(self.data_set_name, iid)
            if not os.path.exists(data_save_dir):
                os.makedirs(data_save_dir) 
            data_save_file = os.path.join(data_save_dir, 'client_{}_alpha_{}_resize_{}.json'.format(self.num_of_clients, self.alpha, self.resize))
            with open(data_save_file, "w", encoding='utf-8') as f:
                json.dump(export_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    def dirichlet_split_noniid(self, train_labels, alpha, n_clients):
        '''
        Dirichlet distribution with parameter alpha divides the data index into n_clients subsets
        '''
        n_classes = train_labels.max() + 1

        # (K, N) class label distribution matrix X, record how much each client occupies in each class
        label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes) 
        
        # Record the sample subscript corresponding to each K category
        class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(n_classes)]

        # Record the index of N clients corresponding to the sample set respectively
        client_idcs = [[] for _ in range(n_clients)] 

        for c, fracs in zip(class_idcs, label_distribution):
            # np.split divides the samples of class k into N subsets according to the proportion
            # for i, idcs is to traverse the index of the sample set corresponding to the i-th client
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
                client_idcs[i] += [idcs]
        
        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

        return client_idcs


    def dataset_allocation(self):
        dataset = GetDataSet(self.data_set_name, self.is_iid, resize=self.resize, split=self.split)

        test_data = dataset.test_data
        test_label = dataset.test_label
        # print(test_data.shape, test_label.shape)
        self.test_data_loader = DataLoader(MyDataset(test_data, test_label, transforms=dataset.test_transforms), batch_size=100, shuffle=False)
        train_data = dataset.train_data
        train_label = dataset.train_label
        
        np.random.seed(42)
        if self.is_iid:
            all_idxs = [i for i in range(dataset.train_data_size)]
            num_items = int(dataset.train_data_size / self.num_of_clients)
            client_idcs = []
            for i in range(self.num_of_clients):
                client_idcs.append(set(np.random.choice(all_idxs, num_items, replace=False)))
                all_idxs = list(set(all_idxs) - client_idcs[-1])
            for i in range(len(client_idcs)):
                client_idcs[i] = np.array(list(client_idcs[i]))
        else:
            client_idcs = self.dirichlet_split_noniid(train_labels=np.array(dataset.train_label), alpha=self.alpha, n_clients=self.num_of_clients)

        if self.is_export:
            export_dict = {}
        
        for i, idc in enumerate(client_idcs):
            
            local_data = train_data[idc]
            local_label = train_label[idc]
            local_theta = int(idc.shape[0]) / dataset.train_data_size

            # 数据重新排序
            if self.rearrange:
                if self.data_set_name in ('cifar10'):
                    sorted_indices, _ = torch.sort(local_label)
                    sample = random.randint(0,9)
                    start_index = torch.where(sorted_indices == sample)[0][0].item()
                    rearrange_label = torch.cat((sorted_indices[start_index:],sorted_indices[:start_index]))
                    rearrange_data = torch.cat((local_data[start_index:],local_data[:start_index]))
                    local_data = rearrange_data
                    local_label = rearrange_label

                if self.data_set_name in ('emnist'):
                    sorted_indices, _ = torch.sort(local_label)
                    print(sorted_indices)
                    sample = random.randint(1,26)
                    print(sample)
                    start_index = torch.where(sorted_indices == sample)[0][0].item()
                    rearrange_label = torch.cat((sorted_indices[start_index:],sorted_indices[:start_index]))
                    rearrange_data = torch.cat((local_data[start_index:],local_data[:start_index]))
                    local_data = rearrange_data
                    local_label = rearrange_label
            
            someone = client(MyDataset(local_data, local_label, transforms=dataset.train_transforms), self.dev, theta=local_theta)
            if self.is_export:
                # count_label = Counter(local_label)
                # count_label = {str(k): count_label[k] for k in count_label}
                count_label = {}
                for label in local_label:
                    n_label = str(int(label))
                    if n_label not in count_label:
                        count_label[n_label] = 0
                    count_label[n_label] += 1
                export_dict['client{}'.format(i)] = count_label
            # print(type(someone))
            self.clients_set['client{}'.format(i)] = someone

        if self.is_export:
            iid = 1 if self.is_iid else 0
            data_save_dir = "./outs/{}/IID_{}".format(self.data_set_name, iid)
            if not os.path.exists(data_save_dir):
                os.makedirs(data_save_dir) 
            data_save_file = os.path.join(data_save_dir, 'client_{}_alpha_{}_resize_{}.json'.format(self.num_of_clients, self.alpha, self.resize))
            with open(data_save_file, "w", encoding='utf-8') as f:
                json.dump(export_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

        # print(dataset.num_cls)
        # import matplotlib.pyplot as plt
        # num_cls = dataset.num_cls
        # train_labels = np.array(dataset.train_label)
        # plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, bins=np.arange(min(train_labels)-0.5, max(train_labels) + 1.5, 1),
        #             label=["Client {}".format(i) for i in range(self.num_of_clients)], rwidth=0.5)
        # plt.xticks(np.arange(num_cls), dataset.train_data_classes)
        # plt.title('data={}_iid_set={}'.format(self.data_set_name, self.is_iid))
        # plt.savefig('./images/data/data={}_settings={}_clients={}.png'.format(self.data_set_name, self.is_iid, self.num_of_clients), bbox_inches='tight')
        # plt.savefig('./images/data/data={}_settings={}_clients={}.eps'.format(self.data_set_name, self.is_iid, self.num_of_clients), bbox_inches='tight')
        # plt.legend()
        # plt.show()


if __name__=="__main__":
    print('client')
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # MyClients = ClientsGroup(dataSetName='mnist', isIID=True, numOfClients=10, dev=dev, alpha=1.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='mnist', isIID=False, numOfClients=10, dev=dev, alpha=1.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar10', isIID=True, numOfClients=10, dev=dev, alpha=1.0, resize=32, is_export=True)
    MyClients = ClientsGroup(dataSetName='emnist', isIID=True, numOfClients=10, dev=dev, alpha=1.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar10', isIID=False, numOfClients=100, dev=dev, alpha=1.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar10', isIID=False, numOfClients=10, dev=dev, alpha=0.5, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar10', isIID=False, numOfClients=100, dev=dev, alpha=0.1, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar100', isIID=False, numOfClients=100, dev=dev, alpha=100.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar100', isIID=False, numOfClients=100, dev=dev, alpha=1.0, resize=32, is_export=True)
    # MyClients = ClientsGroup(dataSetName='cifar100', isIID=False, numOfClients=100, dev=dev, alpha=0.5, resize=32, is_export=True)

