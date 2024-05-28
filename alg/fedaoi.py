import os
import re 
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
from tqdm import tqdm
from collections import defaultdict
from utils import BaseServer
import torch.nn as nn

def extract_numbers(s):
    return re.findall(r'\d+', s)

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class FedAOIServer(BaseServer):

    def __init__(self, args):
        super(FedAOIServer, self).__init__(args=args)
        self.is_local_acc = args['is_local_acc'] if 'is_local_acc' in args else False
        optimers = {}
        for i in range(self.args['num_of_clients']):
            opt_list = []
            for name, params in self.net.named_parameters():
                opt_d = {
                    'params': params,
                    'params_name': name,
                }
                opt_list.append(opt_d)
            optimers['client{}'.format(i)] = torch.optim.SGD(opt_list, lr=self.args['learning_rate'])
        self.optimers = optimers
        self.client_update_arr = args['client_update_arr'] if 'client_update_arr' in args else None

    def run(self):
        accuracy_list = []
        loss_list = []
        phi_list = []
        if 'parmas_mode' in self.args:
            model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}_local_e_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args['epoch'])
            print('model_save_dir={}'.format(model_save_dir))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
        accuracy, loss = self.eval(t=0)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        # max_bound = {
        #     'max_local_grad_bound': 0,
        #     'max_L_bound': 0
        # }
        max_bound = None
        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            cur_phi = 0
            for client in tqdm(clients_in_comm):
                # 客户端索引
                client_idx = int(''.join(extract_numbers(client)))
                # 更新长度
                update_sum = 35 + sum([self.client_update_arr[client_idx][j] for j in range(0, t + 1)]) # cifar10---20人---20,30人---25,50人---35
                # 总数组长度
                total_sum = len(self.client_update_arr[client_idx])
                # print(total_sum)
                # 数据更新比例
                update_sum = min(update_sum, total_sum) / total_sum
                # print(update_sum) 
                # 数据总长度
                local_data_len = len(self.myClients.clients_set[client].train_ds)
                # need_len 必须是整数
                need_len = min(int(update_sum * local_data_len), local_data_len)
                need_len = max(need_len, 1)
                # print(need_len)
                indices = torch.arange(0, need_len)

                # fedavg 和 fedaoi --- fedavg的时候main函数里面使用全0数组, fedaoi选择最优数组
                # local_parameters = self.myClients.clients_set[client].localPartialUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                #                                                             self.loss_func, self.optimers[client], self.global_parameters, max_bound, indices)
                
                # fedprox --- fedprox的时候main函数里面使用全0数组
                # local_parameters = self.myClients.clients_set[client].localPartialUpdate_prox(self.args['epoch'], self.args['batchsize'], self.net,
                #                                                             self.loss_func, self.optimers[client], self.global_parameters, max_bound, indices)

                
                # fedbabu --- ffedbabu的时候main函数里面使用全0数组
                # local_parameters = self.myClients.clients_set[client].localPartialUpdate_babu(self.args['epoch'], self.args['batchsize'], self.net,
                #                                                             self.loss_func, self.optimers[client], self.global_parameters, max_bound, indices)
                # # fedbabu 微调
                # local_parameters = self.myClients.clients_set[client].fine_tune(self.net, self.args['fine_tuning_epochs'], self.args['batchsize'], self.loss_func, self.optimers[client])

                # fedgh --- fedgh的时候main函数里面使用全0数组
                local_parameters = self.myClients.clients_set[client].localPartialUpdate_gh(self.args['epoch'], self.args['batchsize'], self.net,
                                                                             self.loss_func, self.optimers[client], self.global_parameters, max_bound, indices)


                if self.is_local_acc and t == self.num_comm - 1:
                    self.local_eval(t=t, local_parameters=local_parameters, client=client)

                for var in local_parameters:
                    cur_phi += theta_list[client] * float(local_parameters[var].reshape(-1).float().dot(local_parameters[var].reshape(-1).float()))

                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = theta_list[client] * var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] +  theta_list[client] * local_parameters[var]
            phi_list.append(cur_phi)
            for var in self.global_parameters:
                self.global_parameters[var] = sum_parameters[var]
            
            if 'parmas_mode' in self.args:
                torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            
            accuracy, loss = self.eval(t=t)
            accuracy_list.append(accuracy)
            loss_list.append(loss)
            print('accuracy: ' + str(accuracy) + "\n")
            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n")
            # if max_bound is not None: 
            #     print('max_local_grad_bound={}, max_L_bound={}'.format(max_bound['max_local_grad_bound'], max_bound['max_L_bound']))

        s = 'dataset_{}_IID_{}_{}_{}_cli_{}_frac_{}_local_e_{}'.format(self.args['dataset'], self.args['IID'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args["cfraction"], self.args['epoch'])
        self.test_txt.write(s + " start\n")
        self.test_txt.write('phi_list={}'.format(phi_list))
        self.test_txt.write("\n")
        self.test_txt.write('accuracy_list={}'.format(accuracy_list))
        self.test_txt.write("\n")
        self.test_txt.write('loss_list={}'.format(loss_list))
        self.test_txt.write("end\n")
        self.test_txt.close()
        # print('phi_list={}'.format(phi_list))
        print('accuracy_list={}'.format(accuracy_list))
        print('loss_list={}'.format(loss_list))
        # if max_bound is not None:
        #     print('max_local_grad_bound={}, max_L_bound={}'.format(max_bound['max_local_grad_bound'], max_bound['max_L_bound']))

# 以下只针对 fedgh
    def collect_local_proto(self, net, client):
        proto_mean = defaultdict(list)
        net.eval()
        with torch.no_grad():
            for j, batch in enumerate(client.train_dl, 0):
                img, label = tuple(t.to(self.dev) for t in batch)
                rep = net(img)  # Ensure the network returns both predictions and representations
                owned_classes = label.unique().detach().cpu().numpy()
                for cls in owned_classes:
                    filted_reps = [rep[i] for i in range(len(rep)) if label[i] == cls]
                    sum_filted_reps = filted_reps[0].detach()
                    for f in range(1, len(filted_reps)):
                        sum_filted_reps = sum_filted_reps + filted_reps[f].detach()
                    mean_filted_reps = sum_filted_reps / len(filted_reps)
                    proto_mean[cls].append(mean_filted_reps)
        return proto_mean

    def aggregate_protos(self, Protos_Mean):
        for cls in range(self.num_outputs):
            all_protos = [proto for client in Protos_Mean for proto in Protos_Mean[client][cls] if cls in Protos_Mean[client]]
            if all_protos:
                sum_proto = torch.zeros_like(all_protos[0])
                for proto in all_protos:
                    sum_proto += proto
                mean_proto = sum_proto / len(all_protos)
                self.global_proto[cls] = mean_proto

    def update_global_header(self, Protos_Mean):
        net_FC = FC(in_dim=self.num_outputs, out_dim=self.num_outputs).to(self.dev)
        optimizer = torch.optim.Adam(params=net_FC.parameters(), lr=self.args['learning_rate'])
        net_FC.train()
        criteria = torch.nn.CrossEntropyLoss()
        for client in Protos_Mean:
            for cls, reps in Protos_Mean[client].items():
                for rep in reps:
                    pred_server = net_FC(rep)
                    loss = criteria(pred_server.view(1, -1), torch.tensor(cls).view(1).to(self.dev))
                    loss.backward()
                    optimizer.step()
        return net_FC.state_dict()