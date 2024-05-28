import math
from tokenize import Number
from traceback import print_tb
from turtle import left
from matplotlib.pyplot import flag
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet, MyDataset
import sympy 
from sympy import Matrix, Symbol, false, log, posify
import random


class client(object):
    
    def __init__(self, trainDataSet, dev, theta):
        self.train_ds = trainDataSet
        self.dev = dev
        self.theta = theta
        self.train_dl = None
        self.local_parameters = None

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
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
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                opti.zero_grad()
                loss.backward()
                opti.step()
        return Net.state_dict()
    
    def localUpdate_prox(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, prox_mu):
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
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                opti.zero_grad()
                proximal_term = 0.0
                for w, w_t in zip(Net.parameters(), global_parameters.parameters()):
                    proximal_term += (w - w_t).norm(2)
                loss = lossFun(preds, label) + (prox_mu / 2) * proximal_term
                loss.backward()
                opti.step()
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
                    shuffle=True, beta=0.9, min_lr=1e-6, weight_decay=0.99):
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
                # 加载到GPU上
                data, label = data.to(self.dev), label.to(self.dev)
                # 模型上传入数据
                preds = Net(data)
                # 计算损失函数
                '''
                    这里应该记录一下模型得损失值 写入到一个txt文件中
                '''
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
                        for key, var in Net.state_dict().items():
                            local_parameters[key] = var

                        for group in opti.param_groups:
                            var = group['params_name']
                            grad = group['params'][0].grad.reshape(-1)
                            wt = local_parameters[var].reshape(-1)

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
                    f = x - ((beta * theta_i * theta_i) / ((1 - beta) * phi_t)) * left_1 * left_2
                    ans = sympy.solve(f)
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

                    lr = weight_decay * last_lr + (1.0 - weight_decay) * lr
                
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

    def preAdlrUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, client, t, next_params, theta_i, 
                    shuffle=True, beta=0.9, min_lr=1e-6, weight_decay=0.99):
        
        opti.zero_grad()
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=shuffle)
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
                grad = group['params'][0].grad.reshape(-1)
                w_t = global_parameters[var].reshape(-1)
                w_next_t = next_params[var].reshape(-1)
                # print(grad.device, w_t.device, w_next_t.device)
                item1 += grad.dot(w_t - w_next_t)
                item2 +=  grad.dot(grad)
        lr = (weight_decay * item1) / (1.0 - weight_decay + weight_decay * item2)
        if lr <= 0:
            lr = min_lr

        for param_group in opti.param_groups:
            param_group['lr'] = float(lr)

        opti.step()
        # print('t={}, {},lr={}'.format(t, client, lr))
        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):

    '''
        param: dataSetName 数据集的名称
        param: isIID 是否是IID
        param: numOfClients 客户端的数量
        param: dev 设备(GPU)
        param: clients_set 客户端
    '''
    def __init__(self, dataSetName, isIID, numOfClients, dev, alpha=1.0, num_items=None, resize=224):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.alpha = alpha
        self.clients_set = {}
        self.num_items = num_items
        self.test_data_loader = None
        self.resize = resize

        if  self.data_set_name in ('mnist'):
            self.dataSetBalanceAllocation()
        else:
            self.dataset_allocation()

    def dataSetBalanceAllocation(self):

        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2

       
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):

            ## shards_id1
            ## shards_id2
            ## 是所有被分得的两块数据切片
            # 0 2 4 6...... 偶数
            shards_id1 = shards_id[i * 2]
            # 0+1 = 1 2+1 = 3 .... 奇数
            shards_id2 = shards_id[i * 2 + 1]
            #
            # 例如shard_id1 = 10
            # 10* 300 : 10*300+300
            # 将数据以及的标签分配给该客户端
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            local_label = np.argmax(local_label, axis=1)

            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev, theta=1.0/self.num_of_clients)
            self.clients_set['client{}'.format(i)] = someone

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
        dataset = GetDataSet(self.data_set_name, self.is_iid, self.resize)
        
        test_data = dataset.test_data
        test_label = dataset.test_label
        self.test_data_loader = DataLoader(MyDataset(test_data, test_label, transforms=dataset.test_transforms), batch_size=100, shuffle=False)
        train_data = dataset.train_data
        train_label = dataset.train_label
        
        np.random.seed(42)
        if self.is_iid == 1:
            all_idxs = [i for i in range(dataset.train_data_size)]
            num_items = int(dataset.train_data_size / self.num_of_clients)
            client_idcs = []
            for i in range(self.num_of_clients):
                client_idcs.append(set(np.random.choice(all_idxs, num_items, replace=False)))
                all_idxs = list(set(all_idxs) - client_idcs[-1])
            for i in range(len(client_idcs)):
                client_idcs[i] = np.array(list(client_idcs[i]))
        elif self.is_iid == 2:
            all_idxs = [i for i in range(dataset.train_data_size)]
            num_items = self.num_items
            client_idcs = []
            for i in range(self.num_of_clients):
                client_idcs.append(set(np.random.choice(all_idxs, num_items[i], replace=False)))
                all_idxs = list(set(all_idxs) - client_idcs[-1])
            for i in range(len(client_idcs)):
                client_idcs[i] = np.array(list(client_idcs[i]))
        elif self.is_iid == -1:
            all_idxs = [i for i in range(dataset.train_data_size)]
            num_items = self.num_items
            client_idcs = []
            for i in range(self.num_of_clients):
                client_idcs.append(set(np.random.choice(all_idxs, num_items[i], replace=False)))
                all_idxs = list(set(all_idxs) - client_idcs[-1])
            for i in range(len(client_idcs)):
                client_idcs[i] = np.array(list(client_idcs[i]))
            order = np.concatenate(client_idcs)
            train_data = dataset.train_data[order]
            train_label = dataset.train_label[order]
            client_idcs = self.dirichlet_split_noniid(train_labels=train_label, alpha=self.alpha, n_clients=self.num_of_clients)
        else:
            client_idcs = self.dirichlet_split_noniid(train_labels=np.array(dataset.train_label), alpha=self.alpha, n_clients=self.num_of_clients)

        for i, idc in enumerate(client_idcs):
            # print(type(idc), idc.shape)
            local_data = train_data[idc]
            local_label = train_label[idc]
            local_theta = int(idc.shape[0]) / dataset.train_data_size
            someone = client(MyDataset(local_data, local_label, transforms=dataset.train_transforms), self.dev, theta=local_theta)
            self.clients_set['client{}'.format(i)] = someone

        # print(dataset.num_cls)
        import matplotlib.pyplot as plt
        num_cls = dataset.num_cls
        train_labels = np.array(dataset.train_label)
        plt.hist([train_labels[idc] for idc in client_idcs], stacked=True, bins=np.arange(min(train_labels) - 0.5, max(train_labels) + 1.5, 1),
                    label=["Client {}".format(i) for i in range(self.num_of_clients)], rwidth=0.5)
        # print(["Client {}".format(i) for i in range(self.num_of_clients)])
        plt.xticks(np.arange(num_cls), dataset.train_data_classes)
        plt.legend(["Client {}".format(i) for i in range(self.num_of_clients)])
        plt.title('data={}_iid_set={}'.format(self.data_set_name, self.is_iid))
        plt.savefig('./images/noniid/data={}_settings={}_alpha={}.png'.format(self.data_set_name, self.is_iid, self.alpha), bbox_inches='tight')
        plt.savefig('./images/noniid/data={}_settings={}_alpha={}.eps'.format(self.data_set_name, self.is_iid, self.alpha), bbox_inches='tight')
        # plt.show()


if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 0)

    # print(client)
    # print(MyClients.clients_set['client10'].train_ds[0:10])
    # train_ids = MyClients.clients_set['client10'].train_ds[0:10]
    # i = 0
    # for x_train in train_ids[0]:
    #     print("client10 数据:"+str(i))
    #     print(x_train)
    #     i = i+1
    # print(MyClients.clients_set['client11'].train_ds[400:500])


