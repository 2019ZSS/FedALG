from cgi import test
import os
import argparse
from sympy import true
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from Models import LinearNet, Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from model.WideResNet import WideResNet
import clients
import matplotlib.pyplot as plt


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def adjust_learning_rate(optimizer, lr):
    '''
    adjust learning rate of optimizer
    '''
    for param_group in optimizer.param_groups:
        param_group["lr"] = float(lr)


def solve_fedavg(args):
    test_mkdir('./logs')
    test_txt = open("./logs/test_accuracy.txt", mode="a")
    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = None

    num_outputs = 10
    if args['dataset'] == 'mnist':
        num_outputs = 10
    elif args['dataset'] == 'mnist_v2':
        num_outputs = 10
    elif args['dataset'] == 'emnist':
        num_outputs = 62

    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_linear':
        net = LinearNet(num_outputs=num_outputs)
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)
    loss_func = F.cross_entropy
    optimers = {}
    for i in range(args['num_of_clients']):
        optimers['client{}'.format(i)] = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = clients.ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}

    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    theta_list = {}
    for i in range(args['num_of_clients']):
        client = 'client{}'.format(i)
        theta_list[client] = myClients.clients_set[client].theta 

    if 'parmas_mode' in args:
        model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(args['IID'], args['dataset'], args['alg'], args['model_name'], args['num_of_clients'])
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

    phi_list = []
    accuracy_list = []
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        order = np.random.permutation(args['num_of_clients'])
        print("order:", len(order), order)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        print("client: " + str(clients_in_comm))
        sum_parameters = None
        cur_phi = 0
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                            loss_func, optimers[client], global_parameters)
            for var in local_parameters:
                cur_phi += theta_list[client] * float(local_parameters[var].reshape(-1).dot(local_parameters[var].reshape(-1)))

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = theta_list[client] * var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] +  theta_list[client] * local_parameters[var]
                    
        phi_list.append(cur_phi)
        for var in global_parameters:
            global_parameters[var] = sum_parameters[var]

        net.load_state_dict(global_parameters, strict=True)
        
        if 'parmas_mode' in args:
            torch.save(net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(i + 1)))

        sum_accu = 0
        num = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n"+'accuracy: {}'.format(sum_accu / num))
        
        test_txt.write("communicate round "+str(i+1)+"  ")
        test_txt.write('accuracy: '+str(float(sum_accu / num))+"\n")
        accuracy_list.append(float(sum_accu / num))
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['alg'], args['model_name'],
                                                                i, args['epoch'],args['batchsize'], args['learning_rate'], 
                                                                args['num_of_clients'],args['cfraction'])))
    test_txt.write('phi_list={}'.format(phi_list))
    test_txt.write('accuracy_list={}'.format(accuracy_list))
    print('phi_list={}'.format(phi_list))
    print('accuracy_list={}'.format(accuracy_list))
    test_txt.close()


def solve_adp(args):
    test_mkdir('./logs')
    test_txt = open("./logs/test_accuracy.txt", mode="a")
    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_outputs = 10
    if args['dataset'] == 'mnist':
        num_outputs = 10
    elif args['dataset'] == 'mnist_v2':
        num_outputs = 10
    elif args['dataset'] == 'emnist':
        num_outputs = 62

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_linear':
        net = LinearNet(num_outputs=num_outputs)
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    eta = args['eta']
    beta_1 = args['beta_1']
    beta_2 = args['beta_2']
    tau = args['tau']
    

    net = net.to(dev)
    loss_func = F.cross_entropy
    optimers = {}
    for i in range(args['num_of_clients']):
        optimers['client{}'.format(i)] = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = clients.ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    global_parameters = {}
    diff_t = {}
    vt = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
        diff_t[key] = torch.zeros(size=var.shape).to(device=dev)
        vt[key] = torch.full(size=var.shape, fill_value=tau*tau).to(device=dev)

    theta_list = {}
    for i in range(args['num_of_clients']):
        client = 'client{}'.format(i)
        theta_list[client] = myClients.clients_set[client].theta 

    phi_list = []
    accuracy_list = []
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))
    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        print("order:", len(order), order)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        # sum_parameters = None
        count_parameters = {}
        cur_phi = 0
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                            loss_func, optimers[client], global_parameters)
            for var in local_parameters:
                cur_phi += theta_list[client] * float(local_parameters[var].reshape(-1).dot(local_parameters[var].reshape(-1)))                                     

            for var in local_parameters:
                if var not in count_parameters:
                    count_parameters[var] = theta_list[client] * (local_parameters[var] - global_parameters[var])
                else:
                    count_parameters[var] +=  theta_list[client] * (local_parameters[var] - global_parameters[var])

        phi_list.append(cur_phi)
        for var in global_parameters:
            # global_parameters[var] = (sum_parameters[var] / num_in_comm)
            diff_t[var] = beta_1 * diff_t[var] + (1 - beta_1) * (count_parameters[var])
            if args['alg'] == 'fedadagrad':
                vt[var] = vt[var] + diff_t[var] * diff_t[var]
            elif args['alg'] == 'fedyogi':
                twice_t = diff_t[var] * diff_t[var]
                vt[var] = vt[var] - (1 - beta_2) * twice_t * torch.sign(vt[var] - twice_t)
            elif args['alg'] == 'fedadam':
                vt[var] = beta_2 * vt[var] + (1 - beta_2) * diff_t[var] * diff_t[var]
            else:
                raise NotImplementedError('{} not implemented!'.format(args['alg']))
            global_parameters[var] = global_parameters[var] + eta * (diff_t[var] / (torch.sqrt(vt[var]) + tau))
        
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        # 载入测试集
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n"+'accuracy: {}'.format(sum_accu / num))

        test_txt.write("communicate round "+str(i+1)+"  ")
        test_txt.write('accuracy: '+str(float(sum_accu / num))+"\n")
        accuracy_list.append(float(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['alg'], args['model_name'],
                                                                i, args['epoch'],args['batchsize'], args['learning_rate'], 
                                                                args['num_of_clients'],args['cfraction'])))
    print('phi_list={}'.format(phi_list))
    print('accuracy_list={}'.format(accuracy_list))
    test_txt.write('phi_list={}'.format(phi_list))
    test_txt.write('accuracy_list={}'.format(accuracy_list))
    test_txt.close()


def solve_adlr(args):
    test_mkdir('./logs')
    test_txt = open("./logs/test_accuracy.txt", mode="a")
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_outputs = 10
    if args['dataset'] == 'mnist':
        num_outputs = 10
    elif args['dataset'] == 'mnist_v2':
        num_outputs = 10
    elif args['dataset'] == 'emnist':
        num_outputs = 62

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_linear':
        net = LinearNet(num_outputs=num_outputs)
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)

    net = net.to(dev)

    def init_params():
        global_parameters = {}
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
            
        loss_func = F.cross_entropy

        optimers = {}
        for i in range(args['num_of_clients']):
            opt_list = []
            for name, params in net.named_parameters():
                opt_d = {
                    'params': params,
                    'params_name': name,
                }
                opt_list.append(opt_d)
            optimers['client{}'.format(i)] = optim.SGD(opt_list, lr=args['learning_rate'])
        return optimers, loss_func, global_parameters    

    myClients = clients.ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    optimers, loss_func, global_parameters = init_params()

    T = args['num_comm']
    beta = args['beta']
    min_lr = args['min_lr']
    weight_decay = args['weight_decay'] if 'weight_decay' in args else 0.99
    shuffle = args['shuffle'] if 'shuffle' in args else True
    
    theta_list = {}
    for i in range(args['num_of_clients']):
        client = 'client{}'.format(i)
        theta_list[client] = myClients.clients_set[client].theta 

    phi_t = 0.0
    for var in global_parameters:
        phi_t += float(global_parameters[var].reshape(-1).dot(global_parameters[var].reshape(-1)))

    lr_list = []
    phi_list = []
    accuracy_list = []
    for t in range(T):
        print("communicate round {}".format(t + 1))
        order = np.random.permutation(args['num_of_clients'])
        print("order:", len(order), order)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        print("phi_t={}".format(phi_t))
        clients_lr = {}
        next_phi_t = 0    
        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].adplrUpdate(localEpoch=args['epoch'], localBatchSize=args['batchsize'], Net=net, lossFun=loss_func, 
                                                opti=optimers[client], global_parameters=global_parameters, client=client, t=t,
                                                phi_t=phi_t, theta_i=theta_list[client],  shuffle=shuffle, beta=beta, min_lr=min_lr, weight_decay=weight_decay)
            lr = 0.0
            for param in optimers[client].param_groups:
                lr = param['lr']
                break
            
            clients_lr[client] = lr

            for var in local_parameters:
                next_phi_t += theta_list[client] * float(local_parameters[var].reshape(-1).dot(local_parameters[var].reshape(-1)))

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = theta_list[client] * var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + theta_list[client] * local_parameters[var]
        
        phi_list.append(next_phi_t)
        lr_list.append(clients_lr)

        for var in global_parameters:
            global_parameters[var] = sum_parameters[var]
        
        phi_t = next_phi_t
        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        for data, label in testDataLoader:
            data, label = data.to(dev), label.to(dev)
            preds = net(data)
            preds = torch.argmax(preds, dim=1)
            sum_accu += (preds == label).float().mean()
            num += 1
        print("\n"+'accuracy: {}'.format(sum_accu / num))

        test_txt.write("communicate round " + str(t + 1)+"  ")
        test_txt.write('accuracy: '+str(float(sum_accu / num))+"\n")
        accuracy_list.append(float(sum_accu / num))
        if (t + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['alg'], args['model_name'],
                                                                i, args['epoch'],args['batchsize'], args['learning_rate'], 
                                                                args['num_of_clients'],args['cfraction'])))
    
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
        # plt.tight_layout()
        plt.savefig('./images/lr/{}_{}_lr.png'.format(key, args['IID']), bbox_inches='tight')
        # plt.show()
    
    print('phi_list={}'.format(phi_list))
    print('accuracy_list={}'.format(accuracy_list))
    test_txt.write('phi_list={}'.format(phi_list))
    test_txt.write('accuracy_list={}'.format(accuracy_list))
    test_txt.close()
    

def solve_adplr(args):
    test_mkdir('./logs')
    test_txt = open("./logs/test_accuracy.txt", mode="a")
    test_mkdir(args['save_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_outputs = 10
    if args['dataset'] == 'mnist':
        num_outputs = 10
    elif args['dataset'] == 'mnist_v2':
        num_outputs = 10
    elif args['dataset'] == 'emnist':
        num_outputs = 62

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN(num_outputs=num_outputs)
    elif args['model_name'] == 'mnist_linear':
        net = LinearNet(num_outputs=num_outputs)
    elif args['model_name'] == 'wideResNet':
        net = WideResNet(depth=28, num_classes=10).to(dev)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    
    net = net.to(dev)

    def init_params():
        global_parameters = {}
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
            
        loss_func = F.cross_entropy

        optimers = {}
        for i in range(args['num_of_clients']):
            opt_list = []
            for name, params in net.named_parameters():
                opt_d = {
                    'params': params,
                    'params_name': name,
                }
                opt_list.append(opt_d)
            optimers['client{}'.format(i)] = optim.SGD(opt_list, lr=args['learning_rate'])
        return optimers, loss_func, global_parameters     

    myClients = clients.ClientsGroup(args['dataset'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader
    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    theta_list = {}
    for i in range(args['num_of_clients']):
        client = 'client{}'.format(i)
        theta_list[client] = myClients.clients_set[client].theta

    optimers, loss_func, global_parameters = init_params()

    T = args['num_comm']
    pre_t = args['pre_t']
    eps = args['eps'] 
    beta = args['beta']
    min_lr = args['min_lr']
    shuffle = args['shuffle'] if 'shuffle' in args else True
    conv_lr_list = [
        {'client{}'.format(i): args['learning_rate'] for i in range(args['num_of_clients'])} for _ in range(T)
    ]
    model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(args['IID'], args['dataset'], args['alg'], args['model_name'], args['num_of_clients'])
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    for idx in range(pre_t):
        lr_list = []
        optimers, loss_func, global_parameters = init_params()
        print('Pre round {}'.format(idx))
        for t in range(T):
            order = np.random.permutation(args['num_of_clients'])
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
            client_lrs = {}
            sum_parameters = None
            if idx == 0:
                next_params_path = os.path.join(args['init_model'], '{}.pth'.format(t + 1))
            else:
                next_params_path = os.path.join(model_save_dir, '{}.pth'.format(t + 1))
            next_params = torch.load(next_params_path, map_location=lambda storage, loc: storage.cuda())
            for client in tqdm(clients_in_comm):
                local_parameters = myClients.clients_set[client].preAdlrUpdate(localEpoch=args['epoch'], localBatchSize=args['batchsize'], Net=net, lossFun=loss_func, 
                                                    opti=optimers[client], global_parameters=global_parameters, client=client, t=t, theta_i=theta_list[client],
                                                    next_params=next_params, shuffle=shuffle, beta=beta, min_lr=min_lr)
                lr = 0.0
                for param in optimers[client].param_groups:
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
            
            for var in global_parameters:
                global_parameters[var] = sum_parameters[var]
            
            net.load_state_dict(global_parameters, strict=True)
            torch.save(net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            lr_list.append(client_lrs)
            
        conv_lr_list = lr_list
        flag = True
        for t in range(T):
            next_params_path = os.path.join(model_save_dir, '{}.pth'.format(t + 1))
            next_params = torch.load(next_params_path, map_location=lambda storage, loc: storage.cuda())
            sum = 0
            for var in global_parameters:
                sum += torch.sum(global_parameters[var].reshape(-1) - next_params[var].reshape(-1))
            if sum > eps:
                flag = False
                break
        if flag:
            break
        
    print(conv_lr_list)
    accuracy_list = []
    optimers, loss_func, global_parameters = init_params()
    for t in range(T):
        print("communicate round {}".format(t +1))
        order = np.random.permutation(args['num_of_clients'])
        print("order:", len(order), order)
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]
        sum_parameters = None
        for client in tqdm(clients_in_comm):
            adjust_learning_rate(optimizer=optimers[client], lr=conv_lr_list[t][client])
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                        loss_func, optimers[client], global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = theta_list[client] * var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + theta_list[client] * local_parameters[var]
        
        for var in global_parameters:
            global_parameters[var] = sum_parameters[var]

        net.load_state_dict(global_parameters, strict=True)
        sum_accu = 0
        num = 0
        with torch.no_grad():
            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                preds = net(data)
                preds = torch.argmax(preds, dim=1)
                sum_accu += (preds == label).float().mean()
                num += 1
        print("\n"+'accuracy: {}'.format(sum_accu / num))

        accuracy_list.append(float(sum_accu / num))
        test_txt.write("communicate round "+str(i+1)+"  ")
        test_txt.write('accuracy: '+str(float(sum_accu / num))+"\n")

        if (t + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'], '{}_{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['alg'], args['model_name'],
                                                                i, args['epoch'],args['batchsize'], args['learning_rate'], 
                                                                args['num_of_clients'],args['cfraction'])))
    print('accuracy_list={}'.format(accuracy_list))
    test_txt.write('accuracy_list={}'.format(accuracy_list))
    test_txt.close()
    