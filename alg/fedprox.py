import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
from tqdm import tqdm
from utils import BaseServer


class FedProxServer(BaseServer):

    def __init__(self, args):
        super(FedProxServer, self).__init__(args=args)
        self.is_local_acc = args['is_local_acc'] if 'is_local_acc' in args else False
        self.prox_mu = args['prox_mu'] if 'prox_mu' in args else 0.001

    def run(self):
        accuracy_list = []
        phi_list = []
        loss_list = []
        if 'parmas_mode' in self.args:
            model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}_local_e_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args['epoch'])
            print('model_save_dir={}'.format(model_save_dir))
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)
        accuracy, loss = self.eval(t=0)
        accuracy_list.append(accuracy)
        loss_list.append(loss)
        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            cur_phi = 0
            for client in tqdm(clients_in_comm):
                local_parameters = self.myClients.clients_set[client].localUpdate_prox(localEpoch=self.args['epoch'], localBatchSize=self.args['batchsize'], Net=self.net, lossFun=self.loss_func, 
                                                                                       opti=self.optimers[client], global_parameters=self.global_parameters, prox_mu=self.prox_mu)
                
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

        # self.test_txt.write('phi_list={}'.format(phi_list))
        s = 'dataset_{}_IID_{}_{}_{}_cli_{}_frac_{}_local_e_{}'.format(self.args['dataset'], self.args['IID'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'], self.args["cfraction"], self.args['epoch'])
        self.test_txt.write(s + " start\n")
        self.test_txt.write("\n")
        self.test_txt.write('accuracy_list={}'.format(accuracy_list))
        self.test_txt.write("\n")
        self.test_txt.write('loss_list={}'.format(loss_list))
        self.test_txt.write("end\n")
        self.test_txt.close()
        print('phi_list={}'.format(phi_list))
        print('accuracy_list={}'.format(accuracy_list))
        print('loss_list={}'.format(loss_list))