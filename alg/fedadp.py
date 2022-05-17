import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
from tqdm import tqdm
from utils import BaseServer


class FedADPServer(BaseServer):

    def __init__(self, args):
        super(FedADPServer, self).__init__(args=args)
        self.eta = args['eta']
        self.beta_1 = args['beta_1']
        self.beta_2 = args['beta_2']
        self.tau = args['tau']

        self.diff_t = {}
        self.vt = {}
        for key, var in self.net.state_dict().items():
            self.diff_t[key] = torch.zeros(size=var.shape).to(device=self.dev)
            self.vt[key] = torch.full(size=var.shape, fill_value=self.tau * self.tau).to(device=self.dev)

    def run(self):
        accuracy_list = []
        phi_list = []
        if 'parmas_mode' in self.args:
            model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                        self.args['model_name'], self.args['num_of_clients'])
            if not os.path.exists(model_save_dir):
                os.mkdir(model_save_dir)

        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = {}
            cur_phi = 0
            for client in tqdm(clients_in_comm):
                local_parameters = self.myClients.clients_set[client].localUpdate(self.args['epoch'], self.args['batchsize'], self.net,
                                                                            self.loss_func, self.optimers[client], self.global_parameters)
                for var in local_parameters:
                    cur_phi += theta_list[client] * float(local_parameters[var].reshape(-1).dot(local_parameters[var].reshape(-1)))

                for var in local_parameters:
                    if var not in sum_parameters:
                        sum_parameters[var] = theta_list[client] * (local_parameters[var] - self.global_parameters[var])
                    else:
                        sum_parameters[var] +=  theta_list[client] * (local_parameters[var] - self.global_parameters[var])
            
            phi_list.append(cur_phi)
            for var in self.global_parameters:
                self.diff_t[var] = self.beta_1 * self.diff_t[var] + (1 - self.beta_1) * (sum_parameters[var])
                if self.args['alg'] == 'fedadagrad':
                    self.vt[var] = self.vt[var] + self.diff_t[var] * self.diff_t[var]
                elif self.args['alg'] == 'fedyogi':
                    twice_t = self.diff_t[var] * self.diff_t[var]
                    self.vt[var] = self.vt[var] - (1 - self.beta_2) * twice_t * torch.sign(self.vt[var] - twice_t)
                elif self.args['alg'] == 'fedadam':
                    self.vt[var] = self.beta_2 * self.vt[var] + (1 - self.beta_2) * self.diff_t[var] * self.diff_t[var]
                else:
                    raise NotImplementedError('{} not implemented!'.format(self.args['alg']))
                self.global_parameters[var] = self.global_parameters[var] + self.eta * (self.diff_t[var] / (torch.sqrt(self.vt[var]) + self.tau))
            
            if 'parmas_mode' in self.args:
                torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            
            accuracy = self.eval(t=t)
            accuracy_list.append(accuracy)
            print('accuracy: ' + str(accuracy) + "\n")
            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n") 

        self.test_txt.write('phi_list={}'.format(phi_list))
        self.test_txt.write('accuracy_list={}'.format(accuracy_list))
        self.test_txt.close()
        print('phi_list={}'.format(phi_list))
        print('accuracy_list={}'.format(accuracy_list))
        
            