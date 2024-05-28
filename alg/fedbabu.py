import os
import sys
import time
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
import torch
from tqdm import tqdm
from utils import BaseServer

class FedBABUServer(BaseServer):

    def __init__(self, args):
        super(FedBABUServer, self).__init__(args=args)

    def run(self):
        accuracy_list = []
        phi_list = []
        model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                    self.args['model_name'], self.args['num_of_clients'])
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        loss_model = []
        start_time = time.time()
        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            cur_phi = 0
            for client in tqdm(clients_in_comm):
                loss_list = []
                local_parameters, loss_list = self.myClients.clients_set[client].localUpdate_babu(self.args['epoch'], self.args['batchsize'], 
                                                        self.net, self.loss_func, self.optimers[client], self.global_parameters, loss_list)
                
                # 微调
                local_parameters = self.myClients.clients_set[client].fine_tune(self.net, self.args['fine_tuning_epochs'], self.args['batchsize'], self.loss_func, self.optimers[client])


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

            torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            
            accuracy, loss = self.eval(t=t)
            accuracy_list.append(accuracy)
            loss_model.append(loss)

            print('accuracy: ' + str(accuracy) + "\n")
            print('loss: ' + str(loss) + "\n")

            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n") 
            self.test_txt.write('loss: ' + str(loss) + "\n") 

        # # 微调部分
        # fine_tuning_epochs = 5 % self.args['fine_tuning_epochs']
        # print(fine_tuning_epochs)
        # for client in self.myClients.clients_set.values():
        #     client.fine_tune(self.net, fine_tuning_epochs, self.args['batchsize'], self.loss_func, self.optimers[client])


        end_time = time.time()
        training_time = end_time - start_time
        for _ in range(138):
            self.test_txt.write('*') 
        self.test_txt.write("\n")
        self.test_txt.write('**********' + 'FedBABU' + '+' + format(self.args['model_name']) + '+' + format(self.args['dataset']) + '**********'+ "\n")
        self.test_txt.write('accuracy_list={}'.format(accuracy_list)+ "\n")
        self.test_txt.write('loss={}'.format(loss_model)+ "\n")
        self.test_txt.close()
        print('accuracy_list={}'.format(accuracy_list))
        print('loss={}'.format(loss_model))
        print('training_time={}'.format(training_time))
