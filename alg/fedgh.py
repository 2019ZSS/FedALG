import os
import time
import torch
from tqdm import tqdm
from collections import defaultdict
from utils import BaseServer
import torch.nn as nn

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        x = self.fc(x)
        return x

class FedGHServer(BaseServer):

    def __init__(self, args):
        super(FedGHServer, self).__init__(args=args)
        self.global_proto = defaultdict(list)

    def run(self):
        accuracy_list = []
        phi_list = []
        model_save_dir = './checkpoints/parmas/IID_{}/{}_{}_{}_{}'.format(self.args['IID'], self.args['dataset'], self.args['alg'], 
                    self.args['model_name'], self.args['num_of_clients'])
        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        loss_model = []
        start_time = time.time()
        Global_header = None
        for t in range(self.num_comm):
            print("communicate round {}".format(t + 1))
            clients_in_comm, theta_list = self.sample()
            print("client: " + str(clients_in_comm))
            sum_parameters = None
            cur_phi = 0
            Protos_Mean = defaultdict(lambda: defaultdict(list))
            for client in tqdm(clients_in_comm):
                loss_list = []
                local_parameters, loss_list = self.myClients.clients_set[client].localUpdate_gh(self.args['epoch'], self.args['batchsize'], 
                                                        self.net, self.loss_func, self.optimers[client], self.global_parameters, loss_list)

                if sum_parameters is None:
                    sum_parameters = {}
                    for key, var in local_parameters.items():
                        sum_parameters[key] = theta_list[client] * var.clone()
                else:
                    for var in sum_parameters:
                        sum_parameters[var] = sum_parameters[var] +  theta_list[client] * local_parameters[var]
                
                proto_mean = self.collect_local_proto(self.net, self.myClients.clients_set[client])
                Protos_Mean[client] = proto_mean
            
            phi_list.append(cur_phi)
            for var in self.global_parameters:
                self.global_parameters[var] = sum_parameters[var]

            self.aggregate_protos(Protos_Mean)
            Global_header = self.update_global_header(Protos_Mean)

            torch.save(self.net.state_dict(), os.path.join(model_save_dir, '{}.pth'.format(t + 1)))
            
            accuracy, loss = self.eval(t=t)
            accuracy_list.append(accuracy)
            loss_model.append(loss)

            print('accuracy: ' + str(accuracy) + "\n")
            print('loss: ' + str(loss) + "\n")

            self.test_txt.write("communicate round " + str(t + 1) + " ")
            self.test_txt.write('accuracy: ' + str(accuracy) + "\n") 
            self.test_txt.write('loss: ' + str(loss) + "\n") 

        end_time = time.time()
        training_time = end_time - start_time
        for _ in range(138):
            self.test_txt.write('*') 
        self.test_txt.write("\n")
        self.test_txt.write('**********' + 'FedGH' + '+' + format(self.args['model_name']) + '+' + format(self.args['dataset']) + '**********'+ "\n")
        self.test_txt.write('accuracy_list={}'.format(accuracy_list)+ "\n")
        self.test_txt.write('loss={}'.format(loss_model)+ "\n")
        self.test_txt.close()
        print('accuracy_list={}'.format(accuracy_list))
        print('loss={}'.format(loss_model))
        print('training_time={}'.format(training_time))

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