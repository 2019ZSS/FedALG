import argparse
import json
from alg import (
    FedAVGServer,
    FedADPServer,
    FedADLRServer,
    FedADPLRServer,
)

def get_args():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedALG")
    parser.add_argument('-config_path', '--config_path', type=str, default='./configs/fedavg.json', help='Config path of FL algorithm')
    parser.add_argument('-alg', '--alg', type=str, default='fedavg', help='FL algorithms(e.g fedavg, fedadam, fedyogi)')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
    parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
    parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
    parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
    parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
    parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, use value from origin paper as default")
    parser.add_argument('-dataset',"--dataset",type=str,default="mnist",help="需要训练的数据集")
    parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
    parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
    parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
    parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
    parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
    parser.add_argument('-eta', "--eta", type=float, default=0.01, help="Control Factors of FedADP Algorithm")
    parser.add_argument('-beta_1', "--beta_1", type=float, default=0.9, help="weight deacy params of FedADP Algorithm")
    parser.add_argument('-beta_2', "--beta_2", type=float, default=0.9, help="weight deacy params of FedADP Algorithm")
    parser.add_argument('-beta', "--beta", type=float, default=0.99, help="Control Factors of Fedadlr,Fedadplr")
    parser.add_argument('-min_lr', "--min_lr", type=float, default=1e-6, help="weight deacy params of Fedadlr,Fedadplr")
    parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.99, help="weight deacy params of Fedadlr,Fedadplr")
    parser.add_argument('-shuffle', '--shuffle', type=bool, default=True, help="Randomly traverse the local dataset order")
    parser.add_argument('-pre_t', '--pre_t', type=int, default=3, help="Number of pre-training")
    parser.add_argument('-eps', '--eps', type=float, default=1.0, help="convergence error")
    
    return parser.parse_args()


if __name__=="__main__":
    args = get_args()
    args = args.__dict__
    with open(args['config_path'], encoding='utf-8') as f:
        configs = json.load(f)
    args = configs
    if args['alg'] == 'fedavg':
        alg_sever = FedAVGServer(args=args)
    elif args['alg'] in ('fedadam', 'fedyogi', 'fedadagrad'):
        alg_sever = FedADPServer(args=args)
    elif args['alg'] == 'fedadlr':
        alg_sever = FedADLRServer(args=args)
    elif args['alg'] == 'fedadplr':
        alg_sever = FedADPLRServer(args=args)
    else:
        raise NotImplementedError('{} not implement'.format(args['alg']))
    alg_sever.run()
    

    
