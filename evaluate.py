import torch
from torch.autograd import Variable
from model import TCN
import argparse
import os
from dataset import MitbinDataset
import dill

parser = argparse.ArgumentParser(description='Sequence Modeling - MITBIN-TCN')
parser.add_argument('--savedir',  type=str, default='checkpoint0',
                    help='save dir')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--fold',  type=int, default=0,
                    help='use which fold data (default: 0)')
parser.add_argument('--num_threds',  type=int, default=0,
                    help='number of threads to fetch data (default: 0)')
parser.add_argument('--load_epoch',  type=int, default=1,
                    help='number of threads to fetch data (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

num_threds = args.num_threds
save_dir = args.savedir
batch_size = 1
n_classes = 2
input_channels_T = 3
input_channels_E = 1
input_channels_G = 1

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize

print(args)
train_dataset = MitbinDataset(args, is_for_train=True)
train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threds,
            drop_last=False)
test_dataset = MitbinDataset(args, is_for_train=False)
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threds,
            drop_last=False)

def load_network(network, network_label, epoch_label):
    load_filename = 'net_epoch_%d_id_%s.pth' % (epoch_label, network_label)
    load_path = os.path.join(save_dir, load_filename)
    assert os.path.exists(
        load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

    network.load_state_dict(torch.load(load_path))
    print ('loaded net: %s' % load_path)

model_T = TCN(input_channels_T, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
model_E = TCN(input_channels_E, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
model_G = TCN(input_channels_G, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model_T.cuda()
    model_E.cuda()
    model_G.cuda()

load_epoch = args.load_epoch
load_network(model_T, 'T', load_epoch)
load_network(model_E, 'E', load_epoch)
load_network(model_G, 'G', load_epoch)


weights = []
betas = []
subjects = []
labels = []

def test(data_loader, counts):
    with torch.no_grad():
        for batch_idx, (data, data_reverse, order_data, order_data_reverse, \
                        label, subject, feature) in enumerate(data_loader):
            data = torch.Tensor(data)
            data_reverse = torch.Tensor(data_reverse)
            order_data = torch.Tensor(order_data)
            order_data_reverse = torch.Tensor(order_data_reverse)        
            feature = torch.Tensor(feature)
            target = torch.LongTensor(label)
                
            if args.cuda:
                data, data_reverse, order_data, order_data_reverse, feature, target = data.cuda(), data_reverse.cuda(), order_data.cuda(), order_data_reverse.cuda(), \
                                        feature.cuda(), target.cuda()
                    
            data = data.view(-1, input_channels_E, data.shape[0])
            data_reverse = data_reverse.view(-1, input_channels_E, data_reverse.shape[0])
            order_data = order_data.view(-1, input_channels_T, order_data.shape[1])
            order_data_reverse = order_data_reverse.view(-1, input_channels_T, order_data_reverse.shape[1])
               
            data, data_reverse, order_data, order_data_reverse, target, feature = Variable(data), Variable(data_reverse), \
                                                                                    Variable(order_data), Variable(order_data_reverse), \
                                                                                    Variable(target), Variable(feature)

            output_T = model_T.forward_T(order_data, order_data_reverse)       
            output_E, weight = model_E.forward_E(data, data_reverse, feature)
            output_G, beta, soft_beta = model_G.forward_G(data, data_reverse, output_T.detach(), output_E.detach())

            weight = weight.squeeze().cpu().numpy()

            weights.append(weight)
            #print (weight)

            beta = beta.squeeze().cpu().numpy()
            soft_beta = soft_beta.squeeze().cpu().numpy()
            #print (beta)
            #print (soft_beta)
            betas.append(beta)

            label = label.cpu().numpy()[0]
            labels.append(label)
            
            subject = subject.cpu().numpy()[0]
            subjects.append(subject)
            #print (label)
            #print (subject)
            if batch_idx % 500 == 0:
                print ("Sample: [{}/{} ({:.0f}%)]\t".format(batch_idx, counts, 100. * batch_idx / counts) )


if __name__ == "__main__":
    test(train_loader, len(train_dataset))
    test(test_loader, len(test_dataset))
    res = {'w':weights, 'label':labels, 'subject':subjects, 'beta': betas}
    save_path = os.path.join(save_dir, 'mitbin_wb_%dfold.pkl' % args.fold)
    with open(save_path, 'wb') as fout:
        dill.dump(res, fout)#, protocol=2)
