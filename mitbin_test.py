import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
from dataset import MitbinDataset
from model import TCN
import numpy as np
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score,precision_score,recall_score
import os

parser = argparse.ArgumentParser(description='Sequence Modeling - MITBIN-TCN')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='report interval (default: 10')
parser.add_argument('--lr_T', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--lr_E', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--lr_G', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--lr_R', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--fold',  type=int, default=0,
                    help='use which fold data (default: 0)')
parser.add_argument('--num_threds',  type=int, default=0,
                    help='number of threads to fetch data (default: 0)')
parser.add_argument('--alpha',  type=float, default=1.0,
                    help='weight to control loss (default: 1.0)')
parser.add_argument('--beta',  type=float, default=1.0,
                    help='weight to control loss (default: 1.0)')
parser.add_argument('--gamma',  type=float, default=1.0,
                    help='weight to control loss (default: 1.0)')
parser.add_argument('--savedir',  type=str, default='checkpoint0',
                    help='weight to control loss (default: 1.0)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
mkdir(args.savedir)

batch_size = args.batch_size
batch_size = 1
n_classes = 2
input_channels_T = 3
input_channels_E = 1
input_channels_G = 1

#seq_length = 500
epochs = args.epochs
steps = 0
num_threds = args.num_threds

alpha = args.alpha
beta = args.beta
gamma = args.gamma

print(args)
train_dataset = MitbinDataset(args, is_for_train=True)
train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_threds,
            drop_last=False)
test_dataset = MitbinDataset(args, is_for_train=False)
test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threds,
            drop_last=False)
		
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize

model_T = TCN(input_channels_T, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
model_E = TCN(input_channels_E, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)
model_G = TCN(input_channels_G, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model_T.cuda()
    model_E.cuda()
    model_G.cuda()
    
optimizer = getattr(optim, args.optim)([{'params': model_T.parameters(), 'lr': args.lr_T},
                                        {'params': model_E.parameters(), 'lr': args.lr_E},
                                        {'params': model_G.parameters(), 'lr': args.lr_G}
                                        ])#,momentum=0.9)

def save_network(network, network_label, epoch_label):
    save_filename = 'net_epoch_%d_id_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(args.savedir, save_filename)
    torch.save(network.state_dict(), save_path)
    print ('saved net: %s' % save_path)
    
def train(ep):
    global steps
    total_loss = 0
    model_T_loss = 0
    model_E_loss = 0
    model_G_loss = 0
    
    model_T.train()
    model_E.train()
    model_G.train()
    
    correct = 0
  
    for batch_idx, (data, data_reverse, order_data, order_data_reverse, \
                    label, subject, feature) in enumerate(train_loader):
          
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
        output_E, _ = model_E.forward_E(data, data_reverse, feature)
        output_G, _, _ = model_G.forward_G(data, data_reverse, output_T.detach(), output_E.detach())

        #print (output_T.shape)
        #print (output_G.shape)
        #print (output_E.shape)

        optimizer.zero_grad()
        loss_T = F.nll_loss(output_T, target)
        loss_E = F.nll_loss(output_E, target)
        loss_G = F.nll_loss(output_G, target)

        loss = alpha * loss_T + beta * loss_E + gamma * loss_G
        
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        #model_T_loss += float(loss_T)
        #model_E_loss += float(loss_E)
        #model_G_loss += float(loss_G)

        pred = output_G.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}\t ACC: {:.4f}'.format(
                ep, batch_idx * batch_size, len(train_dataset),
                100. * batch_idx / len(train_dataset), total_loss/args.log_interval, float(correct) / (batch_idx*batch_size) ))
            total_loss = 0

def cal_performance(subject_dic, count_dic, real_dic):
    # vote results
    avg_acc = 0
    ind = 0
    real = []
    logits = []
    for key in subject_dic:
        subject_acc = subject_dic[key] / count_dic[key]
        print ('Subject %d: ACC is %f' %(key, subject_acc))
        ind += 1
        avg_acc += subject_acc
        real.append(real_dic[key])
        if subject_acc > 0.5:
            logit = real_dic[key]
        else:
            logit = 1 - real_dic[key]
        logits.append(logit)
    print ('Avg Subjects: ACC is %f' % (float(avg_acc) / ind))
    y_true = np.array(real)
    y_pred = np.array(logits)
    print ('Accuracy of Classifier:%f' % accuracy_score(y_true, y_pred))
    print ('ROC-AUC of Classifier:%f' % roc_auc_score(y_true, y_pred))
    precision, recall, _thresholds = precision_recall_curve(y_true, y_pred)     
    print ('PR-AUC of Classifier:%f' % auc(recall, precision))
    print ('Macro-F1 of Classifier:%f' % f1_score(y_true, y_pred, average='micro'))
    print ("precision:", precision_score(y_true, y_pred))
    print ("recall:", recall_score(y_true, y_pred))
    
def test():
    model_T.eval()
    model_E.eval()
    model_G.eval()
    
    test_loss = 0
    
    correct_T = 0
    correct_E = 0
    correct_G = 0
    subject_dic_T = defaultdict(int)
    subject_dic_E = defaultdict(int)
    subject_dic_G = defaultdict(int)
    
    count_dic = defaultdict(int)

    real_dic = {}

    with torch.no_grad():
        for data, data_reverse, order_data, order_data_reverse, \
                    label, subject, feature in test_loader:
            data = torch.Tensor(data)
            data_reverse = torch.Tensor(data_reverse)
            order_data = torch.Tensor(order_data)
            order_data_reverse = torch.Tensor(order_data_reverse)        
            feature = torch.Tensor(feature)
            target = torch.LongTensor(label)
            
            if args.cuda:
                data, data_reverse, order_data, order_data_reverse, feature, target = data.cuda(), data_reverse.cuda(), order_data.cuda(), order_data_reverse.cuda(), \
                                      feature.cuda(), target.cuda()
                
            #seq_length = data.shape[1]
            #print (seq_length)
            data = data.view(-1, input_channels_E, data.shape[0])
            data_reverse = data_reverse.view(-1, input_channels_E, data_reverse.shape[0])
            order_data = order_data.view(-1, input_channels_T, order_data.shape[1])
            order_data_reverse = order_data_reverse.view(-1, input_channels_T, order_data_reverse.shape[1])
           
            data, data_reverse, order_data, order_data_reverse, target, feature = Variable(data), Variable(data_reverse), \
                                                                                  Variable(order_data), Variable(order_data_reverse), \
                                                                                  Variable(target), Variable(feature)

            output_T = model_T.forward_T(order_data, order_data_reverse)       
            output_E, _ = model_E.forward_E(data, data_reverse, feature)
            output_G, _ , _= model_G.forward_G(data, data_reverse, output_T.detach(), output_E.detach())
           
            test_loss += F.nll_loss(output_G, target, size_average=False).item()

            pred_T = output_T.data.max(1, keepdim=True)[1]
            eq_T = pred_T.eq(target.data.view_as(pred_T)).cpu().sum()
            correct_T += eq_T

            pred_E = output_E.data.max(1, keepdim=True)[1]
            eq_E = pred_E.eq(target.data.view_as(pred_E)).cpu().sum()
            correct_E += eq_E
            
            pred_G = output_G.data.max(1, keepdim=True)[1]
            eq_G = pred_G.eq(target.data.view_as(pred_G)).cpu().sum()
            correct_G += eq_G

            subject_dic_T[subject.cpu().numpy()[0]] += eq_T.numpy()
            subject_dic_E[subject.cpu().numpy()[0]] += eq_E.numpy()
            subject_dic_G[subject.cpu().numpy()[0]] += eq_G.numpy()
            
            
            count_dic[subject.cpu().numpy()[0]] += 1.0
            real_dic[subject.cpu().numpy()[0]] = target.cpu().numpy()[0]

        test_loss /= len(test_dataset)
        print('\nTest set: Average loss: {:.4f}, ACC_T: {:.4f} ACC_E: {:.4f} ACC_G: {:.4f}\n'.format(
            test_loss, float(correct_T) / len(test_dataset), float(correct_E) / len(test_dataset), float(correct_G) / len(test_dataset)))

        print ("-------------------------model_T--------------------------------")
        cal_performance(subject_dic_T, count_dic, real_dic)
        print ("-------------------------model_E--------------------------------")
        cal_performance(subject_dic_E, count_dic, real_dic)
        print ("-------------------------model_G--------------------------------")
        cal_performance(subject_dic_G, count_dic, real_dic)
      
        return test_loss


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        test()
        save_network(model_T, "T", epoch)
        save_network(model_E, "E", epoch)
        save_network(model_G, "G", epoch)
