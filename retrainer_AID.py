#standard  
import os
import sys
import time
import random
import glob
import numpy as np
import logging #logging package 
from apex import amp #can help for half precision package
import json 
from argparse import ArgumentParser , RawTextHelpFormatter # command args 
#torch 
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
#ours 
from data.data import get_loaders ,AID_retrain_loader
from search_arc.hierarchical_controller import  separable_LSTM #,LSTM_index , LSTM_operation
from tqdm import tqdm
from model_arc.model_maker import model_maker
from model_arc.rebuilder import model_rebuild
from model_arc.operation_storage import conv2d_std , Mish , Downsample
import utils
#get form github  https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import Ranger




#command args processing 
parser = ArgumentParser(description='\nexample :\ncmd>>python retrainer_AID.py    --save AID_50_searched2\n\
\t\t\t\t--seed 2\n\
\t\t\t\t--batch_size 32 \n\
\t\t\t\t--child_out_filters 40 \n \
\t\t\t\t--data ../data/exp_data/AID_50 \n \
\t\t\t\t--num_class 30\n  \
\t\t\t\t--epochs 600\n \
\t\t\t\t--blocks "[2,2]"\n \
\t\t\t\t--arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"\n' ,formatter_class = RawTextHelpFormatter)

parser.add_argument('--data', type=str, default='../data/org_for_class/nas_eval_hard_sport', help='location of the data corpus')
parser.add_argument('--num_class', type=int, default=30, help='the class number of data')
parser.add_argument('--drop_prob', type=int, default=0.5, help='droppath probability')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=1, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=2.5e-2,help = 'the maximum learning rate with SuperNet')
#parser.add_argument('--child_lr_min', type=float, default=1e-3)
#parser.add_argument('--child_lr_T_0', type=int, default=10)
#parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_out_filters', type=int, default=32,help = 'init channls')
parser.add_argument('--child_num_branches', type=int, default=5,help = 'fix it not change ')
parser.add_argument('--child_num_cells', type=int, default=5,help = 'fix it not change ')

parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping parameters')
# parser.add_argument('--alpha', default=0.4, type=float,
#                     help='mixup interpolation coefficient (default: 1)')#AID 0.1

parser.add_argument('--blocks', type=str, default='{}'.format([2,2]),help='the block stack parameters')

parser.add_argument('--arch', type=str, default='{}'.format([[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],
                                                              [0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]),
                                                                                                help='architecture sequance')
#command args parsing 
args = parser.parse_args()
args.save = './result/retrain/{}-{}'.format(args.save, time.strftime("%Y-%m-%d-%Hh%Mm%Ss"))
args.blocks = json.loads(args.blocks)
args.arch = json.loads(args.arch)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

#log processing 
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#not fix torch's backend to get reproducibility resault 
#can speed up training , but every time get different reault , only small different 
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    start_time = time.time()
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    #Fix all random seed make result reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = {}'.format(args.gpu))
    logging.info("args = {}".format(args))

    #build SuperNet model 
    model = model_maker(cell_nums=args.child_num_cells,out_filters=args.child_out_filters,normal_block_repeat = args.blocks,classes = args.num_class,aux = True)
    dag = args.arch
    # select all parameters
    weight = model.parameters_selection(dag)
    del model
    #rebuild models from weight and architecture sequance
    model = model_rebuild(weight,dag)
    del weight
    model.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),conv2d_std(in_channels = args.child_out_filters,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish())
    model.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),Downsample(channels=args.child_out_filters, filt_size=3, stride=2))

    utils.BatchNorm2d_replace(model)
    logging.info('Total params: {:.6f}M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0)))

    model.cuda()
    model.apply(utils.initialize_weights)
    parameters = utils.add_weight_decay(model, args.weight_decay)

    #model.load_state_dict(torch.load(r'C:\Users\loyd7\Desktop\autoML\ENAS\Final_version\result\retrain\leaf_transfer-2020-07-03-06h04m45s\weights_199.pt'))

    optimizer = torch.optim.SGD(parameters,args.child_lr_max,momentum=args.momentum,weight_decay=args.weight_decay,)
    criterion = utils.CrossEntropyLabelSmooth(num_classes = args.num_class)#label smooth 
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs )

    #get training and valid data 
    train_loader,  valid_loader = AID_retrain_loader(args)

    lr = args.child_lr_max

    #training cycle 
    for epoch in tqdm(range(args.epochs)):
        training_start = time.time()
        logging.info('epoch {:0>3d} lr {:.6f}'.format(epoch ,lr))
        #special optimally the droppath probabillty is deceasing with epoch  
        if int(epoch < args.epochs*0.7):
            drop_prob = args.drop_prob * (epoch*0.7) / (args.epochs*0.7)
            model.drop_path_prob(drop_prob)

        print('now drop rate{}'.format(drop_prob))

        #make 
        starter =True if epoch == 0 else False
        train_acc = train(train_loader, model, optimizer,criterion,start = starter)
        logging.info('train_acc {:.3f}'.format(train_acc))

        # validation
        scheduler.step()
        lr = scheduler.get_lr()[-1]
        if (epoch+1) % (args.epochs//3 ) == 0:
            valid_acc = infer(valid_loader, model,criterion)
            logging.info('valid_acc {:.3f}'.format(valid_acc))
            utils.save(model, os.path.join(args.save, f'weights_{epoch}.pt'))
        epoch_inter_time = int(time.time()-training_start)
        print('Trainging 1 Epoch ,Total time consumption {} /s '.format(epoch_inter_time))
    logging.info('Trainging Complete ,\n\
            Total time consumption {} /s ,         \
            Epoch Average {} /s'.format(int(time.time()-start_time),epoch_inter_time))


def train(train_loader, model, optimizer,criterion,start=False ):
    """
targert
    training loop and record loss, accuracy 
parameters
    train_loader : training dataloader 
    model        : training model 
    optimizer    : optimizer 
    criterion    : loss function 
    start        : in search stage needed , make all training stable 
return 
    average top 1 accuracy 
    """
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.train()
    aux_nums = len(model.aux_ind)
    for step, (data, target) in enumerate(train_loader):
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()
        # data, targets_a, targets_b, lam = mixup_data(data, target,args.alpha, True)
        optimizer.zero_grad()
        logits,auxs = model(data)
        loss1 = criterion(logits, target).cuda()
        #loss1 =  mixup_criterion(criterion, logits, targets_a, targets_b, lam)
        #stage loss not in paper , every block will return loss and have different weight 
        loss_aux = sum([criterion(auxs[i], target).cuda()*0.1*(i+1)  for i in range(aux_nums)])/sum((i+1)*0.1 for i in range(aux_nums))
        #loss_aux = sum([mixup_criterion(criterion , auxs[i], targets_a, targets_b, lam).cuda()*0.1*(i+1)  for i in range(aux_nums)])/sum((i+1)*0.1 for i in range(aux_nums))

        loss = loss1 + 0.4*loss_aux
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if (step+1) % args.report_freq == 0:
            logging.info('train {:0>3d} {:.6f} {:.3f}'.format(step,total_loss.avg ,total_top1.avg ))
        del loss ,loss1 ,loss_aux
    return total_top1.avg

def infer(valid_loader, model,criterion):
    """
targert
    valid loop and record loss ,accuracy 
parameters
    train_loader : training dataloader 
    model        : training model 
    criterion    : loss function 
    return 
    average top 1 accuracy 
    """
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    aux_nums = len(model.aux_ind)

    total_loss_aux = [utils.AvgrageMeter() for i in range(aux_nums)]
    total_top1_aux = [utils.AvgrageMeter() for i in range(aux_nums)]

    model.eval()

    for step, (data, target) in enumerate(valid_loader):

        data = data.cuda()
        target = target.cuda()
        with torch.no_grad():
            logits,auxs = model(data)
        loss = F.cross_entropy(logits, target).cuda()
        loss_aux = [F.cross_entropy(i, target).cuda() for i in auxs]

        prec1 = utils.accuracy(logits, target)[0]
        prec1_aux = [utils.accuracy(i, target)[0] for i in auxs]

        n = data.size(0)

        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        [total_loss_aux[ind].update(i.item(), n) for ind,i in enumerate(loss_aux)]
        [total_top1_aux[ind].update(i.item(), n) for ind,i in enumerate(prec1_aux)]


        if (step+1) % args.report_freq == 0:
            [logging.info('aux_{}  {:0>3d} {:.6f} {:.3f}'.format(i,step,total_loss_aux[i].avg ,total_top1_aux[i].avg )) for i in range(aux_nums)]
            logging.info('\nvalid  {:0>3d} {:.6f} {:.3f}'.format(step,total_loss.avg ,total_top1.avg ))

    del loss ,logits ,loss_aux

    return total_top1.avg

if __name__ == '__main__':
    main()

