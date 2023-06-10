#standard  
import os
import sys
import time
import random
import glob
from tqdm import tqdm
import numpy as np
import logging #logging package 
from apex import amp #can help for half precision package
import json 
from argparse import ArgumentParser , RawTextHelpFormatter # command args 
import copy
#torch 
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
#ours 
from data.data import get_loaders ,get_retrain_loaders
from search_arc.hierarchical_controller import  separable_LSTM #,LSTM_index , LSTM_operation
from model_arc.model_maker import model_maker ,complex_model_maker
import utils
from model_arc.operation_storage import conv2d_std , Mish #, Downsample
#get form github  https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import Ranger

from search_arc.GA.nas_ga import nas_ga ,seq_creater






parser = ArgumentParser(description='\nexample :\ncmd>>python train_search_cifar.py    --save cifa10_searcher\n\
\t\t\t\t--seed 0\n\
\t\t\t\t--batch_size 96 \n\
\t\t\t\t--child_out_filters 16 \n \
\t\t\t\t--data ../data/cifar10 \n \
\t\t\t\t--num_class 30\n  \
\t\t\t\t--epochs 150 \n \
\t\t\t\t--blocks "[2,2]"\n \
\t\t\t\t--arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"\n' ,formatter_class = RawTextHelpFormatter)
#parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
#parser.add_argument('--data', type=str, default='../data/org_for_class/nas_eval_hard_sport', help='location of the data corpus')

parser.add_argument('--num_class', type=int, default='10', help='the class number of data')
parser.add_argument('--droppath', type=int, default=0.35, help='droppath probability')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=0.025,help='learning rate maximum')
parser.add_argument('--child_lr_min', type=float, default=0.0005,help='learning rate minimum, WARNING not implement ')
parser.add_argument('--child_lr_T_0', type=int, default=10,help='learning rate minimum, WARNING not implement ')
parser.add_argument('--child_lr_T_mul', type=int, default=2,help='special parameters for cycle learning rate')
parser.add_argument('--child_out_filters', type=int, default=48,help = 'init channls')
parser.add_argument('--child_num_branches', type=int, default=5,help = 'fix it not change ')
parser.add_argument('--child_num_cells', type=int, default=5,help = 'fix it not change ')
parser.add_argument('--child_use_aux_heads', type=bool, default=True,help = 'fix it not change ')

parser.add_argument('--controller_lr', type=float, default=0.000035,help = "stage LSTM's learning rate")
# the parameters below is fixed in search_arc folder hierachical_controller.py
# parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
# parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)

# parser.add_argument('--lstm_size', type=int, default=64)
# parser.add_argument('--lstm_num_layers', type=int, default=1)
# parser.add_argument('--lstm_keep_prob', type=float, default=0)
# parser.add_argument('--temperature', type=float, default=5.0)

parser.add_argument('--entropy_weight', type=float, default=0.0001,help = "make stage LSTM generate different archictecture sequence")
parser.add_argument('--bl_dec', type=float, default=0.95,help = "reinforcement learning's parameters for reduce sparse reward effect")

args = parser.parse_args()

args.save = './result/search/{}-{}'.format(args.save, time.strftime("%Y-%m-%d-%Hh%Mm%Ss"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



baseline = None

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def main():
    start_time = time.time()
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = {}'.format(args.gpu))
    logging.info("args = {}".format(args))
    #supernet 
    model = model_maker(cell_nums=args.child_num_cells,out_filters=args.child_out_filters,normal_block_repeat = [4,4],classes = args.num_class,aux = args.child_use_aux_heads)
    #generator 
    #controller = separable_LSTM(2)
    model.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),conv2d_std(in_channels = args.child_out_filters,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish())
    model.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),
                                         conv2d_std(in_channels = args.child_out_filters,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish())
    logging.info('Total params: {:.6f}M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0) ))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    #generator's optimizer
    # controller_optimizer = Ranger(
    #     controller.parameters(),
    #     args.controller_lr,
    #     #betas=(0.1,0.999),
    #     #eps=1e-3,
    # )
    # controller.cuda()
    model.cuda()

    train_loader, reward_loader, valid_loader = get_loaders(args)
    controller =  nas_ga(reward_loader,200,2)
    #utils.BatchNorm2d_replace(model)
    model.cuda()
    model.apply(utils.initialize_weights)
    parameters = utils.add_weight_decay(model, args.weight_decay)
    criterion = nn.CrossEntropyLoss()#utils.CrossEntropyLabelSmooth(num_classes = 10)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs )#int(args.epochs*0.3))

    lr = args.child_lr_max
    for epoch in tqdm(range(args.epochs)):
        training_start = time.time()
        logging.info('epoch {:0>3d} lr {:.6f}'.format(epoch ,lr ))

        # training
        drop_prob = args.droppath * epoch / args.epochs
        model.drop_path_prob(drop_prob)
        starter =True if epoch == 0 else False
        train_acc = train(train_loader, model, controller, optimizer,criterion,start = starter)
        scheduler.step()
        lr = scheduler.get_lr()[-1]
        logging.info('train_acc {:.3f}'.format(train_acc))
        #train_controller(reward_loader, model, controller, controller_optimizer)
        # validation
        valid_acc = infer(valid_loader, model, controller,criterion)
        logging.info('valid_acc {:.3f}'.format(valid_acc))
        if (epoch+1) % args.report_freq == 0:
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            #utils.save(controller, os.path.join(args.save, 'controller.pt'))
        epoch_inter_time = int(time.time()-training_start)
        #print(f'Trainging 1 Epoch ,Total time consumption {epoch_inter_time} /s ')
        print('Trainging 1 Epoch ,Total time consumption {} /s '.format(epoch_inter_time))
    #logging.info(f'Trainging Complete ,Total time consumption {int(time.time()-start_time)} /s ')
    logging.info('Trainging Complete ,Total time consumption {} /s '.format(int(time.time()-start_time)))


def train(train_loader, model, controller, optimizer,criterion,start=False ):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    #controller.eval()
    model.eval()
    controller.GA_training(10,copy.deepcopy(model))
    model.train()
    aux_nums = len(model.aux_ind)
    for step, (data, target) in enumerate(train_loader):
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        # with torch.no_grad():
        #     dag, _, _ = controller()
        #with architecture different GPU memery usage is different when start with maximum memery usage architecutre sequence
        dag =[seq_creater() for i in range(2)]
        if ((step ==0)&(start==True)):
            dag = [[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]]*2

        logits,auxs = model(dag, data)
        loss1 = criterion(logits, target).cuda()
        loss_aux = sum([criterion(auxs[i], target).cuda()*0.1*(i+1)  for i in range(aux_nums)])/sum((i+1)*0.1 for i in range(aux_nums))
        loss = loss1 + 0.4*loss_aux
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        #loss.backward() without amp use this 
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if (step+1) % args.report_freq == 0:
            logging.info('train {:0>3d} {:.6f} {:.3f}'.format(step,total_loss.avg,total_top1.avg  ))
        with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
                f.write('{:.3f} {} share_weight\n'.format(prec1.item() , [i for i in dag]))#
        del loss
    return total_top1.avg

# def train_controller(reward_loader, model, controller, controller_optimizer):
#     global baseline
#     total_loss = utils.AvgrageMeter()
#     total_reward = utils.AvgrageMeter()
#     total_entropy = utils.AvgrageMeter()
#     controller.train()
#     model.eval()
#     for step in range(150):
#         data, target = reward_loader.next_batch()
#         n = data.size(0)

#         data = data.cuda()
#         target = target.cuda()

#         controller_optimizer.zero_grad()

#         dag, log_prob, entropy = controller()
#         log_prob = sum(log_prob)
#         entropy = sum(entropy)
#         with torch.no_grad():
#             logits,auxs = model(dag, data)
#             reward = utils.accuracy(logits, target)[0]

#         if args.entropy_weight is not None:
#             reward = reward + args.entropy_weight*entropy

#         log_prob = torch.sum(log_prob)
#         #baseline is reinforcement learning's technique 
#         if baseline is None:
#             baseline = reward
#         baseline -= (1 - args.bl_dec) * (baseline - reward)

#         loss = (log_prob * (reward - baseline)).sum()

#         loss.backward()

#         controller_optimizer.step()

#         total_loss.update(loss.item(), n)
#         total_reward.update(reward.item(), n)
#         total_entropy.update(entropy.item(), n)
#         if (step+1) % args.report_freq == 0:
#             logging.info('controller {:0>3d} {:.6f} {:.3f} {:.3f}'.format(step,total_loss.avg ,total_reward.avg ,baseline.item() ))
#             logging.info('{}'.format([i for i in dag]))
#         with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
#             f.write('{:.3f} {} controller\n'.format(reward.item() ,[i for i in dag]))
#         del loss , reward ,entropy ,logits
#             #tensorboard.add_scalar('controller/loss', loss, epoch)
#             #tensorboard.add_scalar('controller/reward', reward, epoch)
#             #tensorboard.add_scalar('controller/entropy', entropy, epoch)
def infer(valid_loader, model, controller,criterion):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()
    #controller.eval()

    with torch.no_grad():
        for step in range(20):
            data, target = valid_loader.next_batch()
            data = data.cuda()
            target = target.cuda()
            dag =controller.dag_creater()#[seq_creater() for i in range(2)]
            #dag, _, _ = controller()

            logits,auxs = model(dag, data)
            loss = criterion(logits, target).cuda()

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            logging.info('valid  {:0>3d} {:.6f} {:.3f}'.format(step,loss.item() ,prec1.item()))
            logging.info('{}'.format([i for i in dag]))
            with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
                f.write('{:.3f} {} infer\n'.format( prec1.item() ,[i for i in dag]))#
            del loss ,logits

    return total_top1.avg

if __name__ == '__main__':
    main()

