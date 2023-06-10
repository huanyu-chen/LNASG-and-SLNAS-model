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
#torch 
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
#ours
from data.data import get_loaders ,get_retrain_loaders
from search_arc.hierarchical_controller import  separable_LSTM #,LSTM_index , LSTM_operation
from tqdm import tqdm
from model_arc.model_maker import model_maker ,complex_model_maker
import utils
from model_arc.operation_storage import conv2d_std , Mish #, Downsample
#get form github  https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
from ranger import Ranger


parser = ArgumentParser(description='\nexample :\ncmd>>python retrainer_org.py    --save AID_50_searched2\n\
\t\t\t\t--seed 2\n\
\t\t\t\t--batch_size 32 \n\
\t\t\t\t--child_out_filters 40 \n \
\t\t\t\t--data ../data/exp_data/AID_50 \n \
\t\t\t\t--num_class 30\n  \
\t\t\t\t--epochs 600\n \
\t\t\t\t--blocks "[2,2]"\n \
\t\t\t\t--arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"\n' ,formatter_class = RawTextHelpFormatter)
#parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--data', type=str, default='../data/org_for_class/nas_eval_hard_sport', help='location of the data corpus')

parser.add_argument('--num_class', type=int, default='8', help='the class number of data')
parser.add_argument('--droppath', type=int, default=0.35, help='droppath probability')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=150, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=0.025)
parser.add_argument('--child_lr_min', type=float, default=0.0001)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_out_filters', type=int, default=16)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_use_aux_heads', type=bool, default=True)

parser.add_argument('--controller_lr', type=float, default=0.00035)
# parser.add_argument('--controller_tanh_constant', type=float, default=1.10)
# parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5)

# parser.add_argument('--lstm_size', type=int, default=64)
# parser.add_argument('--lstm_num_layers', type=int, default=1)
# parser.add_argument('--lstm_keep_prob', type=float, default=0)
# parser.add_argument('--temperature', type=float, default=5.0)

parser.add_argument('--entropy_weight', type=float, default=0.0001)
parser.add_argument('--bl_dec', type=float, default=0.95)
parser.add_argument('--blocks', type=str, default='{}'.format([1,1]),help='the block stack parameters')

# parser.add_argument('--arch', type=str, default='{}'.format([[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],
#                                                               [0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]),
#                                                                                                 help='architecture sequance')
args = parser.parse_args()

args.save = './result/search/{}-{}'.format(args.save, time.strftime("%Y-%m-%d-%Hh%Mm%Ss"))
args.blocks = json.loads(args.blocks)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



baseline = None

#fix torch's backend to get reproducibility resault 
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
    #logging.info(f'gpu device = {args.gpu}')
    logging.info('gpu device = {}'.format(args.gpu))
    #logging.info(f"args = {args}")
    logging.info("args = {}".format(args))

    model = model_maker(cell_nums=args.child_num_cells,out_filters=args.child_out_filters,normal_block_repeat = args.blocks,classes = args.num_class,aux = args.child_use_aux_heads)
    controller = separable_LSTM(2)
    baseline = None
    model.start_conv1 =  nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),conv2d_std(in_channels = args.child_out_filters,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish())
    model.start_conv2 = nn.Sequential(conv2d_std(in_channels = 3,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish(),
                                         conv2d_std(in_channels = args.child_out_filters,out_channels = args.child_out_filters,kernel_size = 3,stride=2),
                                          nn.BatchNorm2d( args.child_out_filters , track_running_stats=False),Mish())
    #logging.info(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0) :.6f}M')
    logging.info('Total params: {:.6f}M'.format((sum(p.numel() for p in model.parameters()) / 1000000.0) ))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    controller_optimizer = Ranger(
        controller.parameters(),
        args.controller_lr,
        #betas=(0.1,0.999),
        #eps=1e-3,
    )
    controller.cuda()
    model.cuda()

    #train_loader, reward_loader, valid_loader = get_loaders(args)#get_retrain_loaders(args)
    train_loader, reward_loader, valid_loader = get_retrain_loaders(args)#get_retrain_loaders(args)
    utils.BatchNorm2d_replace(model)
    model.cuda()
    model.apply(utils.initialize_weights)
    parameters = utils.add_weight_decay(model, args.weight_decay)
    criterion = nn.CrossEntropyLoss()#utils.CrossEntropyLabelSmooth(num_classes = 10)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    #warmup(train_loader, model, controller, criterion, optimizer, args.child_lr_max)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs ,eta_min = args.child_lr_min)#int(args.epochs*0.3))

    lr = args.child_lr_min
    #scheduler = utils.LRScheduler(optimizer, args)
    for epoch in tqdm(range(args.epochs)):
        training_start = time.time()
        #lr = scheduler.update(epoch)
        #logging.info(f'epoch {epoch :0>3d} lr {lr :.6f}')
        logging.info('epoch {:0>3d} lr {:.6f}'.format(epoch ,lr ))

        # training
        drop_prob = args.droppath * epoch / args.epochs
        model.drop_path_prob(drop_prob)
        starter =True if epoch == 0 else False
        train_acc = train(train_loader, model, controller, optimizer,criterion,start = starter)
        scheduler.step()
        lr = scheduler.get_lr()[-1]
        #logging.info(f'train_acc {train_acc :.3f}')
        logging.info('train_acc {:.3f}'.format(train_acc))
        train_controller(reward_loader, model, controller, controller_optimizer)
        # validation
        valid_acc = infer(valid_loader, model, controller,criterion)
        logging.info('valid_acc {:.3f}'.format(valid_acc))
        if (epoch+1) % args.report_freq == 0:
            utils.save(model, os.path.join(args.save, 'weights.pt'))
            utils.save(controller, os.path.join(args.save, 'controller.pt'))
        epoch_inter_time = int(time.time()-training_start)
        #print(f'Trainging 1 Epoch ,Total time consumption {epoch_inter_time} /s ')
        print('Trainging 1 Epoch ,Total time consumption {} /s '.format(epoch_inter_time))
    #logging.info(f'Trainging Complete ,Total time consumption {int(time.time()-start_time)} /s ')
    logging.info('Trainging Complete ,Total time consumption {} /s '.format(int(time.time()-start_time)))


def train(train_loader, model, controller, optimizer,criterion,start=False ):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    controller.eval()
    model.train()

    for step, (data, target) in enumerate(train_loader):
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()
        with torch.no_grad():
            dag, _, _ = controller()
        if ((step ==0)&(start==True)):
            dag = [[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]]*2

        logits,auxs = model(dag, data)
        loss1 = criterion(logits, target).cuda()
        loss2 = criterion(auxs[0], target).cuda()
        loss = loss1 + 0.4*loss2
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        #loss.backward()
        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if (step+1) % args.report_freq == 0:
            #logging.info(f'train {step :0>3d} {total_loss.avg :.6f} {total_top1.avg :.3f}')
            logging.info('train {:0>3d} {:.6f} {:.3f}'.format(step,total_loss.avg,total_top1.avg  ))
        with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
                #f.write(f'{ prec1.item() :.3f} {[i for i in dag]} share_weight\n')#
                f.write('{:.3f} {} share_weight\n'.format(prec1.item() , [i for i in dag]))#
        del loss
    return total_top1.avg

def train_controller(reward_loader, model, controller, controller_optimizer):
    global baseline
    total_loss = utils.AvgrageMeter()
    total_reward = utils.AvgrageMeter()
    total_entropy = utils.AvgrageMeter()
    controller.train()
    model.eval()
    for step in range(150):
        data, target = reward_loader.next_batch()
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()

        controller_optimizer.zero_grad()

        dag, log_prob, entropy = controller()
        log_prob = sum(log_prob)
        entropy = sum(entropy)
        with torch.no_grad():
            logits,auxs = model(dag, data)
            reward = utils.accuracy(logits, target)[0]

        if args.entropy_weight is not None:
            reward = reward + args.entropy_weight*entropy

        log_prob = torch.sum(log_prob)
        if baseline is None:
            baseline = reward
        baseline -= (1 - args.bl_dec) * (baseline - reward)

        loss = (log_prob * (reward - baseline)).sum()

        loss.backward()

        controller_optimizer.step()

        total_loss.update(loss.item(), n)
        total_reward.update(reward.item(), n)
        total_entropy.update(entropy.item(), n)
        if (step+1) % args.report_freq == 0:
            #logging.info('controller %03d %e %f %f', step, loss.item(), reward.item(), baseline.item())
            #logging.info(f'controller {step :0>3d} {total_loss.avg :.6f} {total_reward.avg :.3f} {baseline.item() :.3f}')
            logging.info('controller {:0>3d} {:.6f} {:.3f} {:.3f}'.format(step,total_loss.avg ,total_reward.avg ,baseline.item() ))
            # logging.info(f'{[i for i in dag]}')
            logging.info('{}'.format([i for i in dag]))
        with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
            #f.write(f'{reward.item() :.3f} {[i for i in dag]} controller\n')
            f.write('{:.3f} {} controller\n'.format(reward.item() ,[i for i in dag]))
        del loss , reward ,entropy ,logits
            #tensorboard.add_scalar('controller/loss', loss, epoch)
            #tensorboard.add_scalar('controller/reward', reward, epoch)
            #tensorboard.add_scalar('controller/entropy', entropy, epoch)
def infer(valid_loader, model, controller,criterion):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.eval()
    controller.eval()

    with torch.no_grad():
        for step in range(20):
            data, target = valid_loader.next_batch()
            data = data.cuda()
            target = target.cuda()

            dag, _, _ = controller()

            logits,auxs = model(dag, data)
            loss = criterion(logits, target).cuda()

            prec1 = utils.accuracy(logits, target)[0]
            n = data.size(0)
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)

            #if step % args.report_freq == 0:
            # logging.info(f'valid  {step :0>3d} {loss.item() :.6f} {prec1.item() :.3f}')
            logging.info('valid  {:0>3d} {:.6f} {:.3f}'.format(step,loss.item() ,prec1.item()))
            #logging.info(f'{[i for i in dag]}')
            logging.info('{}'.format([i for i in dag]))
            with open(os.path.join(args.save,'dag_all.txt'),'a') as f:
                # f.write(f'{ prec1.item() :.3f} {[i for i in dag]} infer\n')#
                f.write('{:.3f} {} infer\n'.format( prec1.item() ,[i for i in dag]))#
            del loss ,logits

    return total_top1.avg

# def warmup(train_loader, model,controller,criterion,optimizer,target_lr,warmup_epoch=5,start=False):
#     utils.optim_lr_changer(optimizer, args.child_lr_max/10e5)
#     #warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
#     #model, optimizer_temp = amp.initialize(model, optimizer, opt_level="O0")
#     print(args.child_lr_max/10e5)
#     total_loss = utils.AvgrageMeter()
#     total_top1 = utils.AvgrageMeter()
#     controller.eval()
#     model.train()
#     for i in tqdm(range(warmup_epoch)):
#         for step, (data, target) in enumerate(train_loader):
#             n = data.size(0)
#             data = data.cuda()
#             target = target.cuda()
#             optimizer.zero_grad()
#             with torch.no_grad():
#                 dag, _, _ = controller()
#             if ((step ==0)&(start==True)):
#                 dag = [[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]]

#             logits,auxs = model(dag, data)
#             loss1 = criterion(logits, target).cuda()
#             loss2 = criterion(auxs, target).cuda()
#             loss = loss1 + 0.4*loss2
#             with amp.scale_loss(loss, optimizer) as scaled_loss:
#                  scaled_loss.backward()
#             #loss.backward()
#             optimizer.step()

#             prec1 = utils.accuracy(logits, target)[0]
#             total_loss.update(loss.item(), n)
#             total_top1.update(prec1.item(), n)
#             if (step+1) % args.report_freq == 0:
#                 print(f'warmup {step :0>3d} {total_loss.avg :.6f} {total_top1.avg :.3f}')
#             del loss
#             utils.optim_lr_changer(optimizer, target_lr)
#         utils.optim_lr_changer(optimizer, args.child_lr_max/10e5*10**(i+1))
#         #print(args.child_lr_max/10e5*10**(i+1))
#     utils.optim_lr_changer(optimizer, target_lr)
#     return total_top1.avg
if __name__ == '__main__':
    main()

