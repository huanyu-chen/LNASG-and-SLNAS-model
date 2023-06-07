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
from data.data import get_loaders , get_retrain_loaders , get_cifar_retrain_loaders
from search_arc.hierarchical_controller import  separable_LSTM #,LSTM_index , LSTM_operation
from model_arc.model_maker import model_maker
from model_arc.rebuilder import model_rebuild
import utils
#get form github  https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

from ranger import Ranger


#command args processing 
parser = ArgumentParser(description='\nexample :\ncmd>>python retrainer_org.py    --save AID_50_searched2\n\
\t\t\t\t--seed 2\n\
\t\t\t\t--batch_size 32 \n\
\t\t\t\t--child_out_filters 40 \n \
\t\t\t\t--data ../data/exp_data/AID_50 \n \
\t\t\t\t--num_class 30\n  \
\t\t\t\t--epochs 600\n \
\t\t\t\t--blocks "[2,2]"\n \
\t\t\t\t--arch "[[0, 1, 0, 4, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0], [0, 3, 1, 0, 0, 4, 1, 1, 1, 1, 0, 3, 0, 2, 1, 2, 1, 1, 0, 3]]"\n' ,formatter_class = RawTextHelpFormatter)

parser.add_argument('--data', type=str, default='../data/cifar10', help='location of the data corpus')
parser.add_argument('--num_class', type=int, default='10', help='the class number of data')
parser.add_argument('--drop_prob', type=int, default=0.35, help='droppath probability')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--child_lr_max', type=float, default=2.5e-2)
parser.add_argument('--child_lr_min', type=float, default=1e-3)
parser.add_argument('--child_lr_T_0', type=int, default=10)
parser.add_argument('--child_lr_T_mul', type=int, default=2)
parser.add_argument('--child_out_filters', type=int, default=32)
parser.add_argument('--child_num_branches', type=int, default=5)
parser.add_argument('--child_num_cells', type=int, default=5)
parser.add_argument('--child_use_aux_heads', type=bool, default=True)

parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
# parser.add_argument('--alpha', default=0.1, type=float,
#                     help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--blocks', type=str, default='{}'.format([4,4]),help='the block stack parameters')

parser.add_argument('--arch', type=str, default='{}'.format([[0, 4, 0, 1, 0, 4, 1, 1, 0, 1, 3, 4, 1, 0, 0, 3, 1, 1, 0, 4],
                                                              [0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 4, 1, 0, 1, 1, 1, 4]]),
                                                                                                help='architecture sequance')
args = parser.parse_args()

args.save = './result/retrain/{}-{}'.format(args.save, time.strftime("%Y-%m-%d-%Hh%Mm%Ss"))
args.blocks = json.loads(args.blocks)
args.arch = json.loads(args.arch)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

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
    logging.info(f'gpu device = {args.gpu}')
    logging.info(f"args = {args}")

    model = model_maker(cell_nums=args.child_num_cells,out_filters=args.child_out_filters,normal_block_repeat = args.blocks,classes = args.num_class,aux = args.child_use_aux_heads)
    dag = args.arch
    weight = model.parameters_selection(dag)
    del model
    model = model_rebuild(weight,dag)
    utils.BatchNorm2d_replace(model)
    del weight
    logging.info(f'Total params: {(sum(p.numel() for p in model.parameters()) / 1000000.0) :.6f}M')


    model.cuda()
    train_loader,  valid_loader = get_cifar_retrain_loaders(args)
    #model.apply(utils.initialize_weights)
    #model.load_state_dict(torch.load(r'C:\Users\loyd7\Desktop\autoML\clear_and_complete\result\search\EXP-2020-03-05-23h10m25s\weights.pt'))


    parameters = utils.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.SGD(
        parameters,
        args.child_lr_max,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = utils.CrossEntropyLabelSmooth(num_classes = 10)#nn.CrossEntropyLoss()#
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs )#int(args.epochs*0.3))
    lr = args.child_lr_min
    #lr_finder(model, criterion, optimizer,train_loader)

    for epoch in tqdm(range(args.epochs)):
        training_start = time.time()
       #lr = scheduler.update(epoch)
        logging.info(f'epoch {epoch :0>3d} lr {lr :.6f}')

        if int(epoch < args.epochs*0.7):
            drop_prob = args.drop_prob * ((epoch*0.7) / (args.epochs*0.7))
            model.drop_path_prob(drop_prob)
        print(f'now drop rate{drop_prob}')
        starter =True if epoch == 0 else False
        train_acc = train(train_loader, model, optimizer,criterion,start = starter)
        logging.info(f'train_acc {train_acc :.3f}')

        # validation
        scheduler.step()
        lr = scheduler.get_lr()[-1]
        valid_acc = infer(valid_loader, model,criterion)
        logging.info(f'valid_acc {valid_acc :.3f}')
        if (epoch+1) % args.report_freq == 0:
            utils.save(model, os.path.join(args.save, f'weights_{epoch}.pt'))
        epoch_inter_time = int(time.time()-training_start)
        print(f'Trainging 1 Epoch ,Total time consumption {epoch_inter_time} /s ')

    logging.info(f'Trainging Complete ,\n\
        Total time consumption {int(time.time()-start_time)} /s ,         \
        Epoch Average {epoch_inter_time} /s')


def train(train_loader, model, optimizer,criterion,start=False ):
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.train()
    aux_nums = len(model.aux_ind)

    for step, (data, target) in enumerate(train_loader):
        n = data.size(0)

        data = data.cuda()
        target = target.cuda()
        #
        # data, targets_a, targets_b, lam = mixup_data(data, target,
        #                                                args.alpha, use_cuda=True)

        # #
        optimizer.zero_grad()
        logits,auxs = model(data)
        loss1 = criterion(logits, target).cuda()
        #loss1 =  mixup_criterion(criterion, logits, targets_a, targets_b, lam)

        #loss_aux = sum([criterion(auxs[i], target).cuda()*0.1*(i+1)  for i in range(aux_nums)])
        loss_aux = sum([criterion(auxs[i], target).cuda()  for i in range(aux_nums)])
        #loss_aux = sum([mixup_criterion(criterion,auxs[i], targets_a, targets_b, lam).cuda()*0.1*(i+1)  for i in range(aux_nums)])

        loss = loss1 + 0.4*loss_aux
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        #loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)

        optimizer.step()

        prec1 = utils.accuracy(logits, target)[0]
        #_, predicted = torch.max(logits.data, 1)
        #prec1 =  (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
        total_loss.update(loss.item(), n)
        total_top1.update(prec1.item(), n)

        if (step+1) % args.report_freq == 0:
            logging.info(f'train {step :0>3d} {total_loss.avg :.6f} {total_top1.avg :.3f}')
        del loss ,loss1 ,loss_aux
    return total_top1.avg

def infer(valid_loader, model,criterion):
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
            [logging.info(f'aux_{i}  {step :0>3d} {total_loss_aux[i].avg :.6f} {total_top1_aux[i].avg :.3f}') for i in range(aux_nums)]

            logging.info(f'valid  {step :0>3d} {total_loss.avg :.6f} {total_top1.avg :.3f}')

    del loss ,logits ,loss_aux

    return total_top1.avg

def warmup(train_loader, model,criterion,optimizer,target_lr,warmup_epoch=5,start=False):
    utils.optim_lr_changer(optimizer, args.child_lr_max/12/10e5)
    #warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    #model, optimizer_temp = amp.initialize(model, optimizer, opt_level="O0")
    print(args.child_lr_max/12/10e5)
    total_loss = utils.AvgrageMeter()
    total_top1 = utils.AvgrageMeter()
    model.train()
    for i in tqdm(range(warmup_epoch)):
        for step, (data, target) in enumerate(train_loader):
            n = data.size(0)
            data = data.cuda()
            target = target.cuda()
            optimizer.zero_grad()

            logits,auxs = model(data)
            loss1 = criterion(logits, target).cuda()
            loss2 = criterion(auxs, target).cuda()
            loss = loss1 + 0.4*loss2
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                 scaled_loss.backward()
            #loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
            optimizer.step()

            prec1 = utils.accuracy(logits, target)[0]
            total_loss.update(loss.item(), n)
            total_top1.update(prec1.item(), n)
            if (step+1) % args.report_freq == 0:
                print(f'warmup {step :0>3d} {total_loss.avg :.6f} {total_top1.avg :.3f}')
            del loss
        utils.optim_lr_changer(optimizer, args.child_lr_max/12/10e5*10**(i+1))
        print(args.child_lr_max/12/10e5*10**(i+1))
    utils.optim_lr_changer(optimizer, args.child_lr_max/12)
    return total_top1.avg
    return total_top1.avg
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def lr_finder(model, criterion, optimizer,train_loader,names = 'default'):
    aux_nums = len(model.aux_ind)
    import math 
    import matplotlib.pyplot as plt
    
    init_value = 1e-8
    final_value=10.
    beta = 0.98
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs,labels in tqdm(train_loader):
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()

        outputs,auxs = model(inputs)
        loss1 = criterion(outputs, labels).cuda()
        loss2  = sum([criterion(auxs[i], labels).cuda()*0.1*(i+1)  for i in range(aux_nums)])
        loss = loss1 + 0.4*loss2

        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.data
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
             scaled_loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    plt.plot(log_lrs[10:-5],losses[10:-5])
    plt.savefig('{}.jpg'.format(names))
    plt.clf()
    return log_lrs[10:-5],losses[10:-5]
if __name__ == '__main__':
    main()

