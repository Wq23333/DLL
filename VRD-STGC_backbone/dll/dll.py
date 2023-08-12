import argparse
import os
import time
import torch
import random
import shutil
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from utils import g, AverageMeter, load_source

from datasets import get_training_set, get_testing_set
from model import RelationPredictor
from IPython import embed

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=20, type=int)
parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--num-workers", default=8, type=int)
parser.add_argument("--lr", default=0.1, type=int)
parser.add_argument("--momentum", default=0.9, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=int)
parser.add_argument("--num-classes", default=134, type=int)
parser.add_argument("--dump_dir", default="./logs/relation_cls", type=str)
parser.add_argument("--print-freq", default=50, type=int)
parser.add_argument("--save-freq", default=1, type=int)

args = parser.parse_args()
if not os.path.isdir(args.dump_dir):
    os.makedirs(args.dump_dir)
def main():
    model = RelationPredictor()
    model.cuda()
    train_set = get_training_set()
    
    train_loader = data.DataLoader(dataset=train_set,
            num_workers=args.num_workers, batch_size=args.batch_size,
            shuffle=True, pin_memory=True)
    
    visual_parameters = []
    correlation_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name in ['correlation']:
                correlation_parameters.append(p)
            else:
                visual_parameters.append(p)

    criterion = nn.BCEWithLogitsLoss().cuda()
    kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True).cuda()
    optimizer = optim.SGD(visual_parameters, args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
    
    correlation_optim = optim.SGD(correlation_parameters, args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)

    cudnn.benchmark = True
    alpha = 0.1
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(correlation_optim, epoch)

        train(train_loader, model, criterion, optimizer, kl_loss, correlation_optim, epoch, alpha)
        name = "train"
        if epoch % args.save_freq == 0:
            save_dir = os.path.join(args.dump_dir, name)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            filename = os.path.join(args.dump_dir, name, "relation_%d.pth" %
                epoch)
            is_best = False
            save_checkpoint({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "correlation":correlation_optim.state_dict(),
            }, is_best, filename)
        

def get_id():
    spacial_list = [[]for i in range(13)]
    action_list = [[]for i in range(29)]
    spacial_exist = torch.ones(size=(1,133))
    action_exist = torch.ones(size=(1,133))
    spacial_unexist = torch.zeros(size=(1,133))
    action_unexist = torch.zeros(size=(1,133))
    both_exist = torch.ones(size=(1,133))
    for i, line in enumerate(open("../data/relations.txt")):
        label= line.strip().split()[0]  
        spacial_id = int(line.strip().split()[1])
        action_id = int(line.strip().split()[2])
        if spacial_id != -1:
            spacial_list[spacial_id].append(i)
        elif spacial_id == -1:
            spacial_exist[0][i]=0
            both_exist[0][i] = 0 
            spacial_unexist[0][i]=1
        if action_id != -1:
            action_list[action_id].append(i)
        elif action_id == -1:
            action_exist[0][i]=0
            both_exist[0][i] = 0 
            action_unexist[0][i]=1 
    sum = torch.load("../data/sum.pth")
    action_header_list = [[]for i in range(133)]
    spacial_header_list = [[]for i in range(133)]

    for i in range(len(spacial_list)):
        for j in range(len(spacial_list[i])):
            index_current = spacial_list[i][j]
            for k in range(len(spacial_list[i])):
                if k == j:
                    continue
                index_new = spacial_list[i][k]
                if (sum[0][index_new]>sum[0][index_current]):
                    spacial_header_list[index_current].append(index_new)
            if len(spacial_header_list[index_current])==0:
                spacial_header_list[index_current].append(index_current)
    for i in range(len(action_list)):
        for j in range(len(action_list[i])):
            index_current = action_list[i][j]
            for k in range(len(action_list[i])):
                if k == j:
                    continue
                index_new = action_list[i][k]
                if (sum[0][index_new]>sum[0][index_current]):
                    action_header_list[index_current].append(index_new)
            if len(action_header_list[index_current])==0:
                action_header_list[index_current].append(index_current)
    return spacial_list, action_list, spacial_exist, action_exist, spacial_unexist, action_unexist, both_exist, action_header_list, spacial_header_list


def train(train_loader, model, criterion, optimizer, kl_loss, correlation_optim, epoch, alpha):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mutlosses = AverageMeter()
    totales = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    end = time.time()
    log = ""
    name = "train"
    spacial_list, action_list, spacial_exist, action_exist, spacial_unexist, action_unexist, both_exist, action_header_list, spacial_header_list= get_id()
    
    for i, pack in enumerate(train_loader):
        sub_features, obj_features, i3d_sub_features, i3d_obj_features, motion, target, weight = pack
        if len(list(sub_features.shape)) == 2:
            continue
        data_time.update(time.time() - end)
        target = target.cuda()
        sub_var = torch.autograd.Variable(sub_features).cuda()
        obj_var = torch.autograd.Variable(obj_features).cuda()
        i3d_sub_var = torch.autograd.Variable(i3d_sub_features).cuda()
        i3d_obj_var = torch.autograd.Variable(i3d_obj_features).cuda()
        motion = torch.autograd.Variable(motion).cuda()
        target_var = torch.autograd.Variable(target)
        weight = weight.cuda()

        feed_list = [sub_var, obj_var, i3d_sub_var, i3d_obj_var, motion]
        cor_matrix, output, loss_PDL = model(*feed_list)
        loss_t = criterion(output, target_var)

        up_loss = 0
        re_loss = 0
        num = 0
        p_gt = torch.nonzero(target_var[0])
        for pg in p_gt:
            i = pg[0]
            j = pg[1]
            p_res_score = output[0][i]
            p_res_score2 = torch.cat((p_res_score[0:j],p_res_score[j+1:]))
            p_res = F.log_softmax(p_res_score2, dim=-1)

            
            if (spacial_exist[0][j] == True) and (action_exist[0][j] == True):
                header_index_spacial = random.choice(spacial_header_list[j])
                correlation_spacial = cor_matrix[header_index_spacial]
                header_index_action = random.choice(action_header_list[j])
                correlation_action = cor_matrix[header_index_action]

                q_res_score = (correlation_spacial + correlation_action)/2
                q_res_score2 = torch.cat((q_res_score[0:j],q_res_score[j+1:]))
            elif (spacial_exist[0][j] == False) and (action_exist[0][j] == True):
                header_index_action = random.choice(action_header_list[j])
                correlation_action = cor_matrix[header_index_action]

                q_res_score = correlation_action
                q_res_score2 = torch.cat((q_res_score[0:j],q_res_score[j+1:]))
            elif (spacial_exist[0][j] == True) and (action_exist[0][j] == False):
                header_index_spacial = random.choice(spacial_header_list[j])
                correlation_spacial = cor_matrix[header_index_spacial]

                q_res_score = correlation_spacial
                q_res_score2 = torch.cat((q_res_score[0:j],q_res_score[j+1:]))
            q_res = F.log_softmax(q_res_score2, dim=-1)

            q_res_score_tail = cor_matrix[j]
            q_res_score2_tail = torch.cat((q_res_score_tail[0:j],q_res_score_tail[j+1:]))
            q_res_tail = F.log_softmax(q_res_score2_tail, dim=-1)
            
            res_loss = kl_loss(p_res,q_res.clone().detach())
            update_loss = kl_loss(q_res_tail,p_res.clone().detach())

            up_loss = up_loss + update_loss
            re_loss = re_loss + res_loss

            num = num+1
            loss_cm = up_loss/num
            loss_nt = re_loss/num


        
        if epoch > 3:
            alpha = alpha + 0.0001/709
            if alpha > 1.0:
                alpha = 1.0
        gamma = pow(0.99,epoch)
        loss_KDL = loss_t + gamma * (loss_cm + alpha * loss_nt)
        loss = loss_KDL + loss_PDL

        losses.update(loss_t.data, sub_var.size(0)*sub_var.size(1))
        totales.update(loss.data, sub_var.size(0)*sub_var.size(1))
        mutlosses.update(loss_KDL.data, sub_var.size(0)*sub_var.size(1))
        optimizer.zero_grad()
        correlation_optim.zero_grad()

        loss.backward()
        optimizer.step()
        correlation_optim.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            log = ("Epoch: [{0}][{1}/{2}]\t"
                   "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                   "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                   "PLoss {ploss.val:.3f} ({ploss.avg:.3f})\t"
                   "MutLoss {mutloss.val:.3f} ({mutloss.avg:.3f})\t"
                   "TotalLoss {totalloss.val:.4f} ({totalloss.avg:.4f})\t".format(
                   epoch, i+1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, ploss=losses, mutloss=mutlosses, totalloss=totales))
            with open(os.path.join(args.dump_dir, "log_"+name+".txt"), "a+") as f:
                f.writelines(log + "\n")
                f.close()
            print(log)
    with open(os.path.join(args.dump_dir, "state_"+name+".txt"), "a+") as f:
        f.writelines(log + "\n")
        f.close()

def adjust_learning_rate(optimizer, epoch):
    interval = int(args.epochs * 0.4)
    lr = args.lr * (0.1 ** (epoch // interval))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_checkpoint(state, is_best, filename):
    try:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.dump_dir,
                "relation_best.pth"))
    except Exception as e:
        print("save error")

if __name__ == "__main__":
    main()
