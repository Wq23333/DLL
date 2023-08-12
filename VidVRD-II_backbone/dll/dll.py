import numpy as np
import torch
from torch import nn

import os
import pickle
import time
import random

import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info

import dll.utils.common as common
from dll.utils.misc import FocalLoss, binary_focal_loss
from model import IterativeClassifier
from dll.utils.Traindataset import TrainDataset
from torch.optim.lr_scheduler import StepLR

def get_id():
    spacial_list = [[]for i in range(12)]
    action_list = [[]for i in range(28)]
    spacial_exist = torch.ones(size=(1,132))
    action_exist = torch.ones(size=(1,132))
    spacial_unexist = torch.zeros(size=(1,132))
    action_unexist = torch.zeros(size=(1,132))
    both_exist = torch.ones(size=(1,132))
    for i, line in enumerate(open("../data/relation.txt")):
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
    sum = torch.load("../sum.pth")
    action_header_list = [[]for i in range(132)]
    spacial_header_list = [[]for i in range(132)]

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

def get_KDL_loss(p_score, model, gt_target, ep, alpha):
    up_loss = 0
    re_loss = 0
    num = 0
    kl_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)
    spacial_list, action_list, spacial_exist, action_exist, spacial_unexist, action_unexist, both_exist, action_header_list, spacial_header_list= get_id()
    
    loss_t = binary_focal_loss(p_score, gt_target)

    cor_matrix = model.correlation
    p_gt = torch.nonzero(gt_target)
    for pg in p_gt:
        i = pg[0]
        j = pg[1]
        p_res_score = p_score[i]
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

    if ep >5:
        alpha = alpha + 0.01/250
        if alpha > 1.0:
            alpha = 1.0
    gamma = pow(0.99,ep)

    loss_KDL = loss_t + gamma * (loss_cm + alpha * loss_nt)
    return loss_KDL, alpha


def compute_component_dependency(samples, raw_dataset, **param):
    object_num = param['object_num']+1
    predicate_num = param['predicate_num']
    count = np.zeros((object_num, predicate_num, object_num), dtype=np.float32)
    for vid in raw_dataset.get_index(split=param['train_split']):
        rel_insts = raw_dataset.get_relation_insts(vid, no_traj=True)
        for r in rel_insts:
            s, p, o = r['triplet']
            s = raw_dataset.get_object_id(s)
            p = raw_dataset.get_predicate_id(p)
            o = raw_dataset.get_object_id(o)
            count[s, p, o] += 1
    sp2o = np.divide(count+param['model']['smoothing'], np.sum(count, axis=2, keepdims=True)+param['model']['smoothing']*object_num, dtype=np.float32)
    sp2o = sp2o.reshape((-1, object_num))
    po2s = np.divide(count+param['model']['smoothing'], np.sum(count, axis=0, keepdims=True)+param['model']['smoothing']*object_num, dtype=np.float32)
    po2s = np.transpose(po2s, (1, 2, 0)).reshape((-1, object_num))
    so2p = np.divide(count+param['model']['smoothing'], np.sum(count, axis=1, keepdims=True)+param['model']['smoothing']*2, dtype=np.float32)
    so2p = np.transpose(so2p, (0, 2, 1)).reshape((-1, predicate_num))

    return sp2o, po2s, so2p

def _train_worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.rng = np.random.default_rng(worker_info.seed)


def _collate_fn(batch_list):
    items = []
    for item_id in range(len(batch_list[0])):
        item = [x[item_id] for x in batch_list]
        if isinstance(item[0], np.ndarray):
            item = np.concatenate(item)
            item = torch.from_numpy(item)
        items.append(item)
    return items


def _init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def train(raw_dataset, split, use_cuda=False, **param):

    param['object_num'] = raw_dataset.get_object_num()
    param['predicate_num'] = raw_dataset.get_predicate_num()

    train_dataset = TrainDataset(raw_dataset, split, np.random.default_rng(param['rng_seed']), **param)
    sf, of, ppf, pvf, _, _, _ = train_dataset[0]
    param['sub_feature_dim'] = sf.shape[1]
    param['obj_feature_dim'] = of.shape[1]
    param['pred_pos_feature_dim'] = ppf.shape[1]
    param['pred_vis_feature_dim'] = pvf.shape[1]
    print('[info] feature dimension for subject: {}, object: {}, predicate positional: {}, predicate visual: {}'.format(
            param['sub_feature_dim'], param['obj_feature_dim'], param['pred_pos_feature_dim'], param['pred_vis_feature_dim']))

    data_generator = DataLoader(train_dataset, batch_size=param['training_batch_size'], shuffle=True,
            num_workers=param['training_n_workers'], worker_init_fn=_train_worker_init_fn, collate_fn=_collate_fn)

    if param['model']['name'] == 'iterative_classifier':
        if param['model']['use_knowledge']:
            with open(train_dataset.seg_cache_path, 'rb') as fin:
                print('[info] computing triplet component dependency')
                sp2o, po2s, so2p = compute_component_dependency(pickle.load(fin)['samples'], raw_dataset, **param)
            model = IterativeClassifier(sp2o=sp2o, po2s=po2s, so2p=so2p, **param)
        else:
            model = IterativeClassifier(**param)
    else:
        raise ValueError(param['model']['name'])

    model.apply(_init_weights)

    if use_cuda:
        model.cuda()
    object_parameters = []
    visual_parameters = []
    visual_parameters_spacial = []
    visual_parameters_action = []
    spacial_feature = []
    action_feature = []
    s2a_feature = []
    a2s_feature = []
    preferential_parameters = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if name in ['sp2o', 'po2s', 'so2p']:
                preferential_parameters.append(p)
            if name in ['object_predictor.0.weight','object_predictor.0.bias','object_predictor.3.weight']:
                object_parameters.append(p)
            if name in ['action_feature.0.weight','action_feature.0.bias']:
                action_feature.append(p)
            if name in ['spacial_feature.0.weight','spacial_feature.0.bias']:
                spacial_feature.append(p)
            if name in ['spacial_2_action.weight','spacial_2_action.bias']:
                s2a_feature.append(p)
            if name in ['action_2_spacial.weight','action_2_spacial.bias']:
                a2s_feature.append(p)
            if name in ['predicate_predictor_spacial.weight']:
                visual_parameters_spacial.append(p)
            if name in ['predicate_predictor_action.weight']:
                visual_parameters_action.append(p)
            else:
                visual_parameters.append(p)
    object_optim = optim.AdamW(object_parameters, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    scheduler_obj = StepLR(object_optim, step_size=35, gamma=0.1)
    visual_optim = optim.AdamW(visual_parameters, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_spacial = optim.AdamW(visual_parameters_spacial, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_action = optim.AdamW(visual_parameters_action, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_s2a = optim.AdamW(s2a_feature, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_a2s = optim.AdamW(a2s_feature, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_af = optim.AdamW(action_feature, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    visual_optim_sf = optim.AdamW(spacial_feature, lr=param['training_lr'], weight_decay=param['training_weight_decay'])
    
    
    scheduler_obj_v= StepLR(visual_optim, step_size=35, gamma=0.1)
    scheduler_obj_spacial = StepLR(visual_optim_spacial, step_size=35, gamma=0.1)
    scheduler_obj_action = StepLR(visual_optim_action, step_size=35, gamma=0.1)
    scheduler_obj_s2a = StepLR(visual_optim_s2a, step_size=35, gamma=0.1)
    scheduler_obj_a2s = StepLR(visual_optim_a2s, step_size=35, gamma=0.1)
    scheduler_obj_af = StepLR(visual_optim_af, step_size=35, gamma=0.1)
    scheduler_obj_sf = StepLR(visual_optim_sf, step_size=35, gamma=0.1)
    if len(preferential_parameters) > 0:
        preferential_optim = optim.AdamW(preferential_parameters, lr=param['training_lr'], weight_decay=param['model']['weight_decay'])
    focal_loss = FocalLoss(gamma=param['training_focal_gamma'])

    best_loss = float('inf')
    best_ep = 0
    alpha = 0.1
    for ep in range(1, param['training_max_epoch']+1):
        num_iter = len(data_generator)
        s_losses = []
        p_losses = []
        o_losses = []
        total_losses = []
        print('\nEpoch {}'.format(ep))
        model.train()
        t_epoch_start = time.time()
        t_iter_start = time.time()

        for it, data in enumerate(data_generator):
            sf, of, ppf, pvf, s, p_vec, o, middle, start = data

            if use_cuda:
                sf, of, ppf, pvf, s, p_vec, o, middle, start = sf.cuda(), of.cuda(), ppf.cuda(), pvf.cuda(), s.cuda(), p_vec.cuda(), o.cuda(), middle.cuda(), start.cuda()

            t_model_start = time.time()
            visual_optim.zero_grad()
            visual_optim_spacial.zero_grad()
            visual_optim_action.zero_grad()
            visual_optim_a2s.zero_grad()
            visual_optim_s2a.zero_grad()
            visual_optim_af.zero_grad()
            visual_optim_sf.zero_grad()
            object_optim.zero_grad()

            if len(preferential_parameters) > 0:
                preferential_optim.zero_grad()

            s_score, o_score, p_score, loss_PDL = model(sf, of, ppf, pvf, gt_s=s, gt_o=o, gt_p_vec=p_vec)

            s_loss = focal_loss(s_score, s)
            o_loss = focal_loss(o_score, o)

            loss_KDL, alpha = get_KDL_loss(p_score, model, p_vec, ep, alpha)
            total_loss = s_loss+o_loss+loss_PDL+loss_KDL
            total_loss.backward()
            visual_optim.step()
            visual_optim_spacial.step()
            visual_optim_action.step()
            visual_optim_a2s.step()
            visual_optim_s2a.step()
            visual_optim_af.step()
            visual_optim_sf.step()

            if len(preferential_parameters) > 0:
                preferential_optim.step()
            
            s_losses.append(s_loss.data.item())
            o_losses.append(o_loss.data.item())
            p_losses.append(loss_KDL.data.item())
            total_losses.append(total_loss.data.item())
            t_model_end = time.time()

            if it % param['training_display_freq'] == 0:
                str_loss = ('iter: [{}/{}], s_loss: {:.4f}, o_loss: {:.4f}, loss_KDL: {:.4f}, total: {:.4f}, data time: {:.4f}s, total time: {:.4f}s'.format(
                        it, num_iter,
                        s_loss.data.item(), o_loss.data.item(), loss_KDL.data.item(), total_loss.data.item(),
                        t_model_start-t_iter_start, t_model_end-t_iter_start))
                print(str_loss)
            t_iter_start = time.time()
            
        scheduler_obj.step()
        scheduler_obj_v.step()
        scheduler_obj_spacial.step()
        scheduler_obj_action.step()
        scheduler_obj_s2a.step()
        scheduler_obj_a2s.step()
        scheduler_obj_af.step()
        scheduler_obj_sf.step()
        print('-'*120)
        print('Training summary:'.format(ep))
        print('[info] s_loss: {:.4f}, o_loss: {:.4f}, p_loss: {:.4f}, total: {:.4f}, time: {:.4f}s'.format(
                np.mean(s_losses), np.mean(o_losses), np.mean(p_losses),
                np.mean(total_losses), time.time()-t_epoch_start))
        
        if ep % param['training_save_freq'] == 0:
            dump_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}weights_ep{}.t7'.format(param['model'].get('dump_file_prefix', ''), ep))
            print('[info] saving weights to {}'.format(dump_path))
            torch.save(model.state_dict(), dump_path)
        
        if np.mean(total_losses) < best_loss:
            best_loss = np.mean(total_losses)
            best_ep = ep
            dump_path = os.path.join(common.get_model_path(param['exp_id'], param['dataset']), 'weights',
                    '{}weights_best.t7'.format(param['model'].get('dump_file_prefix', '')))
            print('[info] saving weights to {}'.format(dump_path))
            torch.save(model.state_dict(), dump_path)

    print('[info] best training loss {:.4f} after {} training epoch'.format(best_loss, best_ep))
  
    return param








