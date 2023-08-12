from itertools import product
import pdb
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import Function
from typing import Any, Optional, Tuple
import torch.nn.functional as F

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        # print(grad_output.neg() * ctx.coeff)
        return grad_output.neg() * ctx.coeff, None

class GRL_Layer(nn.Module):
    def __init__(self):
        super(GRL_Layer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class IterativeClassifier(nn.Module):
    def __init__(self, sp2o=None, po2s=None, so2p=None, **param):
        super().__init__()

        self.object_num = param['object_num']+1
        self.predicate_num = param['predicate_num']
        self.use_gt_label = param['training_use_gt_label']
        
        if param['model']['use_knowledge']:
            self.use_knowledge = True
            if sp2o is None:
                self.register_parameter('sp2o', nn.Parameter(torch.zeros(self.object_num*self.predicate_num, self.object_num), requires_grad=False))
                self.register_parameter('po2s', nn.Parameter(torch.zeros(self.predicate_num*self.object_num, self.object_num), requires_grad=False))
                self.register_parameter('so2p', nn.Parameter(torch.zeros(self.object_num*self.object_num, self.predicate_num), requires_grad=False))
            else:
                self.register_parameter('sp2o', nn.Parameter(torch.from_numpy(sp2o), requires_grad=False))
                self.register_parameter('po2s', nn.Parameter(torch.from_numpy(po2s), requires_grad=False))
                self.register_parameter('so2p', nn.Parameter(torch.from_numpy(so2p), requires_grad=False))
            self.register_parameter('sp2o_factor', nn.Parameter(torch.zeros(1), requires_grad=True))
            self.register_parameter('po2s_factor', nn.Parameter(torch.zeros(1), requires_grad=True))
            self.register_parameter('so2p_factor', nn.Parameter(torch.zeros(1), requires_grad=True))
        else:
            self.use_knowledge = False
            self.register_parameter('sp2o', nn.Parameter(torch.zeros(self.object_num*self.predicate_num, self.object_num), requires_grad=True))
            self.register_parameter('po2s', nn.Parameter(torch.zeros(self.predicate_num*self.object_num, self.object_num), requires_grad=True))
            self.register_parameter('so2p', nn.Parameter(torch.zeros(self.object_num*self.object_num, self.predicate_num), requires_grad=True))

        self.dropout_ratio = param['model']['dropout']
        self.object_predictor = nn.Sequential(
            nn.Linear(param['obj_feature_dim'], param['model']['hidden_dim']),
            nn.Dropout(param['model']['dropout']),
            nn.ReLU(),
            nn.Linear(param['model']['hidden_dim'], param['object_num']+1, bias=False))

        self.predicate_vis_feat_emb = nn.Sequential(
            nn.Linear(param['pred_vis_feature_dim'], param['model']['hidden_dim']),
            nn.Dropout(param['model']['dropout']),
            nn.ReLU())
        self.predicate_feat_emb  = nn.Sequential(
            nn.Linear(param['model']['hidden_dim']+param['pred_pos_feature_dim'], param['model']['hidden_dim']),
            nn.ReLU())
            
        dim = 384
        self.action_feature = nn.Sequential(
            nn.Linear(param['model']['hidden_dim'], dim),
            nn.ReLU())
        self.spacial_feature = nn.Sequential(
            nn.Linear(param['model']['hidden_dim'], dim),
            nn.ReLU())
        self.action_2_spacial = nn.Linear(dim, dim)
        self.spacial_2_action = nn.Linear(dim, dim)
        self.grl = GRL_Layer()

        self.predicate_predictor_spacial = nn.Linear(dim, 12, bias=False)
        self.predicate_predictor_action = nn.Linear(dim, 28, bias=False)

        self.spacial_list, self.action_list, self.spacial_exist, self.action_exist, self.spacial_unexist, self.action_unexist, self.both_exist= self.get_id()
        self.spacial_list, self.action_list, self.spacial_exist, self.action_exist, self.spacial_unexist, self.action_unexist, self.both_exist = self.spacial_list, self.action_list, self.spacial_exist.cuda(), self.action_exist.cuda(), self.spacial_unexist.cuda(), self.action_unexist.cuda(), self.both_exist.cuda()

        self.correlation = nn.Parameter(torch.zeros(param['predicate_num'],param['predicate_num']),requires_grad=True)
        self.criterion = nn.KLDivLoss(reduction="batchmean",log_target=True)

    def GRL(self, fake_input1, fake_input2, lambda_):
        return self.grl(self.global_T, lambda_)

    def get_id(self):
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
        return spacial_list, action_list, spacial_exist, action_exist, spacial_unexist, action_unexist, both_exist

    def _idx_to_vec(self, idx, dim, device):
        vec = torch.zeros(idx.size(0), dim, device=device).scatter_(1, idx.view(-1, 1), 1)
        return vec
    
    def _get_conditional_bias(self, x_vec, y_vec, embeddings, dropout_ratio=0):
        bias = embeddings(torch.cat((x_vec, y_vec), axis=1))
        if dropout_ratio > 0:
            mask = bias.data.new(bias.data.size(0), 1).uniform_()
            mask = Variable((mask>dropout_ratio).float())
            bias = bias*mask
        return bias

    def infer_zero_shot_preference(self, strategy='global_median'):
        print('[info] inferring zero-shot preference based on {}'.format(strategy))
        if strategy == 'global_mean':
            sp_zs_idx = torch.all(self.sp2o==0, dim=1)
            sp2o_mean = torch.mean(self.sp2o[~sp_zs_idx])
            self.sp2o[sp_zs_idx, :] = sp2o_mean

            po_zs_idx = torch.all(self.po2s==0, dim=1)
            po2s_mean = torch.mean(self.po2s[~po_zs_idx])
            self.po2s[po_zs_idx, :] = po2s_mean

            so_zs_idx = torch.all(self.so2p==0, dim=1)
            so2p_mean = torch.mean(self.so2p[~so_zs_idx])
            self.so2p[so_zs_idx, :] = so2p_mean
        
        elif strategy == 'global_median':
            sp_zs_idx = torch.all(self.sp2o==0, dim=1)
            sp2o_median = torch.median(self.sp2o[~sp_zs_idx])
            self.sp2o[sp_zs_idx, :] = sp2o_median

            po_zs_idx = torch.all(self.po2s==0, dim=1)
            po2s_median = torch.median(self.po2s[~po_zs_idx])
            self.po2s[po_zs_idx, :] = po2s_median

            so_zs_idx = torch.all(self.so2p==0, dim=1)
            so2p_median = torch.median(self.so2p[~so_zs_idx])
            self.so2p[so_zs_idx, :] = so2p_median

        elif strategy == 'class_mean':
            sp_zs_idx = torch.all(self.sp2o==0, dim=1)
            sp2o_mean = torch.mean(self.sp2o[~sp_zs_idx], dim=0)
            self.sp2o[sp_zs_idx, :] = sp2o_mean

            po_zs_idx = torch.all(self.po2s==0, dim=1)
            po2s_mean = torch.mean(self.po2s[~po_zs_idx], dim=0)
            self.po2s[po_zs_idx, :] = po2s_mean

            so_zs_idx = torch.all(self.so2p==0, dim=1)
            so2p_mean = torch.mean(self.so2p[~so_zs_idx], dim=0)
            self.so2p[so_zs_idx, :] = so2p_mean
        
        elif strategy == 'class_median':
            sp_zs_idx = torch.all(self.sp2o==0, dim=1)
            sp2o_median = torch.median(self.sp2o[~sp_zs_idx], dim=0)
            self.sp2o[sp_zs_idx, :] = sp2o_median.values

            po_zs_idx = torch.all(self.po2s==0, dim=1)
            po2s_median = torch.median(self.po2s[~po_zs_idx], dim=0)
            self.po2s[po_zs_idx, :] = po2s_median.values

            so_zs_idx = torch.all(self.so2p==0, dim=1)
            so2p_median = torch.median(self.so2p[~so_zs_idx], dim=0)
            self.so2p[so_zs_idx, :] = so2p_median.values

    def forward(self, inp_sf, inp_of, inp_ppf, inp_pvf, gt_s=None, gt_o=None, gt_p_vec=None):
        device = inp_sf.device
        B, _ = inp_sf.size()

        s_score = self.object_predictor(inp_sf)
        o_score = self.object_predictor(inp_of)

        pvf_emb = self.predicate_vis_feat_emb(inp_pvf)
        pf_emb = self.predicate_feat_emb(torch.cat((pvf_emb, inp_ppf), axis=1))
        action_feat =  self.action_feature(pf_emb)
        spacial_feat =  self.spacial_feature(pf_emb)

        ## the output of actional_pattern and spatial_pattern
        p_score_action = self.predicate_predictor_action(action_feat)
        p_score_spacial = self.predicate_predictor_spacial(spacial_feat)

        tmp_action = self.grl.forward(action_feat)
        wrong_spacial_feat = self.action_2_spacial(tmp_action)
        wrong_spacial_output = self.predicate_predictor_spacial(wrong_spacial_feat)
        tmp_spacial = self.grl.forward(spacial_feat)
        wrong_action_feat = self.spacial_2_action(tmp_spacial)
        wrong_action_output = self.predicate_predictor_action(wrong_action_feat)

        #  wrong_spacial_output
        sig_wrong_spacial_output = torch.cat([torch.sigmoid(wrong_spacial_output).unsqueeze(-1),(1-torch.sigmoid(wrong_spacial_output)).unsqueeze(-1)],dim=-1)
        sig_wrong_spacial_output = sig_wrong_spacial_output.reshape(-1, sig_wrong_spacial_output.size()[-1])

        # wrong_action_output
        sig_wrong_action_output = torch.cat([torch.sigmoid(wrong_action_output).unsqueeze(-1),(1-torch.sigmoid(wrong_action_output)).unsqueeze(-1)],dim=-1)
        sig_wrong_action_output = sig_wrong_action_output.reshape(-1, sig_wrong_action_output.size()[-1])

        # p_score_spacial
        p_s_spacial = p_score_spacial.clone().detach()
        sig_p_s_spacial = torch.cat([torch.sigmoid(p_s_spacial).unsqueeze(-1),(1-torch.sigmoid(p_s_spacial)).unsqueeze(-1)],dim=-1)
        sig_p_s_spacial = sig_p_s_spacial.reshape(-1, sig_p_s_spacial.size()[-1])

        # p_score_action
        p_s_action = p_score_action.clone().detach()
        sig_p_s_action = torch.cat([torch.sigmoid(p_s_action).unsqueeze(-1),(1-torch.sigmoid(p_s_action)).unsqueeze(-1)],dim=-1)
        sig_p_s_action = sig_p_s_action.reshape(-1, sig_p_s_action.size()[-1])

        loss_action = self.criterion(torch.log(sig_wrong_spacial_output+1e-12), torch.log(sig_p_s_spacial+1e-12))*(0.13)
        loss_spacial = self.criterion(torch.log(sig_wrong_action_output+1e-12), torch.log(sig_p_s_action+1e-12))*(0.13)

        ## mapping function
        p_score_s = torch.zeros(size = (inp_ppf.shape[0], 132),dtype=torch.float32).cuda()
        p_score_a = torch.zeros(size = (inp_ppf.shape[0], 132),dtype=torch.float32).cuda()
        for i in range(len(self.spacial_list)):
            p_score_s[:,self.spacial_list[i]] = p_score_spacial[:,i].unsqueeze(1)
        for i in range(len(self.action_list)):
            p_score_a[:,self.action_list[i]] = p_score_action[:,i].unsqueeze(1)
        radio = 0.5
        p_score = p_score_a * radio * self.both_exist + p_score_s * (1-radio) * self.both_exist + p_score_s * self.action_unexist + p_score_a * self.spacial_unexist

        ##  mutual calibration
        for st in range(1):
            sp_score = torch.zeros(size = (p_score.shape[0],12),dtype=torch.float32).cuda()
            ac_score = torch.zeros(size = (p_score.shape[0],28),dtype=torch.float32).cuda()
            for i in range(len(self.spacial_list)):
                sp_score[:,i] = p_score[:,self.spacial_list[i]].mean(dim=1)
            for i in range(len(self.action_list)):
                ac_score[:,i] = p_score[:,self.action_list[i]].mean(dim=1)
            theta = 0.001
            p_score_spacial = p_score_spacial * (1.-theta) + sp_score * theta
            p_score_action = p_score_action * (1.-theta) + ac_score * theta

            p_score_s = torch.zeros(size = (p_score.shape),dtype=torch.float32).cuda()
            p_score_a = torch.zeros(size = (p_score.shape),dtype=torch.float32).cuda()
            for i in range(len(self.spacial_list)):
                p_score_s[:,self.spacial_list[i]] = p_score_spacial[:,i].unsqueeze(1)
            for i in range(len(self.action_list)):
                p_score_a[:,self.action_list[i]] = p_score_action[:,i].unsqueeze(1)
            p_score = p_score_a * 0.5 * self.both_exist + p_score_s * 0.5 * self.both_exist + p_score_s * self.action_unexist + p_score_a * self.spacial_unexist

        ## loss of PDL
        loss_PDL = loss_action + loss_spacial
        if self.use_gt_label:
            s_vec = self._idx_to_vec(gt_s, self.object_num, device)
            o_vec = self._idx_to_vec(gt_o, self.object_num, device)
            p_vec = gt_p_vec
        else:
            s_vec = nn.functional.softmax(s_score, dim=1)
            o_vec = nn.functional.softmax(o_score, dim=1)
            p_vec = torch.sigmoid(p_score)

        po_vec = p_vec.unsqueeze(2).expand(B, self.predicate_num, self.object_num)*o_vec.unsqueeze(1).expand(B, self.predicate_num, self.object_num)
        po_bias = torch.mm(po_vec.view(B, -1), self.po2s)
        if self.use_knowledge:
            po_bias *= torch.exp(self.po2s_factor)

        sp_vec = s_vec.unsqueeze(2).expand(B, self.object_num, self.predicate_num)*p_vec.unsqueeze(1).expand(B, self.object_num, self.predicate_num)
        sp_bias = torch.mm(sp_vec.view(B, -1), self.sp2o)
        if self.use_knowledge:
            sp_bias *= torch.exp(self.sp2o_factor)

        so_vec = s_vec.unsqueeze(2).expand(B, self.object_num, self.object_num)*o_vec.unsqueeze(1).expand(B, self.object_num, self.object_num)
        so_bias = torch.mm(so_vec.view(B, -1), self.so2p)
        if self.use_knowledge:
            so_bias *= torch.exp(self.so2p_factor)

        s_score_biased = s_score+po_bias
        o_score_biased = o_score+sp_bias
        p_score_biased = p_score+so_bias

        return s_score_biased, o_score_biased, p_score_biased, loss_PDL

    def inference(self, inp_sf, inp_of, inp_ppf, inp_pvf, s_init_prob, o_init_prob, p_init_prob,
            steps=2, predicate_confidence_threshold=0.5, inference_problistic=False):
        device = inp_sf.device
        B, _ = inp_sf.size()

        s_score = self.object_predictor(inp_sf)
        o_score = self.object_predictor(inp_of)
        pvf_emb = self.predicate_vis_feat_emb(inp_pvf)
        pf_emb = self.predicate_feat_emb(torch.cat((pvf_emb, inp_ppf), axis=1))
        p_score = self.predicate_predictor(pf_emb)

        s_prob, o_prob, p_prob = s_init_prob, o_init_prob, p_init_prob
        for _ in range(steps):
            po_vec = p_prob.unsqueeze(2).expand(B, self.predicate_num, self.object_num)*o_prob.unsqueeze(1).expand(B, self.predicate_num, self.object_num)
            po_bias = torch.mm(po_vec.view(B, -1), self.po2s)
            if self.use_knowledge:
                po_bias *= torch.exp(self.po2s_factor)
            sp_vec = s_prob.unsqueeze(2).expand(B, self.object_num, self.predicate_num)*p_prob.unsqueeze(1).expand(B, self.object_num, self.predicate_num)
            sp_bias = torch.mm(sp_vec.view(B, -1), self.sp2o)
            if self.use_knowledge:
                sp_bias *= torch.exp(self.sp2o_factor)
            s_score_biased = nn.functional.softmax(s_score+po_bias, dim=1)
            o_score_biased = nn.functional.softmax(o_score+sp_bias, dim=1)
            if inference_problistic:
                s_prob = s_score_biased
                o_prob = o_score_biased
            else:
                s_max_idx = torch.argmax(s_score_biased, 1)
                o_max_idx = torch.argmax(o_score_biased, 1)
                s_prob = self._idx_to_vec(s_max_idx, self.object_num, device)
                o_prob = self._idx_to_vec(o_max_idx, self.object_num, device)

            so_vec = s_prob.unsqueeze(2).expand(B, self.object_num, self.object_num)*o_prob.unsqueeze(1).expand(B, self.object_num, self.object_num)
            so_bias = torch.mm(so_vec.view(B, -1), self.so2p)
            if self.use_knowledge:
                so_bias *= torch.exp(self.so2p_factor)
            p_score_biased = torch.sigmoid(p_score+so_bias)
            if inference_problistic:
                p_prob = p_score_biased
            else:
                p_prob = (p_score_biased>predicate_confidence_threshold).float()
        return s_score_biased, o_score_biased, p_score_biased

    def predict(self, mode, pairs, inp_sf, inp_of, inp_ppf, inp_pvf, trans_mat=None,
            inference_steps=2, inference_problistic=False,
            inference_object_conf_thres=0.1, inference_predicate_conf_thres=0.05):
        B, _ = inp_sf.size()
        device = inp_sf.device
        if trans_mat is not None:
            s_init_prob = torch.mm(trans_mat[0], self.last_s_prob)
            o_init_prob = torch.mm(trans_mat[1], self.last_o_prob)
            p_init_prob = torch.mm(trans_mat[2], self.last_p_prob)
        else:
            s_init_prob = torch.zeros(B, self.object_num, device=device)
            o_init_prob = torch.zeros(B, self.object_num, device=device)
            p_init_prob = torch.zeros(B, self.predicate_num, device=device)

        s_prob, o_prob, p_prob = self.inference(inp_sf, inp_of, inp_ppf, inp_pvf,
                s_init_prob, o_init_prob, p_init_prob,
                steps=inference_steps, inference_problistic=inference_problistic)

        self.last_s_prob = s_prob
        self.last_o_prob = o_prob
        self.last_p_prob = p_prob

        obj_background_id = self.object_num-1
        s_max_idx = torch.argmax(s_prob, 1)
        o_max_idx = torch.argmax(o_prob, 1)
        valid_pair = (s_max_idx!=obj_background_id) & (o_max_idx!=obj_background_id)
        pairs = pairs[valid_pair]
        s_prob = s_prob[valid_pair, :-1]
        o_prob = o_prob[valid_pair, :-1]
        p_prob = p_prob[valid_pair]

        pairs = pairs.cpu().detach().numpy()
        s_prob = s_prob.cpu().detach().numpy()
        o_prob = o_prob.cpu().detach().numpy()
        p_prob = p_prob.cpu().detach().numpy()

        predictions = []
        for pair_id in range(len(pairs)):
            top_s_inds = np.where(s_prob[pair_id]>inference_object_conf_thres)[0]
            top_p_inds = np.where(p_prob[pair_id]>inference_predicate_conf_thres)[0]
            top_o_inds = np.where(o_prob[pair_id]>inference_object_conf_thres)[0]
            for s_class_id, p_class_id, o_class_id in product(top_s_inds, top_p_inds, top_o_inds):
                s_score = s_prob[pair_id, s_class_id]
                p_score = p_prob[pair_id, p_class_id]
                o_score = o_prob[pair_id, o_class_id]
                if mode == "predcls":
                    r_score = p_score
                else: 
                    r_score = s_score*p_score*o_score
                sub_id, obj_id = pairs[pair_id]
                predictions.append({
                    'sub_id': sub_id,
                    'obj_id': obj_id,
                    'triplet': (s_class_id, p_class_id,o_class_id),
                    'score': r_score,
                    'triplet_scores': (s_score, p_score, o_score)
                })

        return predictions
