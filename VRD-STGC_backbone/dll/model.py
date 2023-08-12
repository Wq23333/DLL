import torch
from torch import nn
import os
from IPython import embed
from typing import Any, Optional, Tuple
from torch.autograd  import  Function

class GRL(Function):
    def forward(self,input):
        return input
    def backward(self,grad_output):
        grad_input = grad_output.neg()
        print(grad_input)
        return grad_input


class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 0.01) -> torch.Tensor: #1.0
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


class RelationPredictor(nn.Module):
    def __init__(self):
        super(RelationPredictor, self).__init__()
        num_classes = 133
        self.fc1 = nn.Linear(2048, 512)
        self.fc2= nn.Linear(2048, 512)
        self.fc1_i3d = nn.Linear(832, 512)
        self.fc2_i3d = nn.Linear(832, 512)
        self.fc_motion = nn.Linear(118, 512)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.8)

        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc2.bias, 0)

        self.correlation = nn.Parameter(torch.eye(133,133),requires_grad=True)

        dim = 640
        self.action_feature = nn.Linear(2560,dim)
        self.spacial_feature = nn.Linear(2560,dim)
        self.action_2_spacial = nn.Linear(dim, dim)
        self.spacial_2_action = nn.Linear(dim, dim)
        self.grl = GRL_Layer()

        self.fc_spacial = nn.Linear(dim, 13)
        self.fc_action = nn.Linear(dim, 29)

        nn.init.normal_(self.action_feature.weight, mean=0, std=0.01)
        nn.init.constant_(self.action_feature.bias, 0)
        nn.init.normal_(self.spacial_feature.weight, mean=0, std=0.01)
        nn.init.constant_(self.spacial_feature.bias, 0)

        nn.init.normal_(self.action_2_spacial.weight, mean=0, std=0.01)
        nn.init.constant_(self.action_2_spacial.bias, 0)
        nn.init.normal_(self.spacial_2_action.weight, mean=0, std=0.01)
        nn.init.constant_(self.spacial_2_action.bias, 0)

        nn.init.normal_(self.fc_spacial.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_spacial.bias, 0)
        nn.init.normal_(self.fc_action.weight, mean=0, std=0.01)
        nn.init.constant_(self.fc_action.bias, 0)
        self.criterion = nn.KLDivLoss(reduction="batchmean",log_target=True)
        self.spacial_list, self.action_list, self.spacial_exist, self.action_exist, self.spacial_unexist, self.action_unexist, self.both_exist= self.get_id()
        self.spacial_list, self.action_list, self.spacial_exist, self.action_exist, self.spacial_unexist, self.action_unexist, self.both_exist = self.spacial_list, self.action_list, self.spacial_exist.cuda(), self.action_exist.cuda(), self.spacial_unexist.cuda(), self.action_unexist.cuda(), self.both_exist.cuda()
    
    def get_id(self):
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
        return spacial_list, action_list, spacial_exist, action_exist, spacial_unexist, action_unexist, both_exist
    def forward(self, x1, x2, x1_i3d, x2_i3d, motion):
        ori_shape = x1.shape
        i3d_shape = x1_i3d.shape
        motion_shape = motion.shape
        x1 = x1.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x2 = x2.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x1_i3d = x1_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x2_i3d = x2_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x_m = motion.view(motion_shape[0]*motion_shape[1], -1)

        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x1_i3d = self.drop(x1_i3d)
        x2_i3d = self.drop(x2_i3d)
        x_m = self.drop(x_m)

        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)
        x1_i3d = self.fc1_i3d(x1_i3d)
        x1_i3d = self.relu(x1_i3d)
        x2_i3d = self.fc2_i3d(x2_i3d)
        x2_i3d = self.relu(x2_i3d)
        x_m = self.fc_motion(x_m)
        x_m = self.relu(x_m)
        x = torch.cat((x1, x1_i3d, x_m, x2, x2_i3d), 1)

        action_feat =  self.action_feature(x)
        action_feat = self.relu(action_feat)
        spacial_feat =  self.spacial_feature(x)
        spacial_feat = self.relu(spacial_feat)

        p_score_action = self.fc_action(action_feat)
        p_score_spacial = self.fc_spacial(spacial_feat)

        
        tmp_action = self.grl.forward(action_feat)
        wrong_spacial_feat = self.action_2_spacial(tmp_action)
        wrong_spacial_output = self.fc_spacial(wrong_spacial_feat)

        tmp_spacial = self.grl.forward(spacial_feat)
        wrong_action_feat = self.spacial_2_action(tmp_spacial)
        wrong_action_output = self.fc_action(wrong_action_feat)

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

        loss_action = self.criterion(torch.log(sig_wrong_spacial_output+1e-12), torch.log(sig_p_s_spacial+1e-12))*0.1
        loss_spacial = self.criterion(torch.log(sig_wrong_action_output+1e-12), torch.log(sig_p_s_action+1e-12))*0.1
        
        p_score_s = torch.zeros(size = (x1.shape[0], 133),dtype=torch.float32).cuda()
        p_score_a = torch.zeros(size = (x1.shape[0], 133),dtype=torch.float32).cuda()
        for i in range(len(self.spacial_list)):
            p_score_s[:,self.spacial_list[i]] = p_score_spacial[:,i].unsqueeze(1)
        for i in range(len(self.action_list)):
            p_score_a[:,self.action_list[i]] = p_score_action[:,i].unsqueeze(1)
        radio = 0.3
        p_score = p_score_a * radio * self.both_exist + p_score_s * (1-radio) * self.both_exist + p_score_s * self.action_unexist + p_score_a * self.spacial_unexist
        thred=0.001
        steps = 1
        for st in range(steps):
            sp_score = torch.zeros(size = (p_score.shape[0],13),dtype=torch.float32).cuda()
            ac_score = torch.zeros(size = (p_score.shape[0],29),dtype=torch.float32).cuda()
            for i in range(len(self.spacial_list)):
                sp_score[:,i] = p_score[:,self.spacial_list[i]].mean(dim=1)
            for i in range(len(self.action_list)):
                ac_score[:,i] = p_score[:,self.action_list[i]].mean(dim=1)
            
            p_score_spacial = p_score_spacial * (1-thred) + sp_score * thred
            p_score_action = p_score_action * (1-thred) + ac_score * thred

            p_score_s = torch.zeros(size = (p_score.shape),dtype=torch.float32).cuda()
            p_score_a = torch.zeros(size = (p_score.shape),dtype=torch.float32).cuda()
            for i in range(len(self.spacial_list)):
                p_score_s[:,self.spacial_list[i]] = p_score_spacial[:,i].unsqueeze(1)
            for i in range(len(self.action_list)):
                p_score_a[:,self.action_list[i]] = p_score_action[:,i].unsqueeze(1)
            radio = 0.3
            p_score = p_score_a * radio * self.both_exist + p_score_s * (1-radio) * self.both_exist + p_score_s * self.action_unexist + p_score_a * self.spacial_unexist

      
        x = p_score
        x = x.view(ori_shape[0], ori_shape[1], -1)

        loss_PDL = loss_action + loss_spacial
        correlaiont_matrix = self.correlation
        return  correlaiont_matrix, x, loss_PDL

    def test(self, x1, x2, x1_i3d, x2_i3d, motion):
        ori_shape = x1.shape
        i3d_shape = x1_i3d.shape
        motion_shape = motion.shape
        x1 = x1.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x2 = x2.view(ori_shape[0]*ori_shape[1], ori_shape[2])
        x1_i3d = x1_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x2_i3d = x2_i3d.view(i3d_shape[0]*i3d_shape[1], i3d_shape[2])
        x_m = motion.view(motion_shape[0]*motion_shape[1], -1)

        x1 = self.drop(x1)
        x2 = self.drop(x2)
        x1_i3d = self.drop(x1_i3d)
        x2_i3d = self.drop(x2_i3d)
        x_m = self.drop(x_m)

        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x2 = self.fc2(x2)
        x2 = self.relu(x2)
        x1_i3d = self.fc1_i3d(x1_i3d)
        x1_i3d = self.relu(x1_i3d)
        x2_i3d = self.fc2_i3d(x2_i3d)
        x2_i3d = self.relu(x2_i3d)
        x_m = self.fc_motion(x_m)
        x_m = self.relu(x_m)
        x = torch.cat((x1, x1_i3d, x_m, x2, x2_i3d), 1)

        action_feat =  self.action_feature(x)
        action_feat = self.relu(action_feat)
        spacial_feat =  self.spacial_feature(x)
        spacial_feat = self.relu(spacial_feat)

        p_score_action = self.fc_action(action_feat)
        p_score_spacial = self.fc_spacial(spacial_feat)
    
        p_score_s = torch.zeros(size = (x1.shape[0],133),dtype=torch.float32).cuda()
        p_score_a = torch.zeros(size = (x1.shape[0],133),dtype=torch.float32).cuda()
        for i in range(len(self.spacial_list)):
            p_score_s[:,self.spacial_list[i]] = p_score_spacial[:,i].unsqueeze(1)
        for i in range(len(self.action_list)):
            p_score_a[:,self.action_list[i]] = p_score_action[:,i].unsqueeze(1)
        radio = 0.3
        p_score = p_score_a * radio * self.both_exist + p_score_s * (1-radio) * self.both_exist + p_score_s * self.action_unexist + p_score_a * self.spacial_unexist


        x = p_score
        x = x.view(ori_shape[0], ori_shape[1], -1)
        correlaiont_matrix = self.correlation
        return  correlaiont_matrix, x
