import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def L2_norm(self, A):
        # A:N,V,V
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()

        y = None
        if self.adaptive:
            A = self.PA
            A = self.L2_norm(A)
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):

            A1 = A[i]
            A2 = x.view(N, C * T, V)
            A1 = A1.to(device)
            A2 = A2.to(device)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        
    def forward(self, query, key):
        attn_output, _ = self.attention(query, key, key)
        return attn_output





class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = np.stack([np.eye(num_point)] * num_set, axis=0)
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn1 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn2 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn3 = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.data_bn4 = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.joint_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        # self.joint_l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.joint_l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.joint_l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.joint_l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        # self.joint_l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.joint_l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.joint_l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        # self.joint_l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        # self.joint_l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.joint_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        self.joint_l2 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.joint_l3 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.joint_l4 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.joint_l5 = TCN_GCN_unit(256, 64, A, adaptive=adaptive)


        # self.bone_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        # self.bone_l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bone_l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bone_l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bone_l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        # self.bone_l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.bone_l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.bone_l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        # self.bone_l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        # self.bone_l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.bone_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        self.bone_l2 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.bone_l3 = TCN_GCN_unit(128, 128, A,  adaptive=adaptive)
        self.bone_l4 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.bone_l5 = TCN_GCN_unit(256, 64, A, adaptive=adaptive)


        # self.jointmotion_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        # self.jointmotion_l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.jointmotion_l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.jointmotion_l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.jointmotion_l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        # self.jointmotion_l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.jointmotion_l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.jointmotion_l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        # self.jointmotion_l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        # self.jointmotion_l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.jointmotion_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        self.jointmotion_l2 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.jointmotion_l3 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.jointmotion_l4 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.jointmotion_l5 = TCN_GCN_unit(256, 64, A, adaptive=adaptive)


        # self.bonemotion_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        # self.bonemotion_l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bonemotion_l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bonemotion_l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        # self.bonemotion_l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        # self.bonemotion_l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.bonemotion_l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        # self.bonemotion_l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        # self.bonemotion_l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        # self.bonemotion_l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.bonemotion_l1 = TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive)
        self.bonemotion_l2 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)
        self.bonemotion_l3 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.bonemotion_l4 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)
        self.bonemotion_l5 = TCN_GCN_unit(256, 64, A, adaptive=adaptive)


        self.attention = nn.MultiheadAttention(embed_dim=4*64, num_heads=8)

        self.fc = nn.Linear(64, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn1, 1)
        bn_init(self.data_bn2, 1)
        bn_init(self.data_bn3, 1)
        bn_init(self.data_bn4, 1)

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        feature_dim = 64
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.norm4 = nn.LayerNorm(feature_dim)
        self.norm5 = nn.LayerNorm(feature_dim)
        self.norm6 = nn.LayerNorm(feature_dim)


        self.cross_attention_1 = CrossAttention(dim=64, num_heads=4)  # 用于 joint 和 bone
        self.cross_attention_2 = CrossAttention(dim=64, num_heads=4)  # 用于 jointmotion 和 bonemotion
        self.cross_attention_3 = CrossAttention(dim=64, num_heads=4)  # 用于 joint 和 bone
        self.cross_attention_4 = CrossAttention(dim=64, num_heads=4)  # 用于 joint 和 bone
        self.cross_attention_5 = CrossAttention(dim=64, num_heads=4)  # 用于 joint 和 bone
        self.cross_attention_6 = CrossAttention(dim=64, num_heads=4)  # 用于 joint 和 bone

        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))
        self.weight4 = nn.Parameter(torch.tensor(1.0))
        self.weight5 = nn.Parameter(torch.tensor(1.0))
        self.weight6 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        joint = x[:,:3,:,:,:]
        bone = x[:,3:6,:,:,:]
        jointmotion = x[:,6:9,:,:,:]
        bonemotion = x[:,9:12,:,:,:]

        N, C, T, V, M = joint.size()
        
        joint = joint.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bone = bone.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        jointmotion = jointmotion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        bonemotion = bonemotion.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        joint = self.data_bn1(joint)
        bone = self.data_bn2(bone)
        jointmotion = self.data_bn3(jointmotion)
        bonemotion = self.data_bn4(bonemotion)

        joint = joint.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        bone = bone.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        jointmotion = jointmotion.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        bonemotion = bonemotion.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        
        
        joint = self.joint_l1(joint)
        bone = self.bone_l1(bone)
        jointmotion = self.jointmotion_l1(jointmotion)
        bonemotion = self.bonemotion_l1(bonemotion)

        joint = self.joint_l2(joint)
        bone = self.bone_l2(bone)
        jointmotion = self.jointmotion_l2(jointmotion)
        bonemotion = self.bonemotion_l2(bonemotion)

        joint = self.joint_l3(joint)
        bone = self.bone_l3(bone)
        jointmotion = self.jointmotion_l3(jointmotion)
        bonemotion = self.bonemotion_l3(bonemotion)

        joint = self.joint_l4(joint)
        bone = self.bone_l4(bone)
        jointmotion = self.jointmotion_l4(jointmotion)
        bonemotion = self.bonemotion_l4(bonemotion)

        joint = self.joint_l5(joint)
        bone = self.bone_l5(bone)
        jointmotion = self.jointmotion_l5(jointmotion)
        bonemotion = self.bonemotion_l5(bonemotion)
        

        # N*M,C,T,V
        joint_new = joint.size(1)
        bone_new = bone.size(1)
        jointmotion_new = jointmotion.size(1)
        bonemotion_new = bonemotion.size(1)

        joint = joint.view(N, M, joint_new, -1)
        bone = bone.view(N, M, bone_new, -1)
        jointmotion = jointmotion.view(N, M, jointmotion_new, -1)
        bonemotion = bonemotion.view(N, M, bonemotion_new, -1)

        joint = joint.mean(3).mean(1)
        bone = bone.mean(3).mean(1)
        jointmotion = jointmotion.mean(3).mean(1)
        bonemotion = bonemotion.mean(3).mean(1)



        joint_bone, _ = self.cross_attention_1(joint, bone, bone)
        joint_jointmotion, _ = self.cross_attention_2(joint, jointmotion, jointmotion)
        joint_bonemotion, _ = self.cross_attention_3(joint, bonemotion, bonemotion)
        bone_jointmotion, _ = self.cross_attention_4(bone, jointmotion, jointmotion)
        bone_bonemotion, _ = self.cross_attention_5(bone, bonemotion, bonemotion)
        jointmotion_bonemotion, _ = self.cross_attention_6(jointmotion, bonemotion, bonemotion)

        # 使用 LayerNorm 稳定每对 Cross-Attention 的输出
        joint_bone = self.norm1(joint_bone)
        joint_jointmotion = self.norm2(joint_jointmotion)
        joint_bonemotion = self.norm3(joint_bonemotion)
        bone_jointmotion = self.norm4(bone_jointmotion)
        bone_bonemotion = self.norm5(bone_bonemotion)
        jointmotion_bonemotion = self.norm6(jointmotion_bonemotion)

        # 可学习权重加权融合
        fusion = (self.weight1 * joint_bone + 
                  self.weight2 * joint_jointmotion + 
                  self.weight3 * joint_bonemotion + 
                  self.weight4 * bone_jointmotion + 
                  self.weight5 * bone_bonemotion + 
                  self.weight6 * jointmotion_bonemotion)
        
        # 分类前的 Dropout 和 Flatten 操作
        fusion = self.drop_out(fusion.mean(dim=1))  # 取均值或 Flatten
        return self.fc(fusion)

