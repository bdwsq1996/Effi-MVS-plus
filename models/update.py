import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *





class DepthHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, scale=False):
        super(DepthHead, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=0.1)


    def forward(self, x_d, act_fn=torch.tanh):
        out = self.conv2(self.relu(self.conv1(x_d)))
        if self.training and self.dropout is not None:
            out = self.dropout(out)
        # out = F.relu(out)
        # out = torch.clip(out, max = 2)
        # return out - 1
        return act_fn(out)





class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, *x_list):
        x = torch.cat(x_list, dim=1)
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h






class UpMaskNet(nn.Module):
    def __init__(self, hidden_dim=128, ratio=8):
        super(UpMaskNet, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, feat):
        # scale mask to balence gradients
        mask = .25 * self.mask(feat)
        return mask

class ProjectionInput(nn.Module):
    def __init__(self, cost_dim, hidden_dim, context_dim, out_chs, depth_num=1, G=8):
        super().__init__()

        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.convd1 = nn.Conv2d(depth_num, hidden_dim, 7, padding=3)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convd = nn.Conv2d(hidden_dim + hidden_dim, hidden_dim-context_dim, 3, padding=1)
        # self.convc = nn.Conv2d(hidden_dim, hidden_dim//2, 1, padding=0)
        # self.out_chs = hidden_dim//2
        self.convc = nn.Conv2d(hidden_dim, hidden_dim, 1, padding=0)
        self.out_chs = hidden_dim

        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, disp, cost, context):
        # print(cost.size())
        cor = F.relu(self.convc1(cost))
        cor = F.relu(self.convc2(cor))
        dfm = F.relu(self.convd1(disp))
        dfm = F.relu(self.convd2(dfm))
        cor_dfm = torch.cat([cor, dfm], dim=1)
        cor_dfm = self.convd(cor_dfm)
        cor_dfm = self.convc(torch.cat([cor_dfm, context], dim=1))
        out_d = F.relu(cor_dfm)

        if self.training and self.dropout is not None:
            out_d = self.dropout(out_d)
        return out_d

class BasicUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=128, cost_dim=256, ratio=8, context_dim=64 ,UpMask=False, Inverse=False, cost_num=1, G=8):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = ProjectionInput(cost_dim=cost_dim*cost_num, hidden_dim=hidden_dim, context_dim=context_dim, out_chs=hidden_dim, G=G)
        self.depth_gru = ConvGRU(hidden_dim=hidden_dim, input_dim=self.encoder.out_chs)
        self.depth_head = DepthHead(hidden_dim, hidden_dim=hidden_dim, scale=False)
        self.UpMask = UpMask
        self.Inverse = Inverse
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim*2, ratio*ratio*9, 1, padding=0))

    def forward(self, net, depth_cost_func, inv_depth, context, seq_len=4, scale_inv_depth=None):
        inv_depth_list = []
        mask_list = []

        for i in range(seq_len):
            # TODO detach()
            inv_depth = inv_depth.detach()
            input_features = self.encoder(inv_depth, depth_cost_func(scale_inv_depth(inv_depth)[1], iter=i), context)
            # print(input_features.shape, context.shape)
            # inp_i = torch.cat([context, input_features], dim=1)
            net = self.depth_gru(net, input_features)
            delta_inv_depth = self.depth_head(net)
            inv_depth_test = inv_depth + delta_inv_depth
            # if inv_depth_test.mean() != inv_depth_test.mean():
            #     print("error2")
            #     print(delta_inv_depth.mean())
            #     print(net.mean())
            #     print(inv_depth.mean())
            #     print(input_features.mean())
            #     print(depth_cost_func(scale_inv_depth(inv_depth)[1]).mean())
            inv_depth = inv_depth_test
            inv_depth_list.append(inv_depth)
            if self.UpMask and i == seq_len - 1 :
                mask = .25 * self.mask(net)
                mask_list.append(mask)
            else:
                mask_list.append(inv_depth)
        return net, mask_list, inv_depth_list



# def upsample_depth(depth, mask, ratio=2):
#     """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
#     N, _, H, W = depth.shape
#     mask = mask.view(N, 1, 9, ratio, ratio, H, W)
#     mask = torch.softmax(mask, dim=2)
#
#     up_flow = F.unfold(depth, [3, 3], padding=1)
#     up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)
#
#     up_flow = torch.sum(mask * up_flow, dim=2)
#     up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
#     return up_flow.reshape(N, ratio * H, ratio * W)

