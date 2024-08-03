import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .update import BasicUpdateBlock
from functools import partial
Align_Corners_Range = False

class DepthNet(nn.Module):
    def __init__(self, cnnpixel=False):
        super(DepthNet, self).__init__()


    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, pixel_wise_net, G=8):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(
            depth_values.shape[1], num_depth)
        num_views = len(features)
        view_weights = []
        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        B,C,H,W = ref_feature.shape
        similarity_sum = 0
        pixel_wise_weight_sum = 0
        ref_feature = ref_feature.view(B, G, C // G, H, W)

        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_feature = homo_warping_new(src_fea, src_proj_new, ref_proj_new, depth_values)
            warped_feature = warped_feature.view(B, G, C//G, num_depth, H, W)
            similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
            if pixel_wise_net != None:
                # if self.cnnpixel:
                sim_vol_norm = F.softmax(similarity.squeeze(1).detach(), dim=1)
                entropy = (- sim_vol_norm * torch.log(sim_vol_norm + 1e-7)).sum(dim=1, keepdim=True)
                view_weight = pixel_wise_net(entropy)
                view_weights.append(view_weight)

                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B,1,1,H,W]
                else:
                    similarity_sum += similarity * view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                del warped_feature, src_fea, src_proj, similarity, view_weight
            else:
                if self.training:
                    similarity_sum = similarity_sum + similarity  # [B, G, Ndepth, H, W]
                else:
                    similarity_sum += similarity
                del warped_feature, src_fea, src_proj, similarity
            # warped_volume = homo_warping(src_fea, src_proj[:, 2], ref_proj[:, 2], depth_values)

        del src_features, src_projs

        if pixel_wise_net != None:
            view_weights = torch.cat(view_weights, dim=1)
            similarity = similarity_sum.div_(pixel_wise_weight_sum + 1e-6)
            del ref_feature, pixel_wise_weight_sum, similarity_sum
        else:
            similarity = similarity_sum.div_(num_views-1)
            del ref_feature, similarity_sum

        # aggregate multiple feature volumes by variance
        # step 3. cost volume regularizationmodel_tmpx


        prob_volume_pre, pro = cost_regularization(similarity)
        prob_volume_pre = prob_volume_pre.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)
        # prob_volume_ = F.softmax(prob_volume_pre*5.0, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        return {"depth": depth,  "photometric_confidence": photometric_confidence, "view_weights":view_weights, "reg_volume":prob_volume_pre, "volume":similarity}


def build_gwc_volume(refimg_fea, targetimg_fea, num_groups):
    # B, C, H, W = refimg_fea.shape
    B, C, D, H, W = targetimg_fea.shape
    refimg_fea = refimg_fea.unsqueeze(2).repeat(1, 1, D, 1, 1)
    channels_per_group = C // num_groups
    volume = (refimg_fea * targetimg_fea).view([B, num_groups, channels_per_group, D, H, W]).mean(dim=2)
    volume = volume.contiguous()
    return volume


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img
def pro_bilinear_sampler(pro, depth_sample, depth_min, depth_max):
    D = pro.shape[-1]
    # print(pro.shape)
    b,d,h,w = depth_sample.shape

    disp = depth_to_disp(depth_sample, depth_min, depth_max) * (D-1)

    x0 = disp.permute(0, 2, 3, 1).reshape(b * h * w, 1, d, 1)
    y0 = torch.zeros_like(x0)

    disp_lvl = torch.cat([x0, y0], dim=-1)

    corr = bilinear_sampler(pro, disp_lvl)
    corr = corr.reshape(b, h, w, -1)
    corr = corr.permute(0, 3, 1, 2)

    return corr



def disp_to_depth(disp, min_depth, max_depth):

    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    scaled_disp = min_disp + (max_disp - min_disp) * disp

    scaled_disp = scaled_disp.clamp(min = 1e-4)
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    scaled_disp = 1 / depth

    min_disp = 1 / max_depth

    max_disp = 1 / min_depth

    disp = (scaled_disp - min_disp) / ((max_disp - min_disp)+1e-10)

    return disp


def upsample_depth(depth, mask, ratio=8):
    """ Upsample depth field [H/ratio, W/ratio, 2] -> [H, W, 2] using convex combination """
    N, _, H, W = depth.shape
    mask = mask.view(N, 1, 9, ratio, ratio, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(depth, [3, 3], padding=1)
    up_flow = up_flow.view(N, 1, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2)
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(N, ratio * H, ratio * W)

class GetCost_initvolume(nn.Module):
    def __init__(self):
        super(GetCost_initvolume, self).__init__()

    def forward(self, depth_values, features, proj_matrices, depth_interval, depth_max, depth_min, view_weights, CostNum=4, Inverse=True, G=8, iter=1, inter_iter=[1,1,1,1]):

        proj_matrices = torch.unbind(proj_matrices, 1)
        num_views = len(features)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        depth_interval = depth_interval * inter_iter[iter]
        if Inverse:
            depth_values = 1./depth_values
        depth_range_samples = get_depth_range_samples(cur_depth=depth_values.squeeze(1),
                                                      ndepth=CostNum,
                                                      depth_inteval_pixel=depth_interval.squeeze(1),
                                                      dtype=ref_feature[0].dtype,
                                                      device=ref_feature[0].device,
                                                      shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                             ref_feature.shape[3]],
                                                      max_depth=depth_max,
                                                      min_depth=depth_min)

        if Inverse:
            depth_range_samples = 1./depth_range_samples

        B, C, H, W = ref_feature.shape
        ref_feature = ref_feature.view(B, G, C//G, H, W)

        i = 0
        similarity_sum = 0
        pixel_wise_weight_sum = 0

        for src_fea, src_proj in zip(src_features, src_projs):
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_feature = homo_warping_new(src_fea, src_proj_new, ref_proj_new, depth_range_samples)
            warped_feature = warped_feature.view(B, G, C//G, CostNum, H, W)
            # print(depth_range_samples.mean(),warped_feature.shape, warped_feature.mean())
            similarity = (warped_feature * ref_feature.unsqueeze(3)).mean(2)
            if view_weights!= None:
                view_weight = view_weights[:, i].unsqueeze(1)  # [B,1,H,W]
                i = i + 1
                # gwc_volume = build_gwc_volume(ref_feature, warped_volume, 4)
                if self.training:
                    similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, G, Ndepth, H, W]
                    pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B,1,1,H,W]
                else:
                    similarity_sum += similarity * view_weight.unsqueeze(1)
                    pixel_wise_weight_sum += view_weight.unsqueeze(1)
                del warped_feature, src_fea, src_proj, similarity, view_weight
            else:
                if self.training:
                    similarity_sum = similarity_sum + similarity
                else:
                    similarity_sum += similarity
                del warped_feature, src_fea, src_proj, similarity
        del src_features, src_projs
        if view_weights != None:
            similarity = similarity_sum.div_(pixel_wise_weight_sum + 1e-6)
            del ref_feature, pixel_wise_weight_sum, similarity_sum
        else:
            similarity = similarity_sum.div_(num_views-1)
            del ref_feature, similarity_sum
        similarity = similarity.view(B, G * CostNum, H, W)

        return similarity, depth_range_samples

class GetCost(nn.Module):
    def __init__(self):
        super(GetCost, self).__init__()

    def forward(self, depth_values, pro, features, proj_matrices, depth_interval,
                depth_max, depth_min, view_weights, CostNum=4, Inverse=True,
                G=8, depth_max_cur_volume=0, depth_min_cur_volume=0,
        iter=1, inter_iter=[1,1,1,1]):

        proj_matrices = torch.unbind(proj_matrices, 1)
        num_views = len(features)
        corr = None
        # step 1. feature extraction

        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        depth_interval = depth_interval * inter_iter[iter]

        if Inverse:
            depth_values = 1./depth_values

        depth_range_samples = get_depth_range_samples(cur_depth=depth_values.squeeze(1),
                                                      ndepth=CostNum,
                                                      depth_inteval_pixel=depth_interval.squeeze(1),
                                                      dtype=ref_feature[0].dtype,
                                                      device=ref_feature[0].device,
                                                      shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                             ref_feature.shape[3]],
                                                      max_depth=depth_max,
                                                      min_depth=depth_min)

        if Inverse:
            depth_range_samples = 1./depth_range_samples



        similarity_pro = pro[-1]

        similarity = pro_bilinear_sampler(similarity_pro, depth_range_samples, depth_min_cur_volume, depth_max_cur_volume)




        similarity_list = [similarity]
        corr = pro_bilinear_sampler(pro[0], depth_range_samples, depth_min_cur_volume, depth_max_cur_volume)
        similarity_list.append(corr)



        similarity = torch.cat(similarity_list, dim=1)
        return similarity
def pool2x(x):
    return F.avg_pool2d(x, 3, stride=2, padding=1)

def pool4x(x):
    return F.avg_pool2d(x, 5, stride=4, padding=1)

def interp(x, dest):
    interp_args = {'mode': 'bilinear', 'align_corners': True}
    return F.interpolate(x, dest.shape[2:], **interp_args)


class Effi_MVS_plus(nn.Module):
    def __init__(self, args, refine=False, ndepths=48, depth_interals_ratio=[4,2,1], share_cr=False,
                 CostNum=4, inverse=True, stage_channel=True):
        super(Effi_MVS_plus, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = args.ndepths
        self.inverse = inverse
        self.depth_interals_ratio = depth_interals_ratio
        self.cost_num = 2

        seq_len = [int(e) for e in args.GRUiters.split(",")]
        self.seq_len = seq_len
        self.args = args

        self.num_stage = 3
        self.CostNum = args.CostNum
        self.CostNum_ratio = [4,2,1]
        self.GetCost = GetCost()
        self.GetCost_initvolume = GetCost_initvolume()
        self.stage_channel = stage_channel

        self.hdim_stage = [48,32,16]
        self.cdim_stage = [12,8,4]

        self.context_feature = [60,40,20]

        self.depth_stage_nums = [int(e) for e in args.ndepths.split(",")]
        # self.depth_stage_nums = [48, 8, 8]
        self.hdim = 32
        self.cdim = 32
        self.feat_ratio = [2,2,2]


        self.G = 1
        self.cost_dim_stage = [32, 16, 8]


        self.feature_in_channel = [8,16,32,64]
        self.context_in_channel = [4,8,16,32]
        # self.cost_dim = 16


        print("depths:{}, depth_intervals_ratio:{}, hdim_stage:{}, cdim_stage:{}, context_feature:{}, cost_dim_stage:{},G:{},cost_num:{},feature_in_channel{},context_in_channel{}************".format(
                self.depth_stage_nums, depth_interals_ratio, self.hdim_stage, self.cdim_stage, self.context_feature,self.cost_dim_stage,self.G,self.cost_num,self.feature_in_channel,self.context_in_channel))

        self.PixelwiseNet = nn.Sequential(ConvBnReLU(1, 16), ConvBnReLU(16, 16), ConvBnReLU(16, 8), nn.Conv2d(8, 1, 1),
                                         nn.Sigmoid())

        self.feature = P_1to8_FeatureNet_Fast(base_channels=4, in_channel=self.feature_in_channel,out_channel=self.cost_dim_stage, stage_channel=self.stage_channel)

        self.cnet_depth = P_1to8_FeatureNet_Fast(base_channels=4, in_channel=self.context_in_channel, out_channel=self.context_feature, stage_channel=self.stage_channel)




        self.update_block_depth1 = BasicUpdateBlock(hidden_dim=self.hdim_stage[0],
                                                    cost_dim=self.G * self.CostNum,
                                                    ratio=self.feat_ratio[0],
                                                    context_dim=self.cdim_stage[0], UpMask=True,
                                                    Inverse=self.inverse, cost_num=self.cost_num)

        self.update_block_depth2 = BasicUpdateBlock(hidden_dim=self.hdim_stage[1],
                                                    cost_dim=self.G * self.CostNum,
                                                    ratio=self.feat_ratio[1],
                                                    context_dim=self.cdim_stage[1], UpMask=True,
                                                    Inverse=self.inverse, cost_num=self.cost_num)

        self.update_block_depth3 = BasicUpdateBlock(hidden_dim=self.hdim_stage[2],
                                                    cost_dim=self.G * self.CostNum,
                                                    ratio=self.feat_ratio[2],
                                                    context_dim=self.cdim_stage[2], UpMask=True,
                                                    Inverse=self.inverse, cost_num=self.cost_num)
        self.update_block = nn.ModuleList([self.update_block_depth1,self.update_block_depth2,self.update_block_depth3])
        self.depthnet = DepthNet()



        self.CSP_R1 = cost_up_small(in_channels=self.G, base_channels=8)
        self.CSP_R2 = cost_up_small(in_channels=self.G, base_channels=8)
        self.CSP_R = nn.ModuleList([self.CSP_R1, self.CSP_R2])


        self.CSP_C1 = cost_up_small(in_channels=self.G, base_channels=8)
        self.CSP_C2 = cost_up_small(in_channels=self.G, base_channels=8)
        self.CSP_C = nn.ModuleList([self.CSP_C1, self.CSP_C2])


        self.cost_regularization = CostRegNet_2_sample_FPN3D_Fast(in_channels=self.G, base_channels=8)
        # self.cost_regularization = CostRegNet_2_sample_FPN3D(in_channels=self.G, base_channels=8)


    def forward(self, imgs, proj_matrices, depth_values):

        disp_min = depth_values[:, 0, None, None, None]
        disp_max = depth_values[:, -1, None, None, None]

        if self.inverse:
            depth_max_ = 1. / disp_min
            depth_min_ = 1. / disp_max
        else:
            depth_min_ = disp_min
            depth_max_ = disp_max
        depth_max = depth_max_
        depth_min = depth_min_
        depth_max2 = depth_max_
        depth_min2 = depth_min_

        self.scale_inv_depth = partial(disp_to_depth, min_depth=depth_min_, max_depth=depth_max_)
        depth_interval = (disp_max - disp_min) / depth_values.size(1)
        # step 1. feature extraction
        features = []
        depth_predictions = []
        photometric_confidence = 0
        last_mask = 0
        last_inv_depth = 0

        for nview_idx in range(imgs.size(1)):  # imgs shape (B, N, C, H, W)
            img = imgs[:, nview_idx]
            features.append(self.feature(img))
        cnet_depth = self.cnet_depth(imgs[:, 0])

        view_weights = 0

        hidden_d_list_init = []

        inp_d_list = []
        for stage_idx in range(self.num_stage):
            cnet_depth_stage = cnet_depth["stage{}".format(stage_idx + 1)]
            if self.stage_channel:
                hidden_d, inp_d = torch.split(cnet_depth_stage,
                                              [self.hdim_stage[stage_idx], self.cdim_stage[stage_idx]], dim=1)
            else:
                hidden_d, inp_d = torch.split(cnet_depth_stage, [self.hdim, self.cdim], dim=1)
            current_hidden_d = torch.tanh(hidden_d)
            inp_d = torch.relu(inp_d)
            hidden_d_list_init.append(current_hidden_d)
            inp_d_list.append(inp_d)

        hidden_d_list = hidden_d_list_init
        init_volume = 0


        for stage_idx in range(self.num_stage):

            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            ref_feature = features_stage[0]
            if stage_idx == 0:
                depth_range_samples = get_depth_range_samples(cur_depth=depth_values,
                                                              ndepth=self.depth_stage_nums[0],
                                                              depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                                                              dtype=ref_feature[0].dtype,
                                                              device=ref_feature[0].device,
                                                              shape=[ref_feature.shape[0], ref_feature.shape[2],
                                                                     ref_feature.shape[3]],
                                                              max_depth=disp_max,
                                                              min_depth=disp_min)
                if self.inverse:
                    depth_range_samples = 1./depth_range_samples
                init_depth = self.depthnet(features_stage, proj_matrices_stage, depth_values=depth_range_samples,
                                           num_depth=self.depth_stage_nums[0], cost_regularization=self.cost_regularization, pixel_wise_net=self.PixelwiseNet,G=self.G)

                photometric_confidence = init_depth["photometric_confidence"]
                photometric_confidence = F.interpolate(photometric_confidence.unsqueeze(1),[ref_feature.shape[2]*4, ref_feature.shape[3]*4], mode='nearest')
                photometric_confidence = photometric_confidence.squeeze(1)

                view_weights = init_depth['view_weights']

                init_volume = init_depth['volume']
                cur_volume = init_volume.squeeze(1)
                reg_volume = init_depth['reg_volume']
                init_depth = init_depth['depth']

                cur_depth = init_depth.unsqueeze(1)

                depth_predictions = [init_depth.squeeze(1)]

            else:
                cur_depth = depth_predictions[-1].unsqueeze(1)
                cur_depth = cur_depth.detach()

                view_weights = F.interpolate(view_weights, scale_factor=2, mode='nearest')


                depth_cost_func_initvolume = partial(self.GetCost_initvolume, features=features_stage,
                                          proj_matrices=proj_matrices_stage,
                                          depth_interval=depth_interval * self.depth_interals_ratio[stage_idx],
                                          depth_max=disp_max,
                                          depth_min=disp_min, view_weights=view_weights, CostNum=self.depth_stage_nums[stage_idx],
                                          Inverse=self.inverse,
                                          G=self.G)
                cur_volume, depth_range_samples_ = depth_cost_func_initvolume(cur_depth)
                depth_max2 = depth_range_samples_[:, 0:1]
                depth_min2 = depth_range_samples_[:, -1:]


                B, D, H_, W_ = depth_range_samples_.shape

                depth_range_samples_low = F.interpolate(depth_range_samples_.unsqueeze(1), size=[D,H_//2,W_//2], mode='nearest').squeeze(1)
                #
                cur_volume_ = cur_volume.view(B,self.G,D,H_, W_)
                pro = reg_volume.permute(0, 2, 3, 1).reshape(B * (H_//2)*(W_//2), 1, 1,self.depth_stage_nums[stage_idx-1])

                reg_volume_ = pro_bilinear_sampler(pro, depth_range_samples_low, depth_min, depth_max)

                reg_volume, _ = self.CSP_R[stage_idx-1](cur_volume_, reg_volume_.unsqueeze(1))
                reg_volume = reg_volume.squeeze(1)



                init_volume = init_volume.squeeze(1)
                init_volume = init_volume.permute(0, 2, 3, 1).reshape(B * (H_ // 2) * (W_ // 2), 1, 1,
                                                                   self.depth_stage_nums[stage_idx - 1])
                init_volume = pro_bilinear_sampler(init_volume, depth_range_samples_low, depth_min, depth_max)
                init_volume, _ = self.CSP_C[stage_idx-1](cur_volume_, init_volume.unsqueeze(1))

                cur_volume = init_volume.squeeze(1)

                depth_max = depth_max2
                depth_min = depth_min2


            inv_cur_depth = depth_to_disp(cur_depth, depth_min_, depth_max_)

            B, D, H, W = reg_volume.shape
            pro = [reg_volume.permute(0, 2, 3, 1).reshape(B * H * W, 1, 1, D)]

            B, D, H, W = cur_volume.shape
            cur_volume = cur_volume.permute(0, 2, 3, 1).reshape(B * H * W, 1, 1, D)
            pro.append(cur_volume)

            depth_cost_func = partial(self.GetCost, pro=pro, features=features_stage, proj_matrices=proj_matrices_stage,
                                      depth_interval=depth_interval*self.depth_interals_ratio[stage_idx], depth_max=disp_max,
                                      depth_min=disp_min, view_weights=view_weights, CostNum=self.CostNum, Inverse=self.inverse,
                                      G=self.G, depth_max_cur_volume=depth_max2, depth_min_cur_volume=depth_min2)

            current_hidden_d, up_mask_seqs, inv_depth_seqs = self.update_block[stage_idx](hidden_d_list[stage_idx], depth_cost_func,
                                                                             inv_cur_depth,
                                                                             inp_d_list[stage_idx], seq_len=self.seq_len[stage_idx],
                                                                             scale_inv_depth=self.scale_inv_depth)


            for up_mask_i, inv_depth_i in zip(up_mask_seqs, inv_depth_seqs):
                depth_predictions.append(self.scale_inv_depth(inv_depth_i)[1].squeeze(1))
                last_mask = up_mask_i
                last_inv_depth = inv_depth_i

            inv_depth_up = upsample_depth(last_inv_depth, last_mask, ratio=self.feat_ratio[stage_idx]).unsqueeze(1)
            final_depth = self.scale_inv_depth(inv_depth_up)[1].squeeze(1)

            depth_predictions.append(final_depth)

        return {"depth": depth_predictions, "photometric_confidence": photometric_confidence}