"""This source code is from Vis-MVSNet (https://github.com/jzhangbs/Vis-MVSNet)"""
from typing import List

import torch
import torch.nn.functional as F


def get_pixel_grids(height, width):
    x_coord = (torch.arange(width, dtype=torch.float32).cuda() + 0.5).repeat(height, 1)
    y_coord = (torch.arange(height, dtype=torch.float32).cuda() + 0.5).repeat(width, 1).t()
    ones = torch.ones_like(x_coord)
    indices_grid = torch.stack([x_coord, y_coord, ones], dim=-1).unsqueeze(-1)  # hw31
    return indices_grid


def bin_op_reduce(lst: List, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result


def idx_img2cam(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    idx_cam = cam[:, 1:2, :3, :3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[..., -1:, :] + 1e-9) * depth.permute(0, 2, 3, 1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[..., -1:, :])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo


def idx_cam2world(idx_cam_homo, cam):  # nhw41 -> nhw41
    idx_world_homo = cam[:, 0:1, ...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[..., -1:, :] + 1e-9)  # nhw41
    return idx_world_homo


def idx_world2cam(idx_world_homo, cam):  # nhw41 -> nhw41
    idx_cam_homo = cam[:, 0:1, ...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[..., -1:, :] + 1e-9)  # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):  # nhw41 -> nhw31
    idx_cam = idx_cam_homo[..., :3, :] / (idx_cam_homo[..., 3:4, :] + 1e-9)  # nhw31
    idx_img_homo = cam[:, 1:2, :3, :3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[..., -1:, :] + 1e-9)
    return idx_img_homo


def project_img(src_img, dst_depth, src_cam, dst_cam, height=None, width=None):  # nchw, n1hw -> nchw, n1hw
    if height is None: height = src_img.size()[-2]
    if width is None: width = src_img.size()[-1]
    dst_idx_img_homo = get_pixel_grids(height, width).unsqueeze(0)  # nhw31
    dst_idx_cam_homo = idx_img2cam(dst_idx_img_homo, dst_depth, dst_cam)  # nhw41
    dst_idx_world_homo = idx_cam2world(dst_idx_cam_homo, dst_cam)  # nhw41
    dst2src_idx_cam_homo = idx_world2cam(dst_idx_world_homo, src_cam)  # nhw41
    dst2src_idx_img_homo = idx_cam2img(dst2src_idx_cam_homo, src_cam)  # nhw31
    warp_coord = dst2src_idx_img_homo[..., :2, 0]  # nhw2
    warp_coord[..., 0] /= width
    warp_coord[..., 1] /= height
    warp_coord = (warp_coord * 2 - 1).clamp(-1.1, 1.1)  # nhw2
    in_range = bin_op_reduce([-1 <= warp_coord[..., 0], warp_coord[..., 0] <= 1, -1 <= warp_coord[..., 1], warp_coord[..., 1] <= 1],
                             torch.min).to(src_img.dtype).unsqueeze(1)  # n1hw
    warped_img = F.grid_sample(src_img, warp_coord, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped_img, in_range


def prob_filter(ref_prob, prob_thresh, greater=True):  # n31hw -> n1hw
    mask = None
    for i, p in enumerate(prob_thresh):
        if mask is None:
            mask = (ref_prob[:, [i]] > p)
        else:
            mask = mask & (ref_prob[:, [i]] > p)
    # mask = ref_prob > prob_thresh if greater else ref_prob < prob_thresh
    return mask


def get_reproj(ref_depth, srcs_depth, ref_cam, srcs_cam):  # n1hw, nv1hw -> n1hw
    n, v, _, h, w = srcs_depth.size()
    srcs_depth_f = srcs_depth.view(n * v, 1, h, w)
    srcs_cam_f = srcs_cam.view(n * v, 2, 4, 4)
    ref_depth_r = ref_depth.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 1, h, w)
    ref_cam_r = ref_cam.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 2, 4, 4)
    idx_img = get_pixel_grids(h, w).unsqueeze(0)  # 1hw31

    srcs_idx_cam = idx_img2cam(idx_img, srcs_depth_f, srcs_cam_f)  # Nhw41
    srcs_idx_world = idx_cam2world(srcs_idx_cam, srcs_cam_f)  # Nhw41
    srcs2ref_idx_cam = idx_world2cam(srcs_idx_world, ref_cam_r)  # Nhw41
    srcs2ref_idx_img = idx_cam2img(srcs2ref_idx_cam, ref_cam_r)  # Nhw31
    srcs2ref_xyd = torch.cat([srcs2ref_idx_img[..., :2, 0], srcs2ref_idx_cam[..., 2:3, 0]], dim=-1).permute(0, 3, 1, 2)  # N3hw

    reproj_xyd_f, in_range_f = project_img(srcs2ref_xyd, ref_depth_r, srcs_cam_f, ref_cam_r)  # N3hw, N1hw
    reproj_xyd = reproj_xyd_f.view(n, v, 3, h, w)
    in_range = in_range_f.view(n, v, 1, h, w)
    return reproj_xyd, in_range


def vis_filter(ref_depth, reproj_xyd, in_range, img_dist_thresh, depth_thresh, vthresh):
    n, v, _, h, w = reproj_xyd.size()
    xy = get_pixel_grids(h, w).permute(3, 2, 0, 1).unsqueeze(1)[:, :, :2]  # 112hw
    dist_masks = (reproj_xyd[:, :, :2, :, :] - xy).norm(dim=2, keepdim=True) < 1./img_dist_thresh  # nv1hw
    # depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:, :, 2:, :, :]).abs() < \
    #               (torch.max(ref_depth.unsqueeze(1), reproj_xyd[:, :, 2:, :, :]) * depth_thresh)  # nv1hw
    depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:, :, 2:, :, :]).abs() < 1./depth_thresh  # nv1hw
    in_range = torch.ones_like(dist_masks)
    masks = bin_op_reduce([in_range,dist_masks.to(ref_depth.dtype), depth_masks.to(ref_depth.dtype)], torch.min)  # nv1hw
    # mask = masks.sum(dim=1) >= (vthresh - 1.1)  # n1hw
    mask = masks.sum(dim=1) >= (vthresh - 1.1)  # n1hw
    return masks, mask


def ave_fusion(ref_depth, reproj_xyd, masks):
    ave = ((reproj_xyd[:, :, 2:, :, :] * masks).sum(dim=1) + ref_depth) / (masks.sum(dim=1) + 1)  # n1hw
    return ave

def get_reproj_dynamic(ref_depth, srcs_depth, ref_cam, srcs_cam):  # n1hw, nv1hw -> n1hw
    n, v, _, h, w = srcs_depth.size()
    srcs_depth_f = srcs_depth.view(n * v, 1, h, w)
    srcs_cam_f = srcs_cam.view(n * v, 2, 4, 4)
    ref_cam_r = ref_cam.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 2, 4, 4)
    ref_depth_f = ref_depth.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 1, h, w)
    idx_img = get_pixel_grids(h, w).unsqueeze(0)  # 1hw31  # [1,h,w,3,1]

    ref_idx_cam = idx_img2cam(idx_img, ref_depth_f, ref_cam_r)  # Nhw41     k^-1  [x,y,1] * d
    ref_idx_world = idx_cam2world(ref_idx_cam, ref_cam_r)  # Nhw41       (R-1)  k^-1  [x,y,1] * d  / [:,:,-1]
    ref2src_idx_cam = idx_world2cam(ref_idx_world,
                                     srcs_cam_f)  # Nhw41   R*  [(R-1)  k^-1  [x,y,1] * d  / [:,:,-1] ]  / [:,:,-1]
    ref2src_idx_img = idx_cam2img(ref2src_idx_cam,
                                   srcs_cam_f)  # Nhw31   K * R*  [(R-1)  k^-1  [x,y,1] * d  / [:,:,-1] ]  / [:,:,-1]


    warp_coord = ref2src_idx_img[..., :2, 0]  # nhw2
    proj_x_normalized = warp_coord[...,0] / ((w-1)/2) - 1
    proj_y_normalized = warp_coord[...,1] /((h-1)/2) -1
  
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [n,h,w,2]

    warped_src_depth = F.grid_sample(srcs_depth_f, proj_xy, mode='bilinear', padding_mode='zeros', align_corners=True)

    warp_homo_coord = torch.cat([warp_coord, torch.ones_like(warp_coord[...,-1:])],dim=-1).unsqueeze(-1) # [n,h,w,3]

    src_idx_cam = idx_img2cam(warp_homo_coord, warped_src_depth, srcs_cam_f)  # Nhw41     k^-1  [x,y,1] * d

    src_idx_world = idx_cam2world(src_idx_cam, srcs_cam_f)  # Nhw41
    src2ref_idx_cam = idx_world2cam(src_idx_world, ref_cam_r) # Nhw41
    reproj_depth = src2ref_idx_cam[:,:,:,2,0].clone()    #  # n h w
    src2ref_idx_corrd = idx_cam2img(src2ref_idx_cam, ref_cam_r) # Nhw31

    # bn  3 h w
    reproj_xyd_f = torch.cat([src2ref_idx_corrd[...,:2,0], reproj_depth.unsqueeze(-1)],dim=-1).permute(0,3,1,2)
    reproj_xyd = reproj_xyd_f.reshape(n,v,3,h,w)
    # print(ref_idx_cam[:,:,:,-1,0], src2ref_idx_cam[:,:,:,-1,0])
    return reproj_xyd, ref_idx_cam, src2ref_idx_cam


def vis_filter_dynamic(ref_depth, reproj_xyd, ref_idx_world, src2ref_idx_cam, dist_base=4, rel_diff_base=1300, thres_view=2, relative=False):
    device = reproj_xyd.device
    n, v, _, h, w = reproj_xyd.size()

    ref_idx_world = ref_idx_world[:,:,:,:3,0].reshape(n, v, h, w, 3).permute(0,1,4,2,3)
    src2ref_idx_cam = src2ref_idx_cam[:,:,:,:3,0].reshape(n, v, h, w, 3).permute(0,1,4,2,3)

    # xyz_dff = (ref_idx_world - src2ref_idx_cam).norm(dim=2, keepdim=True)
    # xy_dff = (ref_idx_world[:,:,:2,:,:] - src2ref_idx_cam[:,:,:2,:,:]).norm(dim=2, keepdim=True)
    xy = get_pixel_grids(h, w).permute(3, 2, 0, 1).unsqueeze(1)[:, :, :2]  # 112hw
    corrd_diff = (reproj_xyd[:, :, :2, :, :] - xy).norm(dim=2, keepdim=True) # nv1hw
    if relative:
        depth_diff = (ref_depth.unsqueeze(1) - reproj_xyd[:, :, 2:, :, :]).abs()  / ref_depth.unsqueeze(1) # nv1hw
    else:
        depth_diff = (ref_depth.unsqueeze(1) - reproj_xyd[:, :, 2:, :, :]).abs()  # nv1hw
    # print(xyz_dff[0, 0, 0, h // 2, w // 2], depth_diff[0, 0, 0, h // 2, w // 2], xy_dff[0, 0, 0, h // 2, w // 2])
    # print(xyz_dff.mean(), depth_diff.mean())
    # print(thres_view,v+1)
    dist_thred = torch.arange(thres_view,v+1).reshape(1,1,-1,1,1).repeat(n,v,1,1,1).to(device) / dist_base
    relative_dist_thred = torch.arange(thres_view,v+1).reshape(1,1,-1,1,1).repeat(n,v,1,1,1).to(device) / rel_diff_base
    masks = torch.min(corrd_diff<dist_thred, depth_diff < relative_dist_thred) # [n,v,v-1, h,w]
    # masks = xyz_dff < relative_dist_thred
    mask = masks[:,:,-1:,:,:] # [n,v,1,h,w]

    return masks, mask



def vis_filter_dynamic_open3d(ref_depth, src_depths, ref_cam, src_cams, reproj_xyd, dist_base=4, rel_diff_base=1300, thres_view=2, relative=False):
    device = reproj_xyd.device
    n, v, _, h, w = reproj_xyd.size()

    # src_depths = src_depths[:,0:1]
    # src_cams = src_cams[:,0:1]

    ref_pc, aligned_pcs, dist = filter_depth(ref_depth, src_depths, ref_cam, src_cams)
    aligned_pcs = aligned_pcs.unsqueeze(0)
    dist = dist.unsqueeze(0)
    # dist_thred = torch.arange(thres_view,v+1).reshape(1,1,-1,1,1).repeat(n,v,1,1,1).to(device) / dist_base
    relative_dist_thred = torch.arange(thres_view,v+1).reshape(1,1,-1,1,1).repeat(n,v,1,1,1).to(device) / rel_diff_base
    masks = dist<relative_dist_thred # [n,v,v-1, h,w]
    # masks = xyz_dff < relative_dist_thred
    mask = masks[:,:,-1:,:,:] # [n,v,1,h,w]

    return aligned_pcs, masks, mask,



def generate_points_from_depth(depth, proj):
    '''
    :param depth: (B, 1, H, W)
    :param proj: (B, 4, 4)
    :return: point_cloud (B, 3, H, W)
    '''


    # proj = proj_new
    if len(proj.shape) == 4:
        proj_new = proj[:, 0].clone()
        proj_new[:, :3, :4] = torch.matmul(proj[:, 1, :3, :3], proj[:, 0, :3, :4])
        proj = proj_new
    # print(depth.shape, proj.shape)

    batch, height, width = depth.shape[0], depth.shape[2], depth.shape[3]
    inv_proj = torch.inverse(proj)



    rot = inv_proj[:, :3, :3]  # [B,3,3]
    trans = inv_proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    # print(rot.shape, xyz.shape)
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * depth.view(batch, 1, -1)
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xyz = proj_xyz.view(batch, 3, height, width)

    return proj_xyz


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]
    # print(src_proj.shape, ref_proj.shape)
    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        # print(rot.shape, xyz.shape)
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2) * depth_values.view(-1, 1, 1, height*width)  # [B, 3, 1, H*W]

        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch,  height, width, 2), mode='bilinear',
                                   padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    return warped_src_fea


def filter_depth(ref_depth, src_depths, ref_proj, src_projs):
    '''
    :param ref_depth: (1, 1, H, W)
    :param src_depths: (B, 1, H, W)
    :param ref_proj: (1, 4, 4)
    :param src_proj: (B, 4, 4)
    :return: ref_pc: (1, 3, H, W), aligned_pcs: (B, 3, H, W), dist: (B, 1, H, W)
    '''
    # print(ref_proj.shape, src_projs.shape)
    ref_proj_new = ref_proj[:, 0].clone()
    ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
    B,V,C,H,W = src_depths.shape
    ref_pc = generate_points_from_depth(ref_depth, ref_proj_new)
    ref_pc = ref_pc.repeat(V,1,1,1)


    src_depths = src_depths.view(B*V,C,H,W)
    src_projs = src_projs.view(B*V,2,4,4)
    src_proj_new = src_projs[:, 0].clone()
    src_proj_new[:, :3, :4] = torch.matmul(src_projs[:, 1, :3, :3], src_projs[:, 0, :3, :4])
    src_pcs = generate_points_from_depth(src_depths, src_proj_new)

    ref_proj_new = ref_proj_new.repeat(B*V, 1, 1)
    aligned_pcs = homo_warping(src_pcs, src_proj_new, ref_proj_new, ref_depth)

    x_2 = (ref_pc[:, 0] - aligned_pcs[:, 0])**2
    y_2 = (ref_pc[:, 1] - aligned_pcs[:, 1])**2
    z_2 = (ref_pc[:, 2] - aligned_pcs[:, 2])**2
    dist = torch.sqrt(x_2 + y_2 + z_2).unsqueeze(1)

    return ref_pc, aligned_pcs, dist