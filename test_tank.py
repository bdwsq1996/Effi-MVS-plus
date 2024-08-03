import argparse, os, time, sys, gc, cv2
import torch
import misc.fusion as fusion
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn.functional as F
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import read_pfm, save_pfm
from plyfile import PlyData, PlyElement
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import signal

# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False)
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='mvsnet', help='select model')

parser.add_argument('--dataset', default='dtu_yao_eval', help='select dataset')
parser.add_argument('--testpath', help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', help='testing scene list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=96, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--savedir', default='./outputs_cas', help='output dir')
parser.add_argument('--outdir', default='./outputs_cas', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')

parser.add_argument('--ndepths', type=str, default="96,8,8", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')

parser.add_argument('--maskupsample', type=str, default="last",  help='maskupsample')
parser.add_argument('--hiddenstate', type=str, default="init",  help='hiddenstate')
parser.add_argument('--GRUiters', type=str, default="3,3,3",  help='iters')
parser.add_argument('--CostNum', type=int, default=3,  help='CostNum')
parser.add_argument('--initloss', type=str, default='initloss',  help='initloss')
parser.add_argument('--dispmaxfirst', type=str, default='last',  help='testviews')

parser.add_argument('--interval_scale', type=float, default=0.53, help='the depth interval scale')
parser.add_argument('--num_view', type=int, default=5, help='num of view')
# parser.add_argument('--max_h', type=int, default=864, help='testing max h')
# parser.add_argument('--max_w', type=int, default=1152, help='testing max w')
parser.add_argument('--max_h', type=int, default=1184, help='testing max h')
parser.add_argument('--max_w', type=int, default=1600, help='testing max w')
parser.add_argument('--fix_res', action='store_true', help='scene all using same res')

parser.add_argument('--num_worker', type=int, default=1, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--filter_method', type=str, default='normal', choices=["gipuma", "normal"], help="filter method")

#filter
parser.add_argument('--conf', type=float, default=0, help='prob confidence')
parser.add_argument('--data_type', type=str, default='dtu', help='prob confidence')
parser.add_argument('--thres_view', type=int, default=2, help='threshold of num view')


#filter by gimupa
parser.add_argument('--fusibile_exe_path', type=str, default='../fusibile/fusibile')
parser.add_argument('--prob_threshold', type=float, default='0.5')
parser.add_argument('--disp_threshold', type=float, default='0.25')
parser.add_argument('--num_consistent', type=float, default='4')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)
nviews_pair = args.num_view


# num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
num_stage = 3
def prepare_img(hr_img):
    # w1600-h1200-> 800-600 ; crop -> 640, 512; downsample 1/4 -> 160, 128

    # downsample
    h, w = hr_img.shape
    hr_img_ds = cv2.resize(hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
    # crop
    h, w = hr_img_ds.shape
    target_h, target_w = 512, 640
    start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
    hr_img_crop = hr_img_ds[start_h: start_h + target_h, start_w: start_w + target_w]

    # #downsample
    # lr_img = cv2.resize(hr_img_crop, (target_w//4, target_h//4), interpolation=cv2.INTER_NEAREST)

    return hr_img_crop
def read_mask_hr(filename):
    img = Image.open(filename)
    np_img = np.array(img, dtype=np.float32)
    np_img = (np_img > 10).astype(np.float32)
    np_img = prepare_img(np_img)

    h, w = np_img.shape

    return (np_img > 0.5)
# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] *= 2
    depth_min = float(lines[11].split()[3])
    depth_max = float(lines[11].split()[2])
    if depth_max>425:
        depth_max = 935
        depth_min = 425
    return intrinsics, extrinsics, depth_max, depth_min


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            view_score = f.readline().rstrip()
            src_views = [int(x) for x in view_score.split()[1::2]]
            score = [float(x) for x in view_score.split()[2::2]]
            selected_views = []
            for src_view in src_views:
                if src_view != ref_view:
                    selected_views.append(src_view)

            if len(src_views) > 0:
                data.append((ref_view, selected_views))
                # data.append((ref_view, src_views))
    return data

def write_cam(file, cam, depth_max, depth_min):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(depth_max) + ' ' + str(depth_min) + '\n')

    f.close()

def save_depth(testlist, num_view=11):

    for scene in testlist:
        save_scene_depth([scene], num_view=num_view)

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist, num_view=11):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)

    input_view = args.num_view

    test_dataset = MVSDataset(args.testpath, input_view, args.numdepth, img_wh=(1920,1056), scan=testlist)

    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    # model

    model = Effi_MVS_plus(args, refine=False)

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    # state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=False)
    # model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):

            if args.dispmaxfirst == 'first':
                depth_max = 1. / sample["depth_values"][:, -1]
                depth_min = 1. / sample["depth_values"][:, 0]
            else:
                depth_max = 1. / sample["depth_values"][:, 0]
                depth_min = 1. / sample["depth_values"][:, -1]


            sample_cuda = tocuda(sample)
            if "scale" in sample_cuda.keys():
                scale = sample_cuda["scale"]
                scale = tensor2numpy(scale)
            else:
                scale = 1
            depth_max = tensor2numpy(depth_max) / scale
            depth_min = tensor2numpy(depth_min) / scale

            torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

            end_time = time.time()
            del sample_cuda
            print(torch.cuda.max_memory_allocated() / 1024 ** 2)
            torch.cuda.empty_cache()


            outputs = tensor2numpy(outputs)

            if batch_idx == 10:
                torch.cuda.empty_cache()
            filenames = sample["filename"]

            # depth_est = [outputs["depth"][0]/scale, outputs["depth"][-1]/scale]




            cams = sample["proj_matrices"]["stage4"].numpy()

            cams[:,:,0,:3,3] /= scale

            imgs = sample["imgs"].numpy()
            # print('Iter {}/{}, Time:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time))
            print('Iter {}/{}, Time:{} Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))
            depth_out_num = -1
            # # save depth maps and confidence maps
            for filename, cam, img, depth_est, photometric_confidence, depth_max_, depth_min_ in zip(filenames, cams, imgs, outputs["depth"][depth_out_num]/scale, outputs["photometric_confidence"], depth_max, depth_min):


                img = img[0]  #ref view


                _, h, w = img.shape


                cam = cam[0]  #ref cam
                # cam = cam  # ref cam
                depth_filename = os.path.join(args.outdir, filename.format('depth_est', '.pfm'))

                #

                print(depth_est.shape,depth_max_,depth_min_)

                #
                confidence_filename = os.path.join(args.outdir, filename.format('confidence', '.pfm'))
                cam_filename = os.path.join(args.outdir, filename.format('cams', '_cam.txt'))
                img_filename = os.path.join(args.outdir, filename.format('images', '.jpg'))
                ply_filename = os.path.join(args.outdir, filename.format('ply_local', '.ply'))
                os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
                os.makedirs(ply_filename.rsplit('/', 1)[0], exist_ok=True)
                #save depth maps
                save_pfm(depth_filename, depth_est)
                #save confidence maps
                save_pfm(confidence_filename, photometric_confidence)
                #save cams, img
                write_cam(cam_filename, cam, depth_max_, depth_min_)
                img = np.clip(np.transpose(img, (1, 2, 0)) * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # print(img_bgr.shape)
                cv2.imwrite(img_filename, img_bgr)


    torch.cuda.empty_cache()
    gc.collect()

class TTDataset(Dataset):
    def __init__(self, pair_folder, scan_folder, n_src_views=10):
        super(TTDataset, self).__init__()
        pair_file = os.path.join(pair_folder, "pair_new.txt")
        if not os.path.exists(pair_file):
            pair_file = pair_file.replace("/ETH3D/eth3d_high_res_test/", "/ETH3D_DATA/")
            if not os.path.exists(pair_file):
                pair_file = os.path.join(pair_folder, "pair.txt")
        self.scan_folder = scan_folder
        self.pair_data = read_pair_file(pair_file)
        # print(self.pair_data)
        self.n_src_views = n_src_views

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[:self.n_src_views]
        # print(id_srcs)
        ref_intrinsics, ref_extrinsics, depth_max, depth_min = read_camera_parameters(
            os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(id_ref)))
        ref_cam = np.zeros((2, 4, 4), dtype=np.float32)
        ref_cam[0] = ref_extrinsics
        ref_cam[1, :3, :3] = ref_intrinsics
        ref_cam[1, 3, 3] = 1.0
        # load the reference image
        ref_img = read_img(os.path.join(self.scan_folder, 'images/{:0>8}.jpg'.format(id_ref)))
        ref_img = ref_img.transpose([2, 0, 1])
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(id_ref)))[0]
        ref_depth_est = np.array(ref_depth_est, dtype=np.float32)
        # load the photometric mask of the reference view
        # confidence = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref)))[0]
        # conf_path = os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref))
        confidence = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(id_ref)))[0]
        confidence = np.array(confidence, dtype=np.float32)

        src_depths, src_confs, src_cams = [], [], []
        for ids in id_srcs:
            if not os.path.exists(os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(ids))):
                continue
            src_intrinsics, src_extrinsics, _, _ = read_camera_parameters(
                os.path.join(self.scan_folder, 'cams/{:0>8}_cam.txt'.format(ids)))
            src_proj = np.zeros((2, 4, 4), dtype=np.float32)
            src_proj[0] = src_extrinsics
            src_proj[1, :3, :3] = src_intrinsics
            src_proj[1, 3, 3] = 1.0
            src_cams.append(src_proj)
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(self.scan_folder, 'depth_est/{:0>8}.pfm'.format(ids)))[0]
            src_depths.append(np.array(src_depth_est, dtype=np.float32))
            # src_conf = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(ids)))[0]
            confidence_src = read_pfm(os.path.join(self.scan_folder, 'confidence/{:0>8}.pfm'.format(ids)))[0]
            confidence_src = np.array(confidence_src, dtype=np.float32)
            src_confs.append(confidence_src)
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_confs = np.stack(src_confs, axis=0)
        src_cams = np.stack(src_cams, axis=0)
        # print(ref_depth_est.mean(), ref_depth_est.max(), ref_depth_est.min())
        return {"ref_depth": np.expand_dims(ref_depth_est, axis=0),
                "ref_cam": ref_cam,
                "ref_conf": confidence,  # np.expand_dims(confidence, axis=0),
                "src_depths": src_depths,
                "src_cams": src_cams,
                "src_confs": src_confs,
                "ref_img": ref_img,
                "ref_id": id_ref,
                "depth_max": depth_max,
                "depth_min": depth_min}

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected = np.where(K_xyz_reprojected == 0, 1e-5, K_xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    xy_reprojected = np.clip(xy_reprojected, -1e8, 1e8)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency_tank(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,dh_pixel_dist_num):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)

    masks = []
    for i in range(dh_pixel_dist_num[0], 11):
        mask = np.logical_and(dist < i / dh_pixel_dist_num[1], depth_diff < dh_pixel_dist_num[2])

        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


def dynamic_filter_depth(pair_folder, scan_folder, plyfilename, filter_dixt,relative=False):
    print(filter_dixt)
    tt_dataset = TTDataset(pair_folder, scan_folder, n_src_views=10)
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(tt_dataset, batch_size=1, shuffle=False, sampler=sampler, num_workers=2,
                               pin_memory=True, drop_last=False)
    views = {}
    point_list = []
    color_list = []
    prob_threshold = filter_dixt["prob_threshold"]
    # prob_threshold = [float(p) for p in prob_threshold.split(',')]
    for batch_idx, sample_np in enumerate(tt_dataloader):
        num_src_views = sample_np['src_depths'].shape[1]
        dy_range = num_src_views + 1  # 10
        sample = tocuda(sample_np)
        B,H,W = sample['ref_conf'].shape
        h,w = sample['ref_depth'].shape[-2:]
        # print(h,w)
        sample['ref_conf'] = F.interpolate(sample['ref_conf'].unsqueeze(1), size=[h,w], mode='nearest')
        prob_mask = sample['ref_conf'] > prob_threshold
        prob_mask = prob_mask.squeeze(1)

        n, v, _, h, w = sample['src_depths'].size()
        if v<(filter_dixt["dh_view_num"]+1):
            continue

        ref_depth = sample['ref_depth']  # [n 1 h w ]
        depth_max, depth_min = sample['depth_max'], sample['depth_min']
        depth_mask = torch.logical_and((ref_depth > depth_min), (ref_depth < depth_max))
        # print(ref_depth.mean(), ref_depth.max(), ref_depth.min())
        device = ref_depth.device
        reproj_xyd, ref_idx_cam, src2ref_idx_cam = fusion.get_reproj_dynamic(
            *[sample[attr] for attr in ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']])
        # reproj_xyd   nv 3 h w
        dh_view_num = filter_dixt["dh_view_num"]
        dh_pixel_dist_num = [dh_view_num, filter_dixt["dist_filter"], filter_dixt["depth_filter"]]
        vis_masks, vis_mask = fusion.vis_filter_dynamic(sample['ref_depth'], reproj_xyd, ref_idx_cam, src2ref_idx_cam, dist_base=filter_dixt["dist_filter"],  # 4 1300
                                                        rel_diff_base=filter_dixt["depth_filter"], thres_view=dh_view_num, relative=relative)


        # mask reproj_depth
        reproj_depth = reproj_xyd[:, :, -1]  # [1 v h w]
        # print(reproj_depth.shape, vis_mask.shape)
        if vis_mask.shape[2] == 0:
            print("continue")
            continue
        reproj_depth[~vis_mask.squeeze(2)] = 0  # [n v h w ]
        geo_mask_sums = vis_masks.sum(dim=1)  # 0~v
        geo_mask_sum = vis_mask.sum(dim=1)
        depth_est_averaged = (torch.sum(reproj_depth, dim=1, keepdim=True) + ref_depth) / (
                    geo_mask_sum + 1)  # [1,1,h,w]


        geo_mask = geo_mask_sum >= dy_range  # all zero
        for i in range(dh_view_num, dy_range):
            geo_mask = torch.logical_or(geo_mask, geo_mask_sums[:, i - dh_view_num] >= i)

        mask = fusion.bin_op_reduce([prob_mask, geo_mask], torch.min)
        idx_img = fusion.get_pixel_grids(*depth_est_averaged.size()[-2:]).unsqueeze(0)
        idx_cam = fusion.idx_img2cam(idx_img, depth_est_averaged, sample['ref_cam'])
        points = fusion.idx_cam2world(idx_cam, sample['ref_cam'])[..., :3, 0].permute(0, 3, 1, 2)

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool)

        ref_img = sample_np['ref_img'].data.numpy()
        # print(ref_img.shape, mask_np.shape)
        # ref_img = ref_img[:, :, 1::2, 1::2]

        # mask = torch.logical_and(mask, depth_mask)
        for i in range(points_np.shape[0]):
            # print(np.sum(np.isnan(points_np[i])))
            p_f_list = [points_np[i, k][mask_np[i, 0]] for k in range(3)]
            p_f = np.stack(p_f_list, -1)
            c_f_list = [ref_img[i, k][mask_np[i, 0]] for k in range(3)]
            c_f = np.stack(c_f_list, -1) * 255

            ref_id = str(sample_np['ref_id'][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))

            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
            # print(prob_mask.shape, geo_mask.shape, mask.shape, points_np.shape)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_id)), prob_mask[i].cpu().data.numpy())
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_id)), geo_mask[i, 0].cpu().data.numpy())
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_id)), mask[i, 0].cpu().data.numpy())

            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}/{}".format(scan_folder, int(ref_id),
                                                                                           prob_mask[
                                                                                               i].float().mean().item(),
                                                                                           geo_mask[
                                                                                               i].float().mean().item(),
                                                                                           depth_mask[
                                                                                               i].float().mean().item(),
                                                                                           mask[
                                                                                               i].float().mean().item()))

    print('Write combined PCD')
    point_list = [v[0] for key, v in views.items()]
    color_list = [v[1] for key, v in views.items()]
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]
    vertexs = np.array([tuple(v) for v in p_all], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in c_all], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    # vertex_all = np.empty(len(vertexs), vertexs.dtype.descr)
    # for prop in vertexs.dtype.names:
    #     vertex_all[prop] = vertexs[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)





def init_worker():
    '''
    Catch Ctrl+C signal to termiante workers
    '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)



if __name__ == '__main__':

    # step1. save all the depth maps and the masks in outputs directory
    #
    if args.testlist != "all":
        with open(args.testlist) as f:
            content = f.readlines()
            testlist = [line.rstrip() for line in content]
    else:
        #for tanks & temples or eth3d or colmap
        testlist = [e for e in os.listdir(args.testpath) if os.path.isdir(os.path.join(args.testpath, e))] \
            if not args.testpath_single_scene else [os.path.basename(args.testpath_single_scene)]

    # step1. save all the depth maps and the masks in outputs directory
    #

    filter_dict_tank8_test = {
        "Family":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "Francis":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "Horse":{"views":11,"dh_view_num":2,"dist_filter":4,"depth_filter":6000, "prob_threshold": 0.3},
        "Lighthouse":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "M60":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "Panther":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "Playground":{"views":11,"dh_view_num":2,"dist_filter":2,"depth_filter":6000, "prob_threshold": 0.5},
        "Train": {"views": 11, "dh_view_num": 2, "dist_filter": 2, "depth_filter": 6000, "prob_threshold": 0.5},
        "Auditorium": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 500, "prob_threshold": 0.3},
        "Ballroom": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 1600, "prob_threshold": 0.3},
        "Courtroom": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 1600, "prob_threshold": 0.3},
        "Museum": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 1600, "prob_threshold": 0.3},
        "Palace": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 1600, "prob_threshold": 0.3},
        "Temple": {"views": 11, "dh_view_num": 2, "dist_filter": 1, "depth_filter": 1600, "prob_threshold": 0.3}
    }

    # testlist = ['Family','Francis','Horse','Lighthouse','M60', 'Panther', 'Playground', 'Train', 'Auditorium', 'Ballroom', 'Courtroom','Museum', 'Palace', 'Temple']
    testlist = ['Horse']

    save_depth(testlist)






    path = args.testpath
    plypath = args.savedir
    for scan in testlist:
        if scan in ['Family', 'Francis', 'Horse', 'Lighthouse', 'M60', 'Panther', 'Playground', 'Train']:
            path = args.testpath + 'intermediate/'
        else:
            path = args.testpath + 'advanced/'
        pair_folder = os.path.join(path, scan)
        scan_folder = os.path.join(args.outdir, scan)
        out_folder = os.path.join(args.outdir, scan)
        # step2. filter saved depth maps with photometric confidence maps and geometric constraints
        if not os.path.exists(os.path.join(args.savedir, '{}'.format(scan))):
            os.makedirs(os.path.join(args.savedir, '{}'.format(scan)))
        # print(filter_dict_tank8_train[scan])
        dynamic_filter_depth(pair_folder, scan_folder, os.path.join(args.savedir, '{}.ply'.format(scan)), filter_dict_tank8_test[scan], relative=False)

