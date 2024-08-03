from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
import random
# from datasets.preprocess import *
import  cv2

# the DTU dataset preprocessed by Yao Yao (only for training)

def cropaug(crop_x_size,crop_y_size,image,intrinsics):
    h,w,_ = image.shape
    x, y = random.randint(0, w-crop_x_size), random.randint(0, h-crop_y_size)
    image = image[y:y+crop_y_size,x:x+crop_x_size,:]
    intrinsics[0,2] = intrinsics[0,2] - x
    intrinsics[1, 2] = intrinsics[1, 2] - y
    return image, intrinsics,x,y

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths = 192, interval_scale = 1.06, dispmaxfirst = 'first', **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        if self.mode == "train":
            self.listfile = os.path.join(self.datapath, "training_list.txt")
        else:
            self.listfile = os.path.join(self.datapath, "validation_list.txt")
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        # self.inverse_depth = inverse_depth
        # self.origin_size = origin_size
        # self.light_idx = light_idx
        # self.image_scale = image_scale  # use to resize image
        # self.reverse = reverse
        # self.both = both


        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # if len(src_views) < self.nviews - 1:
                    #     print('less ref_view small {}'.format(self.nviews - 1))
                    #     continue
                    if len(src_views) < 7:
                        print('less ref_view small {}'.format(self.nviews - 1))
                        continue
                    # if self.both:
                    #     metas.append((scan, ref_view, src_views, 1))  # add 1, 0 for reverse depth
                    metas.append((scan, ref_view, src_views, 0))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        # if self.image_scale != 1.0:  # origin: 1.0
        #     intrinsics[:2, :] *= self.image_scale

        depth_min = float(lines[11].split()[0])
        # depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_max = float(lines[11].split()[3])
        return intrinsics, extrinsics, depth_min, depth_max

    # def read_img(self, filename):
    #     img = Image.open(filename)
    #     # if self.image_scale != 1.0:
    #     #     w, h = img.size
    #     #     img = img.resize((int(self.image_scale * w), int(self.image_scale * h)))  # origin: 0.25
    #     # scale 0~255 to 0~1
    #     # np_img = np.array(img, dtype=np.float32) / 255. # origin version on 2020/02/20
    #     # return np.array(img, dtype=np.float32) / 255.
    #     return self.center_img(np.array(img, dtype=np.float32))
    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img
    def center_img(self, img):  # this is very important for batch normalization
        img = img.astype(np.float32)
        var = np.var(img, axis=(0, 1), keepdims=True)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return (img - mean) / (np.sqrt(var) + 0.00000001)

    def read_depth(self, filename):
        # read pfm depth file
        depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        # depth_image = scale_image(depth_image, scale=self.image_scale, interpolation='nearest')
        return depth_image
    def final_crop(self, img, depth, intrinsic, mask, crop_h, crop_w, offset_y=None, offset_x=None):
        h, w, _ = img.shape
        if offset_x is None or offset_y is None:
            offset_y = random.randint(0, h - crop_h)
            offset_x = random.randint(0, w - crop_w)

        cropped_image = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w, :]

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        if depth is not None:
            cropped_depth = depth[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_depth = None

        if mask is not None:
            cropped_mask = mask[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_mask = None

        return cropped_image, cropped_depth, output_intrinsics, cropped_mask, offset_y, offset_x
    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        # print('idx: {}, flip_falg {}'.format(idx, flip_flag))
        meta = self.metas[idx]
        scan, ref_view, src_views, flip_flag = meta
        # use only the reference view and first nviews-1 source views
        if self.mode == "finetune":
            src_views_ids = random.sample(src_views, self.nviews - 1)
        else:
            src_views_ids = src_views[:self.nviews - 1]
        view_ids = [ref_view] + src_views_ids

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        depth_ms = {}
        mask_ms = {}
        crop_w = 640
        crop_h = 512
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 000000000
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            # if i == 0:
            #     print('process in {}, {}'.format(idx, img_filename))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            if i == 0:
                depth_name = depth_filename
            # print('debug in dtu_yao', i, depth_filename)
            image = self.read_img(img_filename)

            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(proj_mat_filename)
            # if self.mode == "train":
            #     image, intrinsics,x_,y_ = cropaug(crop_w,crop_h,image,intrinsics)
            imgs.append(image)
            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)
            # proj_mat = extrinsics.copy()
            # proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            # proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                disp_min = 1 / depth_max
                disp_max = 1 / depth_min
                depth_values = np.linspace(disp_min, disp_max, self.ndepths, endpoint=False)
                # depth_values = 1.0 / depth_values
                depth_values = depth_values.astype(np.float32)

                    # depth_values = np.concatenate((depth_values[1::2],depth_values[::-2]),axis=0)


                # mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)
                h, w = depth.shape
                # if self.mode == "train":
                #     depth = depth[y_:y_+crop_h,x_:x_+crop_w]
                #     depth_ms = {
                #         "stage1": cv2.resize(depth, (crop_w // 8, crop_h // 8), interpolation=cv2.INTER_NEAREST),
                #         "stage2": cv2.resize(depth, (crop_w // 4, crop_h // 4), interpolation=cv2.INTER_NEAREST),
                #         "stage3": cv2.resize(depth, (crop_w // 2, crop_h // 2), interpolation=cv2.INTER_NEAREST),
                #         "stage4": depth,
                #     }
                # else:
                depth_ms = {
                    "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
                    "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
                    "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
                    "stage4": depth,
                }
                # mask = np.array((depth > depth_min+depth_interval) & (depth < depth_min+(self.ndepths-2)*depth_interval), dtype=np.float32)
                mask_ms = {
                    "stage1": np.array((depth_ms["stage1"] >= depth_min) & (depth_ms["stage1"] <= depth_max), dtype=np.float32),
                    "stage2": np.array((depth_ms["stage2"] >= depth_min) & (depth_ms["stage2"] <= depth_max), dtype=np.float32),
                    "stage3": np.array((depth_ms["stage3"] >= depth_min) & (depth_ms["stage3"] <= depth_max), dtype=np.float32),
                    "stage4": np.array((depth_ms["stage4"] >= depth_min) & (depth_ms["stage4"] <= depth_max), dtype=np.float32),
                }
                # mask = np.array((depth >= depth_min) & (depth <= depth_end), dtype=np.float32)
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        proj_matrices = np.stack(proj_matrices)
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 16.0
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 8.0
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 1

        proj_matrices_ms = {
            "stage0": stage0_pjmats,
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats,
            "stage4": stage4_pjmats,
        }

        # if (flip_flag and self.both) or (self.reverse and not self.both):
        #     depth_values = np.array([depth_values[len(depth_values) - i - 1] for i in range(len(depth_values))])

        # print('img:{}, depth:{}, depth_values:{}, mask:{}, depth_interval:{}'.format(imgs.shape, depth.shape, depth_values.shape,mask.shape,depth_interval))
        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,  # generate depth index
                "mask": mask_ms,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
