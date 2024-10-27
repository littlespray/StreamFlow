import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp
import copy
from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


def cd_dotdot(path_or_filename):
    return os.path.abspath(os.path.join(os.path.dirname(path_or_filename), ".."))

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):
        pass

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class SpringSubmission(FlowDataset):
    def __init__(self, root=None, input_frames=4):
        super().__init__(sparse=False)
        self.nframes = input_frames
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        for scene in os.listdir(root):
            for cam in ["left", 'right']:
                img_filenames = sorted(glob(osp.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(img_filenames)
                # forward
                i = 0
                while True:
                    if i + self.nframes <= len(img_filenames):
                        imgs = img_filenames[i:i+self.nframes]
                        self.extra_info.append([scene, 'FW', cam, [j+1 for j in range(i, i+self.nframes)]])
                    else:
                        imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                        ids = [-1 if j < i else j+1 for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                        self.extra_info.append([scene, 'FW', cam, ids])
                    self.image_list.append(imgs)
                    # print('FW', self.image_list[-1], self.extra_info[-1])
                    if i + self.nframes >= len(img_filenames): break
                    else: i += self.nframes-1

                # backward
                img_filenames = img_filenames[::-1]
                i = 0
                while True:
                    if i + self.nframes <= len(img_filenames):
                        imgs = img_filenames[i:i+self.nframes]
                        self.extra_info.append([scene, 'BW', cam, [len(img_filenames) - j for j in range(i, i+self.nframes)]])
                    else:
                        imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                        ids = [-1 if j < i else len(img_filenames) - j for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                        self.extra_info.append([scene, 'BW', cam, ids])
                    self.image_list.append(imgs)
                    # print('BW', self.image_list[-1], self.extra_info[-1])

                    if i + self.nframes >= len(img_filenames): break
                    else: i += self.nframes-1
    
    def __getitem__(self, index):
        index = index % len(self.image_list)
        
        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
    
        return imgs, self.extra_info[index]

class SpringEval(FlowDataset):
    def __init__(self, aug_params=None, root=None, input_frames=4, forward_warp=False, subsample_groundtruth=True, split=False):
        super().__init__(aug_params=aug_params, sparse=False)
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []
        self.nframes = input_frames
        all_scenes = []
        self.subsample_groundtruth = subsample_groundtruth
        print("[whether subsample groundtruth: {}]".format(self.subsample_groundtruth))
        self.forward_warp = forward_warp
        print("[return forward warped flow {}]".format(self.forward_warp))


        for scene in sorted(os.listdir(root)):
            if not scene == '0041':
                continue
            print('===============train-val split===============')
            all_scenes.append(scene)
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(images)
                # forward
                _future_flow_list = []
                for i in range(1, len_image):
                    _future_flow_list.append(os.path.join(root, scene, f"flow_FW_{cam}", f"flow_FW_{cam}_{i:04d}.flo5"))
                # my modification
                i = 0
                while True:
                    # print('xxxx')
                    if i + self.nframes <= len(images):
                        imgs = images[i:i+self.nframes]
                        flows = _future_flow_list[i:i+self.nframes-1]
                        self.extra_info += [[cam+'FW', [j for j in range(i, i+self.nframes)]]]
                    else:
                        imgs = images[len(images)-self.nframes:len(images)]
                        flows = _future_flow_list[len(_future_flow_list)-self.nframes+1:len(_future_flow_list)]
                        ids = [-1 if j < i else j for j in range(len(images)-self.nframes, len(images))]
                        self.extra_info += [[cam+'FW', ids]]
                    self.image_list.append(imgs)
                    self.flow_list.append(flows)
                    if i + self.nframes >= len(images): break
                    else: i += self.nframes-1
                # backward
                images = images[::-1]
                _past_flow_list = []
                for i in range(len_image, 1, -1):
                    _past_flow_list.append(os.path.join(root, scene, f"flow_BW_{cam}", f"flow_BW_{cam}_{i:04d}.flo5"))
                # my modification
                i = 0
                while True:
                    if i + self.nframes <= len(images):
                        imgs = images[i:i+self.nframes]
                        flows = _past_flow_list[i:i+self.nframes-1]
                        self.extra_info += [[cam+'BW', [j for j in range(i, i+self.nframes)]]]
                    else:
                        imgs = images[len(images)-self.nframes:len(images)]
                        flows = _past_flow_list[len(_past_flow_list)-self.nframes+1:len(_past_flow_list)]
                        ids = [-1 if j < i else j for j in range(len(images)-self.nframes, len(images))]
                        self.extra_info += [[cam+'BW', ids]]
                    self.image_list.append(imgs)
                    self.flow_list.append(flows)
                    if i + self.nframes >= len(images): break
                    else: i += self.nframes-1
                

    
    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None
        # print(self.image_list[index])
        # print(self.flow_list[index])
        
        flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]
        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        flows = [np.array(flow).astype(np.float32) for flow in flows]
        # print(flows[0].shape, 'flow-before')
        if self.subsample_groundtruth:
            flows = [flow[::2, ::2] for flow in flows]

        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        # print(imgs[0].shape)

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        
        # print(imgs[0].shape, 'img1')
        # print(flows[0].shape, 'flow-after')

        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        o_valids = False
        # if not self.forward_warp:
        return imgs, flows, valids, self.extra_info[index]
#         else:
#             new_size = (flows[0].shape[1] // 8, flows[0].shape[2] // 8)
#             if not o_valids:
#                 downsampled_flow = [F.interpolate(flow.unsqueeze(0), size=new_size, mode='bilinear', align_corners=True).squeeze(0) / 8 for flow in flows[:-1]]
#                 forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] + [forward_interpolate(flow) for flow in downsampled_flow]
#             else:
#                 forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] * len(flows)
#             return imgs, flows, valids, forward_warped_flow, self.extra_info[index]





class Spring(FlowDataset):
    def __init__(self, aug_params=None, root=None, input_frames=4, forward_warp=False, subsample_groundtruth=False, split=False):
        super().__init__(aug_params=aug_params, sparse=False)
        if not os.path.exists(root):
            raise ValueError(f"Spring train directory does not exist: {root}")

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []
        self.nframes = input_frames
        all_scenes = []
        self.subsample_groundtruth = subsample_groundtruth
        print("[whether subsample groundtruth: {}]".format(self.subsample_groundtruth))
        self.forward_warp = forward_warp
        print("[return forward warped flow {}]".format(self.forward_warp))


        for scene in sorted(os.listdir(root)):
            if split and scene == '0041':
                print('===============train-val split===============')
                continue
            all_scenes.append(scene)
            for cam in ["left", "right"]:
                images = sorted(glob(os.path.join(root, scene, f"frame_{cam}", '*.png')))
                len_image = len(images)
                # forward
                _future_flow_list = []
                for i in range(1, len_image):
                    _future_flow_list.append(os.path.join(root, scene, f"flow_FW_{cam}", f"flow_FW_{cam}_{i:04d}.flo5"))
                # my modification
                for idx_image in range(0, len_image - input_frames + 1):
                    self.image_list.append(images[idx_image:idx_image + input_frames])
                    self.flow_list.append(_future_flow_list[idx_image:idx_image + input_frames - 1])
                    self.extra_info += [['0', 0]]
                # backward
                images = images[::-1]
                _past_flow_list = []
                for i in range(len_image, 1, -1):
                    _past_flow_list.append(os.path.join(root, scene, f"flow_BW_{cam}", f"flow_BW_{cam}_{i:04d}.flo5"))
                for idx_image in range(0, len_image - input_frames + 1):
                    self.image_list.append(images[idx_image:idx_image + input_frames])
                    self.flow_list.append(_past_flow_list[idx_image:idx_image + input_frames - 1])
                    self.has_gt_list.append([True] * (input_frames - 1))
                    self.extra_info += [['0', 0]]
    
    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None
        
        flows = [frame_utils.read_gen(path) for path in self.flow_list[index]]
        imgs = [frame_utils.read_gen(path) for path in self.image_list[index]]
        flows = [np.array(flow).astype(np.float32) for flow in flows]
        # print(flows[0].shape, 'flow-before')
        if self.subsample_groundtruth:
            flows = [flow[::2, ::2] for flow in flows]

        imgs = [np.array(img).astype(np.uint8) for img in imgs]
        # print(imgs[0].shape)

        # grayscale images
        if len(imgs[0].shape) == 2:
            imgs = [np.tile(img[..., None], (1, 1, 3)) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        if self.augmentor is not None:
            imgs, flows = self.augmentor(imgs, flows)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        
        # print(imgs[0].shape, 'img1')
        # print(flows[0].shape, 'flow-after')

        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        o_valids = False

        if not self.forward_warp:
            return imgs, flows, valids, ['0', [0]*self.nframes]
        else:
            new_size = (flows[0].shape[1] // 8, flows[0].shape[2] // 8)
            if not o_valids:
                downsampled_flow = [F.interpolate(flow.unsqueeze(0), size=new_size, mode='bilinear', align_corners=True).squeeze(0) / 8 for flow in flows[:-1]]
                forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] + [forward_interpolate(flow) for flow in downsampled_flow]
            else:
                forward_warped_flow = [torch.zeros(2, new_size[0], new_size[1])] * len(flows)
            return imgs, flows, valids, forward_warped_flow, ['0', [0]*self.nframes]



class FlyingThings3DMultiFrame(FlowDataset):
    def __init__(self, aug_params=None, root='/datasets/flow/flyingthings3d', split='training', dstype='frames_cleanpass', nframes=4):
        super().__init__(aug_params)
        self.nframes = nframes
        self.image_list = []
        self.flow_list = []

        for cam in ['left']:
            image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])
            for direction in ['into_future', 'into_past']:
                flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
                flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])
                for idir, fdir in zip(image_dirs, flow_dirs):
                    images = sorted(glob(osp.join(idir, '*.png')))
                    flows = sorted(glob(osp.join(fdir, '*.pfm')))
                    assert len(images) >= self.nframes

                    if direction == 'into_future':
                        i, N = 0, len(images)
                        while True:
                            if i + self.nframes <= N:
                                imgs, flos = images[i:i+self.nframes], flows[i:i+self.nframes-1]
                            else:
                                imgs, flos = images[N-self.nframes:N], flows[N-self.nframes:N-1]
                            self.image_list += [imgs]
                            self.flow_list += [flos]
                            self.extra_info += [['0', 0]]
                            if i + self.nframes >= N: break
                            else: i += 1
                        # print('origin')
                        # print(self.image_list[-1])
                        # print(self.flow_list[-1])
                    elif direction == 'into_past':
                        i, N = 0, len(images)
                        while True:
                            if i + self.nframes <= N:
                                imgs, flos = images[i:i+self.nframes], flows[i:i+self.nframes]
                            else:
                                imgs, flos = images[N-self.nframes:N], flows[N-self.nframes:N]
                            self.image_list += [imgs[::-1]]
                            self.flow_list += [flos[::-1][:-1]]
                            self.extra_info += [['0', 0]]
                            if i + self.nframes >= N: break
                            else: i += 1
                        # print('reverse')
                        # print(self.image_list[-1])
                        # print(self.flow_list[-1])

        # print(len(self.image_list))
        # print(len(self.flow_list))
        # print(self.image_list[:10])
        # print(self.flow_list[:10])

        # print(self.image_list[-10:])
        # print(self.flow_list[-9:])

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]

        # print('Here YES!!')

        for i in range(len(imgs)):
            if len(imgs[i].shape) == 2: imgs[i] = np.tile(imgs[i][...,None], (1, 1, 3))
            else: imgs[i] = imgs[i][..., :3]
            # print(img.shape)
        
        if self.augmentor is not None:
            # print('YES!!')
            imgs, flows = self.augmentor(imgs, flows)
        # for img in imgs:
        #     print(img.shape)

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]

        return imgs, flows, valids, ['0', [0]*self.nframes]






class SintelMultiframeEval_stride1(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/backup/data/sintel', dstype='clean', nframes=4, occ_map=False):
        super().__init__(aug_params)
        self.nframes = nframes
        self.is_test = True if split == 'test' else False
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion  = occ_map
        if self.occlusion: self.occ_list = []

        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(image_root, "*/*.png")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))
        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            if not self.is_test: 
                flo_filenames = [x for x in all_flo_filenames if base_folder in x]
                occ_filenames = [x for x in all_occ_filenames if base_folder in x]
            assert len(img_filenames) >= self.nframes

            i = 0
            # print(len(flo_filenames))
            while True:
                if i + self.nframes < len(img_filenames):
                    imgs = img_filenames[i:i+self.nframes]
                    if not self.is_test: 
                        flows = flo_filenames[i:i+self.nframes-1]
                        occs = occ_filenames[i:i+self.nframes-1]
                    self.extra_info += [[base_folder, [i] + [-1] * (self.nframes-1)]]
                else:
                    imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                    if not self.is_test: 
                        flows = flo_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]
                        occs = occ_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]


                    ids = [j for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                    self.extra_info += [[base_folder, ids]]

                self.image_list += [imgs]
                if not self.is_test: 
                    self.flow_list += [flows]
                    if occ_map: self.occ_list += [occs]

                if i + self.nframes >= len(img_filenames): break
                else: i += 1
        # print(self.flow_list[:50])
        # print(self.extra_info[:50])

    def __getitem__(self, index):
        if self.is_test:
            imgs_filenames = self.image_list[index]
            imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
            imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
            return imgs, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        info_index = index % len(self.extra_info)
        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]
        assert len(imgs) == len(flows) + 1
          
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        
        if self.occ_list is not None:
            occ_filenames = self.occ_list[index]
            occs = [torch.from_numpy(np.array(frame_utils.read_gen(occ)).astype(np.uint8) // 255).bool() for occ in occ_filenames]
            return imgs, flows, valids, self.extra_info[info_index], occs
        return imgs, flows, valids, self.extra_info[info_index]




class Bi_FlyingThings3DMultiFrame(FlowDataset):
    def __init__(self, aug_params=None, root='/datasets/flow/flyingthings3d', split='training', dstype='frames_cleanpass', nframes=4):
        super().__init__(aug_params)
        self.nframes = nframes
        self.image_list = []
        self.flow_list = []

        for cam in ['left']:
            image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
            image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

            flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
            flow_dirs1 = sorted([osp.join(f, 'into_future', cam) for f in flow_dirs])
            flow_dirs2 = sorted([osp.join(f, 'into_past', cam) for f in flow_dirs])

            for idir, fdir1, fdir2 in zip(image_dirs, flow_dirs1, flow_dirs2):
                images = sorted(glob(osp.join(idir, '*.png')))
                flows1 = sorted(glob(osp.join(fdir1, '*.pfm')))
                flows2 = sorted(glob(osp.join(fdir2, '*.pfm')))

                assert len(images) >= self.nframes

                i, N = 0, len(images)
                while True:
                    if i + self.nframes <= N:
                        imgs, flos1, flos2 = images[i:i+self.nframes], flows1[i:i+self.nframes-1], flows2[i:i+self.nframes-1]
                    else:
                        imgs, flos1, flos2 = images[N-self.nframes:N], flows1[N-self.nframes:N-1], flows2[i:i+self.nframes-1]
                    
                    self.image_list += [imgs]
                    self.flow_list += [[flos1, flos2]]
                    self.extra_info += [['0', 0]]
                    if i + self.nframes >= N: break
                    else: i += self.nframes - 1

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames1, flows_filenames2 = self.flow_list[index][0], self.flow_list[index][1]

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows1 = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames1]
        flows2 = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames2]

        # print('Here YES!!')

        for i in range(len(imgs)):
            if len(imgs[i].shape) == 2: imgs[i] = np.tile(imgs[i][...,None], (1, 1, 3))
            else: imgs[i] = imgs[i][..., :3]
        
        if self.augmentor is not None:
            imgs, flows1 = self.augmentor(imgs, flows1)
            copied_imgs = copy.deepcopy(imgs)
            _, flows2 = self.augmentor(copied_imgs, flows2)


        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows1 = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows1]
        flows2 = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows2]

        valids1 = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows1]
        valids2 = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows2]


        return imgs, flows1, flows2, valids1, valids2, ['0', [0]*self.nframes]



class SintelMultiframe(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/datasets/flow/sintel', dstype='clean',
                occlusion=False, segmentation=False, nframes=4, reverse=False):
        super().__init__(aug_params)
        self.nframes = nframes
        self.reverse = reverse
        self.is_test = True if split == 'test' else False
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)

        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(image_root, "*/*.png")))

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))

        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            if not self.is_test: flo_filenames = [x for x in all_flo_filenames if base_folder in x]
            assert len(img_filenames) >= self.nframes

            i = 0
            # print(len(flo_filenames))
            while True:
                if i + self.nframes <= len(img_filenames):
                    imgs = img_filenames[i:i+self.nframes]
                    if not self.is_test: flows = flo_filenames[i:i+self.nframes-1]
                    self.extra_info += [[base_folder, [j for j in range(i, i+self.nframes)]]]
                else:
                    imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                    if not self.is_test: flows = flo_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]

                    ids = [-1 if j < i else j for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                    self.extra_info += [[base_folder, ids]]
                    # print(self.extra_info[-1][-1], 'bubububu')

                self.image_list += [imgs]
                if not self.is_test: self.flow_list += [flows]


                if i + self.nframes >= len(img_filenames): break
                else: i += 1
                # else: i += self.nframes-1



    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        info_index = index % len(self.extra_info)
        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]
        # print(imgs_filenames)
        # print(flows_filenames)

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]
        assert len(imgs) == len(flows) + 1

        # grayscale images

        # for img in imgs:
        #     if len(img.shape) == 2: img = np.tile(img[...,None], (1, 1, 3))
        #     else: img = img[..., :3]
        # # print('Here YES!!')
        if self.augmentor is not None:
            imgs, flows = self.augmentor(imgs, flows)
            
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]


        return imgs, flows, valids, self.extra_info[info_index]


# begin modified
class KITTIMultiFrame(FlowDataset):
    def __init__(self, multi_root='/sunshangkun/multi-kitti/', split='training', aug_params=None, nframes=3):
        super().__init__(aug_params=aug_params, sparse=True)
        self.image_list = []
        self.flow_list = []
        self.is_gt_list = []
        self.extra_info = []
        self.nframes = nframes

        images_root_2015 = osp.join(multi_root, split, 'image_2')
        flow_root_2015 = osp.join(multi_root, split, 'flow_occ')

        for idx_list in range(200):
            # each sequence
            for start_i in range(9, 9-nframes+2, -1): # 10-nframes+2, ..., 10        (12 - 5)
                # print(start_i)
                imgs, flows, gts = [], [], []
                frame_names = []

                for i in range(start_i, start_i+nframes):
                    img = images_root_2015+"/000{:03}_{:02}.png".format(idx_list, i)
                    imgs.append(img)
                    frame_names.append(img.split('/')[-1])
                    flow = flow_root_2015+"/000{:03}_10.png".format(idx_list)
                    gt = True if i == 10 else False
                    flows.append(flow)
                    gts.append(gt)
                # print(imgs, flows)
                self.image_list.append(imgs)
                self.flow_list.append(flows[:-1])
                self.is_gt_list.append(gts[:-1])
                self.extra_info.append(frame_names)


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.is_gt_list = v * self.is_gt_list
        return self

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]
        is_gts = self.is_gt_list[index]
        # print(is_gts)
        # print(imgs_filenames)
        # print(flows_filenames)
        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]


        flows, valids = [], []

        for i in range(len(is_gts)):
            flow, valid = frame_utils.readFlowKITTI(flows_filenames[i])
            if not is_gts[i]:
                valid = valid * 0.0
            flows.append(np.array(flow).astype(np.float32))
            valids.append(valid)
        

        if self.augmentor is not None:
            imgs, flows, valids = self.augmentor(imgs, flows, valids)
        # print(len(flows), len(valids), len(imgs))
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        for i in range(len(flows)):
            flows[i] = torch.from_numpy(flows[i]).permute(2, 0, 1).float()
            valids[i] = torch.from_numpy(valids[i]).float()
            if not is_gts[i]: valids[i] = valids[i] * 0.0

        # print(imgs[0].shape, flows[0].shape, valids[0].shape, valids[0].dtype)
        # print(flows)
        return imgs, flows, valids, ['0', [0]*self.nframes]



# begin modified
class KITTIMultiFrame_T4(FlowDataset):
    def __init__(self, multi_root='/sunshangkun/multi-kitti/', split='training', aug_params=None, nframes=3):
        super().__init__(aug_params=aug_params, sparse=True)
        self.image_list = []
        self.flow_list = []
        self.is_gt_list = []
        self.extra_info = []
        self.nframes = nframes

        images_root_2015 = osp.join(multi_root, split, 'image_2')
        flow_root_2015 = osp.join(multi_root, split, 'flow_occ')
        # [9,10,11,12]
        for idx_list in range(200):
            # each sequence
            start_i = 9
            # print(start_i)
            imgs, flows, gts = [], [], []
            frame_names = []

            for i in range(start_i, start_i+nframes):
                img = images_root_2015+"/000{:03}_{:02}.png".format(idx_list, i)
                imgs.append(img)
                frame_names.append(img.split('/')[-1])
                flow = flow_root_2015+"/000{:03}_10.png".format(idx_list)
                gt = True if i == 10 else False
                flows.append(flow)
                gts.append(gt)
            # print(imgs, flows)
            self.image_list.append(imgs)
            self.flow_list.append(flows[:-1])
            self.is_gt_list.append(gts[:-1])
            self.extra_info.append(frame_names)


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.is_gt_list = v * self.is_gt_list
        return self

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]
        is_gts = self.is_gt_list[index]
        # print(is_gts)
        # print(imgs_filenames)
        # print(flows_filenames)
        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]


        flows, valids = [], []

        for i in range(len(is_gts)):
            flow, valid = frame_utils.readFlowKITTI(flows_filenames[i])
            if not is_gts[i]:
                valid = valid * 0.0
            flows.append(np.array(flow).astype(np.float32))
            valids.append(valid)
        

        if self.augmentor is not None:
            imgs, flows, valids = self.augmentor(imgs, flows, valids)
        # print(len(flows), len(valids), len(imgs))
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        for i in range(len(flows)):
            flows[i] = torch.from_numpy(flows[i]).permute(2, 0, 1).float()
            valids[i] = torch.from_numpy(valids[i]).float()
            if not is_gts[i]: valids[i] = valids[i] * 0.0


        # print(flows)
        return imgs, flows, valids, ['0', [0]*self.nframes]



# class KITTIMultiFrameCenter(FlowDataset):
#     def __init__(self, multi_root='/sunshangkun/multi-kitti/', split='training', aug_params=None, nframes=3):
#         super().__init__(aug_params=aug_params, sparse=True)
#         self.image_list = []
#         self.flow_list = []
#         self.is_gt_list = []
#         self.extra_info = []
#         self.nframes = nframes

#         images_root_2015 = osp.join(multi_root, split, 'image_2')
#         flow_root_2015 = osp.join(multi_root, split, 'flow_occ')

#         for idx_list in range(200):
#             # each sequence
#             for start_i in range(9, 9-nframes+2, -1): # 10-nframes+2, ..., 10        (12 - 5)
#                 # print(start_i)
#                 imgs, flows, gts = [], [], []
#                 frame_names = []

#                 for i in range(start_i, start_i+nframes):
#                     img = images_root_2015+"/000{:03}_{:02}.png".format(idx_list, i)
#                     imgs.append(img)
#                     frame_names.append(img.split('/')[-1])
#                     flow = flow_root_2015+"/000{:03}_10.png".format(idx_list)
#                     gt = True if i == 10 else False
#                     flows.append(flow)
#                     gts.append(gt)
#                 # print(imgs, flows)
#                 self.image_list.append(imgs)
#                 self.flow_list.append(flows[:-1])
#                 self.is_gt_list.append(gts[:-1])
#                 self.extra_info.append(frame_names)


#     def __rmul__(self, v):
#         self.flow_list = v * self.flow_list
#         self.image_list = v * self.image_list
#         self.is_gt_list = v * self.is_gt_list
#         return self

#     def __getitem__(self, index):
#         if not self.init_seed:
#             worker_info = torch.utils.data.get_worker_info()
#             if worker_info is not None:
#                 torch.manual_seed(worker_info.id)
#                 np.random.seed(worker_info.id)
#                 random.seed(worker_info.id)
#                 self.init_seed = True

#         index = index % len(self.image_list)
#         imgs_filenames = self.image_list[index]
#         flows_filenames = self.flow_list[index]
#         is_gts = self.is_gt_list[index]
#         # print(is_gts)
#         # print(imgs_filenames)
#         # print(flows_filenames)
#         imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]


#         flows, valids = [], []

#         for i in range(len(is_gts)):
#             flow, valid = frame_utils.readFlowKITTI(flows_filenames[i])
#             if not is_gts[i]:
#                 valid = valid * 0.0
#             flows.append(np.array(flow).astype(np.float32))
#             valids.append(valid)
        

#         if self.augmentor is not None:
#             imgs, flows, valids = self.augmentor(imgs, flows, valids)
#         # print(len(flows), len(valids), len(imgs))
#         imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
#         for i in range(len(flows)):
#             flows[i] = torch.from_numpy(flows[i]).permute(2, 0, 1).float()
#             valids[i] = torch.from_numpy(valids[i]).float()


#         # print(flows)
#         return imgs, flows, valids, ['0', [0]*self.nframes]





class KITTIMultiFrameEval(FlowDataset):
    def __init__(self,  multi_root='/sunshangkun/multi-kitti/', split='training', aug_params=None, nframes=3):
        super().__init__(aug_params=aug_params, sparse=True)
        self.image_list = []
        self.flow_list = []
        self.extra_info = []
        self.is_test = True if split == 'testing' else False
        

        images_root_2015 = osp.join(multi_root, split, 'image_2')
        flow_root_2015 = osp.join(multi_root, split, 'flow_occ')

        for idx_list in range(200):
            # each sequence
            # Try: [9, 10, 11].   
            imgs = [images_root_2015+"/000{:03}_{:02}.png".format(idx_list, i) for i in range(12-nframes, 12)]
            flows = [None for i in range(nframes-2)] + [flow_root_2015+"/000{:03}_10.png".format(idx_list)]
            frame_names = [img.split('/')[-1] for img in imgs]
            self.image_list.append(imgs)
            self.flow_list.append(flows)
            self.extra_info.append("000{:03}_10.png".format(idx_list))
            # print(imgs, flows)
        
        # print(self.extra_info[:5])
        # print(self.extra_info[:-5])


    def __getitem__(self, index):
        if self.is_test:
            imgs_filenames = self.image_list[index]
            imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
            imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
            return imgs, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        info_index = index % len(self.extra_info)
        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]
        # print(imgs_filenames)
        # print(flows_filenames)

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows, valids = [], []

        for filename in flows_filenames:
            if filename is not None:
                flow, valid = frame_utils.readFlowKITTI(filename)
                flows.append(np.array(flow).astype(np.float32))
                valids.append(valid)
            else:
                flows.append(None)
                valids.append(None)


        # print(len(flows), len(valids), len(imgs))
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        for i in range(len(flows)):
            if flows[i] is not None:
                flows[i] = torch.from_numpy(flows[i]).permute(2, 0, 1).float()
                if valids[i] is not None: valids[i] = torch.from_numpy(valids[i])
                else: valids[i] = (flows[i][0].abs() < 1000) & (flows[i][1].abs() < 1000)

        # print(flows)
        return imgs, flows, valids, self.extra_info[info_index]



# class KITTIMultiFrameEval_T4(FlowDataset):
#     def __init__(self,  multi_root='/sunshangkun/multi-kitti/', split='training', aug_params=None, nframes=3):
#         super().__init__(aug_params=aug_params, sparse=True)
#         self.image_list = []
#         self.flow_list = []
#         self.extra_info = []
#         self.is_test = True if split == 'testing' else False
        

#         images_root_2015 = osp.join(multi_root, split, 'image_2')
#         flow_root_2015 = osp.join(multi_root, split, 'flow_occ')

#         for idx_list in range(200):
#             # each sequence
#             # Try: [9, 10, 11, 12].   
#             imgs = [images_root_2015+"/000{:03}_{:02}.png".format(idx_list, i) for i in range(12-nframes, 12)]
#             flows = [None for i in range(nframes-2)] + [flow_root_2015+"/000{:03}_10.png".format(idx_list)]
#             frame_names = [img.split('/')[-1] for img in imgs]
#             self.image_list.append(imgs)
#             self.flow_list.append(flows)
#             self.extra_info.append("000{:03}_10.png".format(idx_list))
#             # print(imgs, flows)
        
#         # print(self.extra_info[:5])
#         # print(self.extra_info[:-5])


#     def __getitem__(self, index):
#         if self.is_test:
#             imgs_filenames = self.image_list[index]
#             imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
#             imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
#             return imgs, self.extra_info[index]

#         if not self.init_seed:
#             worker_info = torch.utils.data.get_worker_info()
#             if worker_info is not None:
#                 torch.manual_seed(worker_info.id)
#                 np.random.seed(worker_info.id)
#                 random.seed(worker_info.id)
#                 self.init_seed = True

#         index = index % len(self.image_list)
#         info_index = index % len(self.extra_info)
#         imgs_filenames = self.image_list[index]
#         flows_filenames = self.flow_list[index]
#         # print(imgs_filenames)
#         # print(flows_filenames)

#         imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
#         flows, valids = [], []

#         for filename in flows_filenames:
#             if filename is not None:
#                 flow, valid = frame_utils.readFlowKITTI(filename)
#                 flows.append(np.array(flow).astype(np.float32))
#                 valids.append(valid)
#             else:
#                 flows.append(None)
#                 valids.append(None)


#         # print(len(flows), len(valids), len(imgs))
#         imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
#         for i in range(len(flows)):
#             if flows[i] is not None:
#                 flows[i] = torch.from_numpy(flows[i]).permute(2, 0, 1).float()
#                 if valids[i] is not None: valids[i] = torch.from_numpy(valids[i])
#                 else: valids[i] = (flows[i][0].abs() < 1000) & (flows[i][1].abs() < 1000)

#         # print(flows)
#         return imgs, flows, valids, self.extra_info[info_index]



class SintelMultiframeEval(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/backup/data/sintel', dstype='clean', nframes=4, occ_map=False):
        super().__init__(aug_params)
        self.nframes = nframes
        self.is_test = True if split == 'test' else False
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion  = occ_map
        if self.occlusion: self.occ_list = []

        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(image_root, "*/*.png")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))
        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            if not self.is_test: 
                flo_filenames = [x for x in all_flo_filenames if base_folder in x]
                occ_filenames = [x for x in all_occ_filenames if base_folder in x]
            assert len(img_filenames) >= self.nframes

            i = 0
            # print(len(flo_filenames))
            while True:
                if i + self.nframes <= len(img_filenames):
                    imgs = img_filenames[i:i+self.nframes]
                    if not self.is_test: 
                        flows = flo_filenames[i:i+self.nframes-1]
                        occs = occ_filenames[i:i+self.nframes-1]
                    self.extra_info += [[base_folder, [j for j in range(i, i+self.nframes)]]]
                else:
                    imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                    if not self.is_test: 
                        flows = flo_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]
                        occs = occ_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]


                    ids = [-1 if j < i else j for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                    self.extra_info += [[base_folder, ids]]

                self.image_list += [imgs]
                if not self.is_test: 
                    self.flow_list += [flows]
                    if occ_map: self.occ_list += [occs]

                if i + self.nframes >= len(img_filenames): break
                else: i += self.nframes-1

    def __getitem__(self, index):
        if self.is_test:
            imgs_filenames = self.image_list[index]
            imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
            imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
            return imgs, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        info_index = index % len(self.extra_info)
        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]
        assert len(imgs) == len(flows) + 1
        
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        
        if self.occ_list is not None:
            occ_filenames = self.occ_list[index]
            occs = [torch.from_numpy(np.array(frame_utils.read_gen(occ)).astype(np.uint8) // 255).bool() for occ in occ_filenames]
            return imgs, flows, valids, self.extra_info[info_index], occs
        return imgs, flows, valids, self.extra_info[info_index]





class SintelMultiframeEval_stride2(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='/backup/data/sintel', dstype='clean', nframes=4, occ_map=False):
        super().__init__(aug_params)
        self.nframes = nframes
        self.is_test = True if split == 'test' else False
        flow_root = osp.join(root, split, 'flow')
        image_root = osp.join(root, split, dstype)
        occ_root = osp.join(root, split, 'occlusions')
        self.occlusion  = occ_map
        if self.occlusion: self.occ_list = []

        all_flo_filenames = sorted(glob(os.path.join(flow_root, "*/*.flo")))
        all_img_filenames = sorted(glob(os.path.join(image_root, "*/*.png")))
        all_occ_filenames = sorted(glob(os.path.join(occ_root, "*/*.png")))

        # ------------------------------------------------------------------------
        # Get unique basenames
        # ------------------------------------------------------------------------
        # e.g. base_folders = [alley_1", "alley_2", "ambush_2", ...]
        substract_full_base = cd_dotdot(all_img_filenames[0])
        base_folders = sorted(list(set([
            os.path.dirname(fn.replace(substract_full_base, ""))[1:] for fn in all_img_filenames
        ])))
        for base_folder in base_folders:
            img_filenames = [x for x in all_img_filenames if base_folder in x]
            if not self.is_test: 
                flo_filenames = [x for x in all_flo_filenames if base_folder in x]
                occ_filenames = [x for x in all_occ_filenames if base_folder in x]
            assert len(img_filenames) >= self.nframes

            i = 0
            # print(len(flo_filenames))
            while True:
                if i + self.nframes <= len(img_filenames):
                    imgs = img_filenames[i:i+self.nframes]
                    if not self.is_test: 
                        flows = flo_filenames[i:i+self.nframes-1]
                        occs = occ_filenames[i:i+self.nframes-1]
                    self.extra_info += [[base_folder, [j for j in range(i, i+self.nframes)]]]
                else:
                    imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                    if not self.is_test: 
                        flows = flo_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]
                        occs = occ_filenames[len(flo_filenames)-self.nframes+1:len(flo_filenames)]


                    ids = [j for j in range(len(img_filenames)-self.nframes, len(img_filenames))]
                    self.extra_info += [[base_folder, ids]]

                self.image_list += [imgs]
                if not self.is_test: 
                    self.flow_list += [flows]
                    if occ_map: self.occ_list += [occs]

                if i + self.nframes >= len(img_filenames): break
                else: i += self.nframes-2


    def __getitem__(self, index):
        if self.is_test:
            imgs_filenames = self.image_list[index]
            imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
            imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
            return imgs, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        info_index = index % len(self.extra_info)
        index = index % len(self.image_list)

        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]

        # print(imgs_filenames, flows_filenames, self.extra_info[info_index])

        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]
        assert len(imgs) == len(flows) + 1
          
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        
        if self.occ_list is not None:
            occ_filenames = self.occ_list[index]
            occs = [torch.from_numpy(np.array(frame_utils.read_gen(occ)).astype(np.uint8) // 255).bool() for occ in occ_filenames]
            return imgs, flows, valids, self.extra_info[info_index], occs
        return imgs, flows, valids, self.extra_info[info_index]






# class FlyingThings3DMultiFrame(FlowDataset):
#     def __init__(self, aug_params=None, root='/backup/data/flyingthings3d', split='training', dstype='frames_cleanpass'):
#         super().__init__(aug_params)

#         if split == 'training':
#             for cam in ['left']:
#                 for direction in ['into_future', 'into_past']:
#                     image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
#                     image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

#                     flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
#                     flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

#                     for idir, fdir in zip(image_dirs, flow_dirs):
#                         images = sorted(glob(osp.join(idir, '*.png')) )
#                         flows = sorted(glob(osp.join(fdir, '*.pfm')) )
#                    for i in range(len(flows)-1):
#                             if direction == 'into_future':
#                                 self.image_list += [ [images[i], images[i+1]] ]
#                                 self.flow_list += [ flows[i] ]
#                             elif direction == 'into_past':
#                                 self.image_list += [ [images[i+1], images[i]] ]
#                                 self.flow_list += [ flows[i+1] ]

 
class HD1KMultiFrame(FlowDataset):
    def __init__(self, aug_params=None, root='/datasets/flow/HD1k', nframes=3):
        super().__init__(aug_params, sparse=True)
        self.image_list = []
        self.flow_list = []
        self.nframes = nframes

        seq_ix = 0
        while 1:
            flo_filenames = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
            img_filenames = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

            if len(flo_filenames) == 0:
                break
            #####
            i = 0
            while True:
                if i + self.nframes <= len(img_filenames):
                    imgs = img_filenames[i:i+self.nframes]
                    flows = flo_filenames[i:i+self.nframes-1]
                else:
                    imgs = img_filenames[len(img_filenames)-self.nframes:len(img_filenames)]
                    flows = flo_filenames[len(img_filenames)-self.nframes:len(img_filenames)-1]
                self.image_list += [imgs]
                self.flow_list += [flows]

                if i + self.nframes >= len(img_filenames): break
                else: i += 1
            #####
            seq_ix += 1
    
    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        imgs_filenames = self.image_list[index]
        flows_filenames = self.flow_list[index]
        # print(imgs_filenames)
        # print(flows_filenames)
        imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
        flows, valids = [], []
        for filename in flows_filenames:
            flow, valid = frame_utils.readFlowKITTI(filename)
            flows.append(np.array(flow).astype(np.float32))
            valids.append(valid)

        assert len(imgs) == len(flows) + 1

        # grayscale images
        for i in range(len(imgs)):
            if len(imgs[i].shape) == 2: imgs[i] = np.tile(imgs[i][...,None], (1, 1, 3))
            else: imgs[i] = imgs[i][..., :3]
        # # print('Here YES!!')
        if self.augmentor is not None:
            imgs, flows, valids = self.augmentor(imgs, flows, valids)
        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]
        for i in range(len(valids)):
            if valids[i] is not None: valids[i] = torch.from_numpy(valids[i]).float()
            else: valids[i] = ((flows[i][0].abs() < 1000) & (flows[i][1].abs() < 1000)).float()
        # print(imgs[0].shape, flows[0].shape, valids[0].shape, valids[0].dtype, 'hd1k')
        return imgs, flows, valids, ['0', [0]*self.nframes]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding training set """
    TRAIN_DS = args.train_ds
    if args.stage == 'things':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = FlyingThings3DMultiFrame(aug_params, dstype='frames_cleanpass', split='training', root=args.things_root, nframes=args.T)
        final_dataset = FlyingThings3DMultiFrame(aug_params, dstype='frames_finalpass', split='training', root=args.things_root, nframes=args.T)
        train_dataset = clean_dataset + final_dataset
    elif args.stage == 'sintel':
        print('Yes!!!!')
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = SintelMultiframe(aug_params, split='training', dstype='clean', root=args.sintel_root, nframes=args.T)
        sintel_final = SintelMultiframe(aug_params, split='training', dstype='final', root=args.sintel_root, nframes=args.T)
        things = FlyingThings3DMultiFrame(aug_params, dstype='frames_cleanpass', split='training', root=args.things_root, nframes=args.T)
        hd1k = HD1KMultiFrame({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}, root=args.hd1k_root, nframes=args.T)
        kitti = KITTIMultiFrame_T4(multi_root=args.multi_root, split='training', aug_params={'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}, nframes=args.T)
        # print('****')
        # print(things[0][0][0].shape, sintel_final[0][0][0].shape)
        # print('====')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + 50 * kitti +  5 * hd1k + things
    elif args.stage == 'sintel2':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = SintelMultiframe(aug_params, split='training', dstype='clean', root=args.sintel_root, nframes=args.T)
        sintel_final = SintelMultiframe(aug_params, split='training', dstype='final', root=args.sintel_root, nframes=args.T)
        things = FlyingThings3DMultiFrame(aug_params, dstype='frames_cleanpass', split='training', root=args.things_root, nframes=args.T)
        hd1k = HD1KMultiFrame({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}, root=args.hd1k_root, nframes=args.T)
        kitti = KITTIMultiFrame_T4(multi_root=args.multi_root, split='training', aug_params={'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}, nframes=args.T)
        # print('****')
        # print(things[0][0][0].shape, sintel_final[0][0][0].shape)
        # print('====')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + 200 * kitti +  5 * hd1k + things
    elif args.stage == 'sintel3':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}

        sintel_clean = SintelMultiframe(aug_params, split='training', dstype='clean', root=args.sintel_root, nframes=args.T)
        sintel_final = SintelMultiframe(aug_params, split='training', dstype='final', root=args.sintel_root, nframes=args.T)
        things = FlyingThings3DMultiFrame(aug_params, dstype='frames_cleanpass', split='training', root=args.things_root, nframes=args.T)
        hd1k = HD1KMultiFrame({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True}, root=args.hd1k_root, nframes=args.T)
        kitti = KITTIMultiFrame(multi_root=args.multi_root, split='training', aug_params={'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True}, nframes=args.T)
        # print('****')
        # print(things[0][0][0].shape, sintel_final[0][0][0].shape)
        # print('====')
        train_dataset = 100 * sintel_clean + 100 * sintel_final + 50 * kitti + 5 * hd1k + things
    elif args.stage == 'spring':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
        spring = Spring(aug_params, input_frames=args.T, forward_warp=args.forward_warp, root=os.path.join(args.spring_root, 'train'), subsample_groundtruth=True, split=args.split)
        print("[dataset len: ]", len(spring))

        train_dataset = spring
    elif args.stage == 'kitti':
            aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
            if args.T == 4:
                train_dataset = KITTIMultiFrame_T4(multi_root=args.multi_root, split='training', aug_params=aug_params, nframes=args.T)
            else:
                train_dataset = KITTIMultiFrame(multi_root=args.multi_root, split='training', aug_params=aug_params, nframes=args.T)
    elif args.stage == 'bithings':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}

        clean_dataset = Bi_FlyingThings3DMultiFrame(aug_params, dstype='frames_cleanpass', split='training', root=args.things_root, nframes=args.T)
        final_dataset = Bi_FlyingThings3DMultiFrame(aug_params, dstype='frames_finalpass', split='training', root=args.things_root, nframes=args.T)
        train_dataset = clean_dataset + final_dataset


    # print('xxxxxxxxx')
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                    pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    print('Training with %d image pairs' % len(train_dataset))
    return train_loader




if __name__ == '__main__':
    # KITTIMultiFrame()
    # data = HD1KMultiFrame({'crop_size': [368, 768], 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
    # data = KITTIMultiFrame(nframes=5)
    # images, flows, valids, extra_infos = data[0]
    # print(torch.sum(valids[0]))
    # print(torch.sum(valids[1]))
    # print(torch.sum(valids[2]))
    # print(torch.sum(valids[3]))
    # print('=====')
    # images, flows, valids, extra_infos = data[1]
    # print(torch.sum(valids[0]))
    # print(torch.sum(valids[1]))
    # print(torch.sum(valids[2]))
    # print(torch.sum(valids[3]))
    # print('=====')
    # images, flows, valids, extra_infos = data[2]
    # print(torch.sum(valids[0]))
    # print(torch.sum(valids[1]))
    # print(torch.sum(valids[2]))
    # print(torch.sum(valids[3]))
    # print('=====')
    # images, flows, valids, extra_infos = data[3]
    # print(torch.sum(valids[0]))
    # print(torch.sum(valids[1]))
    # print(torch.sum(valids[2]))
    # print(torch.sum(valids[3]))
    # print('=====')

    # data = KITTIMultiFrameEval(nframes=3)
    data = Spring(input_frames=4, forward_warp=False, root='OpticalFlowDataset/spring/train', subsample_groundtruth=True)
    data[0]
    data[1]
    data[2]
    data[-1]
    # data = SintelMultiframeEval_stride2(split='training', root='/mnt/apdcephfs_sh3/share_301074934/shangkunsun/OpticalFlowDataset/sintel', nframes=4)
    # data[0]
    # data[1]
    # data[2]
    # data[3]
    # data[4]
    # data[5]
    # data[6]
