import sys
sys.path.append('core')

import argparse
import numpy as np
import torch
from tqdm import tqdm
import os

import cv2
import datasets
import mf_datasets
from models import *
from utils import flow_viz
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate
from PIL import Image
import imageio
import torch.nn.functional as F
from matplotlib import pyplot as plt

@torch.no_grad()
def validate_kitti_mf(model, iters=6, multi_root=None, nframes=3):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = mf_datasets.KITTIMultiFrameEval(split='training', multi_root=multi_root, nframes=nframes)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        images, flow_gts, valids, frame_names = val_dataset[val_id]
        images = [x[None].cuda() for x in images]

        padder = InputPadder(images[0].shape, mode='kitti')
        images = padder.pad_list(images)
        flows = model(images, iters=iters, test_mode=True)
        flows = [padder.unpad(flow[0]).cpu() for flow in flows]
        # print(flow_gts)
        for i in range(len(flow_gts)):
            if flow_gts[i] is not None:
                epe = torch.sum((flows[i] - flow_gts[i])**2, dim=0).sqrt()
                mag = torch.sum(flow_gts[i]**2, dim=0).sqrt()

                epe = epe.view(-1)
                mag = mag.view(-1)
                val = valids[i].view(-1) >= 0.5

                out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                epe_list.append(epe[val].mean().item())
                out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}



def EPE(flow, flow_gt):
    return torch.sum((flow - flow_gt)**2, dim=0).sqrt()

def patch_EPE(flow, flow_gt):
    return torch.sum(flow - flow_gt)

def vis_worst_flow_img(model, iters=6, dstype='final', val_id=0, save_root='./vis/model_name/val_id', nframes=2):
    """ Peform validation using the Sintel (train) split """
    model.eval()

    val_dataset = mf_datasets.SintelMultiframeEval(split='training', dstype=dstype, nframes=nframes)


    images, flows_gt, _, _ = val_dataset[val_id]
    images = [x[None].cuda() for x in images]
    padder = InputPadder(images[0].shape)
    images = padder.pad_list(images)
    flows = model(images, iters=iters, test_mode=True)
    flows = [padder.unpad(flow[0]).cpu() for flow in flows]


    for i in range(len(flows)):
        epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
        print(epe.mean().item())


@torch.no_grad()
def get_worst_case(model, iters=12, root='/data/Sintel', tqdm_miniters=1, nframes=3, save_root='/share/sunshangkun/worst_cases_mf', ):
    model.eval()
    results = {}

    for dstype in ['clean', 'final']:
        val_dataset = mf_datasets.SintelMultiframeEval(split='training', dstype=dstype, root=root, nframes=nframes)
        epe_list = []
        result_list = []

        for val_id in tqdm(range(len(val_dataset))):
            images, flows_gt, _, (_, frame_ids) = val_dataset[val_id]
            images = [x[None].cuda() for x in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).cpu() for flow in flows]


            for i in range(len(flows)):
                # print(len(flows), flows[i].shape, flows_gt[i].shape)
                epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                if frame_ids[i] != -1:
                    epe_list.append(epe.view(-1).numpy())
                    # print(val_id, frame_ids, epe.mean().item())
                    result_list.append(epe.numpy().mean())

        result_list = np.array(result_list)
        worst_index = np.argsort(result_list)[-30:]
        worst_index = worst_index[::-1]
        if not os.path.exists(save_root): os.makedirs(save_root)
        print(worst_index)

        for i in range(len(worst_index)):
            images, flows_gt, _, (_, frame_ids) = val_dataset[worst_index[i]]
            images = [x[None].cuda() for x in images]
            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).cpu() for flow in flows]
            dir_path = os.path.join(save_root, str(i))

            # print('ploting :', i, worst_index[i], 'EPE:', result_list[worst_index[i]])

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for j, img in enumerate(images):
                out_path = os.path.join(dir_path, f'{worst_index[i]}-image{j}.png')
                Image.fromarray(np.uint8(img.squeeze(0).permute(1,2,0).cpu().numpy())).save(out_path)

            for j, img in enumerate(flows):
                out_path = os.path.join(dir_path, f'{worst_index[i]}-flowdiff{j}.png')
                flow_diff = abs(flows[j] - flows_gt[j]).squeeze(0).mean(0).cpu().numpy()
                plt.imshow(flow_diff, cmap='Greens')
                plt.savefig(out_path)
                epe = np.mean(torch.sum((flows[j] - flows_gt[j])**2, dim=0).sqrt().numpy())
                print(f'{worst_index[i]}-{j}: {epe}')
                img = img.permute(1, 2, 0).numpy()
                out_path = os.path.join(dir_path, f'{worst_index[i]}-pred{j}.png')
                Image.fromarray(flow_viz.flow_to_image(img)).save(out_path)

            for j, img in enumerate(flows_gt):
                out_path = os.path.join(dir_path, f'{worst_index[i]}-gt{j}.png')
                img = img.permute(1, 2, 0).cpu().numpy()
                Image.fromarray(flow_viz.flow_to_image(img)).save(out_path)
            
        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def create_sintel_submission_mf(args, model, iters, output_path='sintel_submission', nframes=3):
    """ Create submission for the Sintel leaderboard """
    model.eval()

    for dstype in ['clean', 'final']:
        test_dataset = mf_datasets.SintelMultiframeEval(root=args.sintel_root, split='test', aug_params=None, dstype=dstype, nframes=nframes)

        for test_id in tqdm(range(len(test_dataset))):
            images, (scene, frame_ids) = test_dataset[test_id]
            images = [image[None].cuda() for image in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy() for flow in flows]

            output_dir = os.path.join(output_path, dstype, scene)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            vis_dir = f'{output_path}/{dstype}/'
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)

            assert len(flows) == len(frame_ids) - 1
            for i in range(len(flows)):
                if frame_ids[i] != -1:
                    output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame_ids[i]+1))
                    frame_utils.writeFlow(output_file, flows[i])
                    flow_img = Image.fromarray(flow_viz.flow_to_image(flows[i]))
                    flow_img.save(os.path.join(vis_dir, f'{scene}-{frame_ids[i]+1}.png'))




@torch.no_grad()
def create_sintel_submission_mf_warmup(args, model, iters, output_path='sintel_submission', nframes=3):
    """ Create submission for the Sintel leaderboard """
    model.eval()

    for dstype in ['clean', 'final']:
        test_dataset = mf_datasets.SintelMultiframeEval(root=args.sintel_root, split='test', aug_params=None, dstype=dstype, nframes=nframes)

        flow_prev, scene_prev = None, None
        for test_id in tqdm(range(len(test_dataset))):
            images, (scene, frame_ids) = test_dataset[test_id]
            images = [image[None].cuda() for image in images]

            if scene != scene_prev:
                flow_prev = None
            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows, flows_lowres = model(images, iters=iters, flow_init=flow_prev, test_mode=True)
            flow_prev = [forward_interpolate(flows_lowres[i][0])[None].cuda() for i in range(len(flows_lowres))]
            flows = [padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy() for flow in flows]

            output_dir = os.path.join(output_path, dstype, scene)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            vis_dir = f'{output_path}/{dstype}/'
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)

            assert len(flows) == len(frame_ids) - 1
            for i in range(len(flows)):
                if frame_ids[i] != -1:
                    output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame_ids[i]+1))
                    frame_utils.writeFlow(output_file, flows[i])
                    flow_img = Image.fromarray(flow_viz.flow_to_image(flows[i]))
                    flow_img.save(os.path.join(vis_dir, f'{scene}-{frame_ids[i]+1}.png'))


@torch.no_grad()
def create_sintel_submission_vis(model, iters, warm_start=False, output_path='sintel_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    vis_path = output_path.split('/')[-1]
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype, root=args.sintel_root)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=iters, flow_init=flow_prev, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/{vis_path}/{dstype}/'):
                os.makedirs(f'vis_test/{vis_path}/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            # image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/{vis_path}/{dstype}/flow/{test_id}.png')
            # imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def validate_chairs(model, iters=6, root='/data/FlyingChairs_release/data'):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation', root=root)
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6, root='/data/FlyingThings3D'):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = []
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation', root=root)
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel(model, iters=6, root='/data/Sintel', tqdm_miniters=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, root=root)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape, factor=8)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_mf(model, iters=6, root='/data/Sintel', tqdm_miniters=1, nframes=3):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = mf_datasets.SintelMultiframeEval(split='training', dstype=dstype, root=root, nframes=nframes)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            images, flows_gt, _, (_, frame_ids) = val_dataset[val_id]
            images = [x[None].cuda() for x in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).cpu() for flow in flows]

            for i in range(len(flows)):
                # print(len(flows), flows[i].shape, flows_gt[i].shape)
                epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                if frame_ids[i] != -1:
                    epe_list.append(epe.view(-1).numpy())
                    # print(val_id, frame_ids, epe.mean().item())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ_mf(model, iters=6, root='/data/Sintel', tqdm_miniters=1, nframes=3):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
        val_dataset = mf_datasets.SintelMultiframeEval(split='training', dstype=dstype, root=root, nframes=nframes, occ_map=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            images, flows_gt, _, (_, frame_ids), occs = val_dataset[val_id]
            images = [x[None].cuda() for x in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).cpu() for flow in flows]

            for i in range(len(flows)):
                epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                if frame_ids[i] != -1:
                    epe_list.append(epe.view(-1).numpy())
                    epe_noc_list.append(epe[~occs[i]].numpy())
                    epe_occ_list.append(epe[occs[i]].numpy())

        epe_all = np.concatenate(epe_list)
        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_mf_warp(model, iters=6, root='/data/Sintel', tqdm_miniters=1, nframes=3):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        test_dataset = mf_datasets.SintelMultiframeEval(root=args.sintel_root, split='test', aug_params=None, dstype=dstype, nframes=nframes)
        epe_list = []

        for test_id in tqdm(range(len(test_dataset))):
            images, (scene, frame_ids) = test_dataset[test_id]
            images = [image[None].cuda() for image in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy() for flow in flows]


            for i in range(len(flows)):
                # print(len(flows), flows[i].shape, flows_gt[i].shape)
                
                epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                if frame_ids[i] != -1:
                    epe_list.append(epe.view(-1).numpy())
                    # print(val_id, frame_ids, epe.mean().item())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results




@torch.no_grad()
def vis_case_mf(model, imgs_filenames, flows_filenames, iters=6, vis_root='/share2/sunshangkun/vis_case'):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    imgs = [np.array(frame_utils.read_gen(filename)).astype(np.uint8) for filename in imgs_filenames]
    flows = [np.array(frame_utils.read_gen(filename)).astype(np.float32) for filename in flows_filenames]
    images = [torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs]
    flows_gt = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

    images = [x[None].cuda() for x in images]
    padder = InputPadder(images[0].shape)
    images = padder.pad_list(images)
    flows = model(images, iters=iters, test_mode=True)
    flows = [padder.unpad(flow[0]).cpu() for flow in flows]

    for i in range(len(flows)):
        # print(len(flows), flows[i].shape, flows_gt[i].shape)
        epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
        flow_diff = abs(flows[i] - flows_gt[i]).squeeze(0).mean(0).cpu().numpy()
        if not os.path.exists(vis_root): os.makedirs(vis_root)
        plt.imshow(flow_diff, cmap='Greens')
        plt.savefig(f'{vis_root}/flow_diff_{i}.png')
        epe_item = np.mean(epe.numpy())
        print(f'{i}: {epe_item}')

    # flow1 = flow_viz.flow_to_image(flows[0].permute(1,2,0).cpu().numpy())
    # flow2 = flow_viz.flow_to_image(flows[1].permute(1,2,0).cpu().numpy())
    # cv2.imwrite('flow1.jpg', flow1)
    # cv2.imwrite('flow2.jpg', flow2)





@torch.no_grad()
def validate_sintel_tri(model, iters=6, root='/data/Sintel', tqdm_miniters=1, nframes=3):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final']:
        val_dataset = mf_datasets.SintelMultiframe(split='training', dstype=dstype, root=root, nframes=nframes)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            images, flows_gt, _, (_, frame_ids) = val_dataset[val_id]
            images = [x[None].cuda() for x in images]

            padder = InputPadder(images[0].shape, factor=8)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow).cpu() for flow in flows]

            epe = torch.sum((flows[-1] - flows_gt[-1])**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())
            if frame_ids[0] == 0:
                for i in range(len(flows) - 1):
                    epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                    epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def vis_worst_case_mf(model, iters=6, root='/data/Sintel', tqdm_miniters=1, nframes=3):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['final']:
        val_dataset = mf_datasets.SintelMultiframeEval(split='training', dstype=dstype, root=root, nframes=nframes)
        epe_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            images, flows_gt, _, (_, frame_ids) = val_dataset[val_id]
            images = [x[None].cuda() for x in images]

            padder = InputPadder(images[0].shape)
            images = padder.pad_list(images)
            flows = model(images, iters=iters, test_mode=True)
            flows = [padder.unpad(flow[0]).cpu() for flow in flows]

            for i in range(len(flows)):
                # print(len(flows), flows[i].shape, flows_gt[i].shape)
                epe = torch.sum((flows[i] - flows_gt[i])**2, dim=0).sqrt()
                if frame_ids[i] != -1:
                    flow_diff = abs(flows[i] - flows_gt[i]).squeeze(0).mean(0).cpu().numpy()
                    if not os.path.exists('./ours_real'): os.makedirs('./ours_real')
                    plt.imshow(flow_diff, cmap='Greens')
                    # plt.colorbar()
                    plt.savefig(f'./ours_real/flow_diff_{frame_ids[i]}.png')
                    epe_item = np.mean(epe.numpy())
                    print(f'{frame_ids[i]}: {epe_item}')
                    epe_list.append(epe.view(-1).numpy())
                    # print(val_id, frame_ids, epe.mean().item())

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_sintel_occ(model, iters=6, root='/data/Sintel', tqdm_miniters=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in tqdm(range(len(val_dataset)), miniters=tqdm_miniters):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def create_kitti_submission(model, iters, output_path='kitti_submission'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def create_kitti_submission_mf(args, model, iters, output_path='kitti_submission', nframes=3):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = mf_datasets.KITTIMultiFrameEval(multi_root=args.multi_root, split='testing', aug_params=None, nframes=nframes) # 和sintel submission的dataset有何处不同

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        images, frame_name = test_dataset[test_id]
        images = [x[None].cuda() for x in images]
        padder = InputPadder(images[0].shape) # TODO: Mode?
        images = padder.pad_list(images)
        flows = model(images, iters=iters, test_mode=True)
        flows = [padder.unpad(flow[0]).permute(1, 2, 0).cpu().numpy() for flow in flows]
        print(frame_name, output_path)
        output_filename = os.path.join(output_path, frame_name)
        frame_utils.writeFlowKITTI(output_filename, flows[-1])
        print(frame_name)


        flow_img = flow_viz.flow_to_image(flows[-1])
        image = Image.fromarray(flow_img)

        if not os.path.exists(f'vis_kitti2'):
            os.makedirs(f'vis_kitti2/flow')

        image.save(f'vis_kitti2/flow/{frame_name}')


TRAIN_SIZE = [432, 960]

class InputPadder2:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'kitti432':
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == 'kitti400':
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == 'kitti376':
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='constant', value=0.0) for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    
    def pad_list(self, inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

import math
def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

@torch.no_grad()
def validate_kitti_tile(model, iters=6, root='/data/kitti15'):
    sigma=0.05
    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [376, 720]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    from core.datasets import KITTI
    val_dataset = KITTI(split='training', root=root)


    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        # print(image1.shape, 'ooo')
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 376
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder2(image1.shape, mode='kitti376')
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())
        # print(image1.shape, 'xxx')

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
            print(image1_tile.shape, '=======', image2_tile.shape, h, w, TRAIN_SIZE[0], TRAIN_SIZE[1])
            flow_s = model([image1_tile, image2_tile], test_mode=True)
            flow_pre = flow_s[-1]

            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti-epe': epe, 'kitti-f1': f1}


@torch.no_grad()
def validate_kitti_mf_tile(model, iters=6, multi_root=None, nframes=3):
    """ Peform validation using the KITTI-2015 (train) split """
    sigma=0.05
    IMAGE_SIZE = [432, 1242]
    TRAIN_SIZE = [432, 960]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = mf_datasets.KITTIMultiFrameEval(split='training', multi_root=multi_root, nframes=nframes)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        images, flow_gts, valids, frame_names = val_dataset[val_id]
        # print(images[0].shape, '===========xxxxx')
        padder = InputPadder2(images[0].shape, mode='kitti432')

        new_shape = images[0].shape[1:]
        if new_shape[1] != IMAGE_SIZE[1]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)


        images = [x[None].cuda() for x in images]
        # print(images[0].shape, '=======', images[1].shape, h, w, TRAIN_SIZE[0], TRAIN_SIZE[1])
        images = padder.pad_list(images)
        # print(images[0].shape, '=zzzzzzz=====xxxxx')


        flows = [0 for _ in flow_gts]
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            images_tile = [x[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]] for x in images]
            # print(images[0].shape, '=======', images[1].shape, h, w, TRAIN_SIZE[0], TRAIN_SIZE[1])

            pred_flows = model(images_tile, iters=iters, test_mode=True)
            padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
            flows = [flows[j] + F.pad(pred_flows[j] * weights[idx], padding) for j in range(len(flows))]
            flow_count += F.pad(weights[idx], padding)


        flows = [flow / flow_count for flow in flows]
        flows = [padder.unpad(flow[0]).cpu() for flow in flows]
        # print(flow_gts)
        for i in range(len(flow_gts)):
            if flow_gts[i] is not None:
                epe = torch.sum((flows[i] - flow_gts[i])**2, dim=0).sqrt()
                mag = torch.sum(flow_gts[i]**2, dim=0).sqrt()

                epe = epe.view(-1)
                mag = mag.view(-1)
                val = valids[i].view(-1) >= 0.5

                out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                epe_list.append(epe[val].mean().item())
                out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}

@torch.no_grad()
def validate_kitti(model, iters=6, root='/data/kitti15'):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training', root=root)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", default='sintel')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name', type=str, default='SKFlow_MF8')
    parser.add_argument('--chairs_root', type=str, default=None)
    parser.add_argument('--sintel_root', type=str, default='/share2/public/datasets/flow/sintel')
    parser.add_argument('--things_root', type=str, default=None)
    parser.add_argument('--kitti_root', type=str, default=None)
    parser.add_argument('--multi_root', type=str, default='/share/sunshangkun/multi-kitti')
    parser.add_argument('--hd1k_root', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='./submission')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    parser.add_argument('--large_kernel_size', type=int, default=31)
    parser.add_argument('--T', type=int, default=3)
    parser.add_argument('--kernels', type=int, nargs='+', default=[7, 7, 15])
    parser.add_argument('--k_pool', type=int, nargs='+', default=[5, 9, 17])
    parser.add_argument('--k_conv', type=int, nargs='+', default=[1, 15])
    parser.add_argument('--PCBlock', type=str, default='PCBlock')
    parser.add_argument('--corr_k_conv', type=int, nargs='+', default=[1, 3])
    parser.add_argument('--conv_corr_kernel', type=int, default=1)
    parser.add_argument('--lkpretrain', help="restore checkpoint")
    parser.add_argument('--UpdateBlock', type=str, default='SKUpdateBlock_TAM')
    parser.add_argument('--PCUpdater_conv', type=int, nargs='+', default=[1, 7])
    parser.add_argument('--SKII_Block', type=str, default='SKII_Block_v1')
    parser.add_argument('--perceptual_kernel', type=int, default=1)
    parser.add_argument('--no_depthwise', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--vis_save_root', type=str, default=None)
    parser.add_argument('--worstcase_root', type=str, default='/share2/sunshangkun/worst_cases_mf/SKFlow_TAM-things')
    parser.add_argument('--use_gma', default=True, action='store_true',
                        help='use gma module')
    parser.add_argument('--use_se', default=False, action='store_true',
                        help='use se module')
    parser.add_argument('--cost_encoder_v1', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--cost_encoder_v2', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--cost_encoder_v3', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--cost_encoder_v4', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--cost_encoder_v5', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--use_resnext', default=False, action='store_true',
                        help='use se module')
    parser.add_argument('--fixed_kernel_weights', type=float, nargs='+', default=None)

    parser.add_argument('--perceptuals', type=int, nargs='+', default=[1, 7])
    parser.add_argument('--dilated_kernel', type=int, default=None)
    parser.add_argument('--dilation_rate', type=int, default=None)
    parser.add_argument('--flow_propogation_1', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--flow_propogation_2', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--corr_propogation', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--corr_propogation_v2', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--corr_propogation_v3', default=False, action='store_true',
                        help='conv encoder')
    parser.add_argument('--dilated_kernels', type=int, nargs='+', default=[7])
    parser.add_argument('--MotionEncoder', type=str, default='SKMotionEncoder6_Deep_nopool_res')
    parser.add_argument('--Encoder', type=str, default='Twins_CSC')
    parser.add_argument('--decoder_dim', type=int, default=256)
    parser.add_argument('--use_prev_flow', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--use_mfc', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--use_temporal_decoder', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--freeze_bn', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--freeze_others', default=False, action='store_true',
                        help='self-attn encoder')
                        
    parser.add_argument('--temporal_ckpt', default=None, type=str,
                        help='self-attn encoder')
    parser.add_argument('--no_temporal_project', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--fuse2', default=False, action='store_true',
                        help='self-attn encoder')
    parser.add_argument('--encoder_ckpt', type=str, default='b16_ptk710_ftk710_ftk700_f8_res224.pth')
    parser.add_argument('--planB', default=False, action='store_true')
    parser.add_argument('--use_spatio_attn', default=False, action='store_true')
    parser.add_argument('--fusion1', default=False, action='store_true')




                        
    args = parser.parse_args()


    model = torch.nn.DataParallel(eval(args.model_name)(args))
    # if args.model is not None:
    #     ckpt = torch.load(args.model)
    #     target_keys = [key for key in ckpt.keys() if 'update_block.encoder' in key]
    #     if len(target_keys) != 0:
    #         for key in target_keys:
    #             ckpt[key.replace('update_block.encoder', 'motion_encoder')] = ckpt[key]
    #             del ckpt[key]
        # print(ckpt.keys())
        # print('====')
        # print(model.state_dict().keys())
        # model.load_state_dict(ckpt, strict=True)
    if args.model is not None:
        ckpt = torch.load(args.model)
        if 'model' not in ckpt.keys():
            model.load_state_dict(torch.load(args.model), strict=True)
        else:
            model.load_state_dict(torch.load(args.model)['model'], strict=True)
    
    # if args.temporal_ckpt is not None:
    #     ckpt = torch.load(args.temporal_ckpt)
    #     target_keys = [key for key in ckpt.keys() if 'temporal' in key]
    #     model_dict = model.state_dict()
    #     for key in model_dict:
    #         if key in target_keys: 
    #             print('loading ' + key)
    #             model_dict[key] = ckpt[key]
    #     model.load_state_dict(model_dict)

    model.cuda()
    model.eval()

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('begin evaluation!')
    print(f"Parameter Count: {count_parameters(model)}")

    with torch.no_grad():
        # validate_kitti_mf(model.module, iters=args.iters, root=args.kitti_root, multi_root=args.multi_root, nframes=args.T)
        # vis_worst_flow_img(model.module, iters=args.iters, val_id=0)
        # get_worst_case(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T, save_root=args.worstcase_root)
        # create_sintel_submission_vis(model, args.iters, warm_start=True, output_path=args.output_path)
        # create_kitti_submission(model, args.iters, output_path=args.output_path)
        
        # validate_sintel_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)

        
        # imgs_filenames = ['/share2/public/datasets/flow/sintel/training/clean/ambush_6/frame_0014.png', '/share2/public/datasets/flow/sintel/training/clean/ambush_6/frame_0015.png', '/share2/public/datasets/flow/sintel/training/clean/ambush_6/frame_0016.png']
        # flows_filenames = ['/share2/public/datasets/flow/sintel/training/flow/ambush_6/frame_0014.flo', '/share2/public/datasets/flow/sintel/training/flow/ambush_6/frame_0015.flo']
        # vis_case_mf(model.module, imgs_filenames=imgs_filenames, flows_filenames=flows_filenames, iters=args.iters)
        
        # validate_sintel_mf_tile(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)
        # vis_worst_case_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)
        # validate_sintel_occ_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)
        
        create_kitti_submission_mf(args, model.module, args.iters, output_path=args.output_path, nframes=args.T)
        # create_sintel_submission_mf_warmup(args, model.module, args.iters, output_path=args.output_path, nframes=args.T)



        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters, root=args.chairs_root)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters, root=args.things_root)

        elif args.dataset == 'sintel':
            validate_sintel_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)

        elif args.dataset == 'sintel_occ':
            validate_sintel_occ_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)

        elif args.dataset == 'kitti':
            validate_kitti_mf(model.module, iters=args.iters, multi_root=args.multi_root, nframes=args.T)
            # validate_kitti_mf_tile(model.module, iters=args.iters, multi_root=args.multi_root, nframes=args.T)
        
        elif args.dataset == 'all':
            # validate_sintel_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)
            # validate_kitti_mf_tile(model.module, iters=args.iters, multi_root=args.multi_root, nframes=args.T)
            validate_kitti_mf(model.module, iters=args.iters, multi_root=args.multi_root, nframes=args.T)
            validate_sintel_occ_mf(model.module, iters=args.iters, root=args.sintel_root, nframes=args.T)
            # validate_kitti_tile(model.module, iters=args.iters, root=args.kitti_root)
