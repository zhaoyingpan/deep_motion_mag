import random
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
# from utils.utils import im2tracks, sample_color
import sys
from torchvision import models
sys.path.append('./m_utils')
sys.path.append('./m_utils/raft_model')

from raft_model.raft import RAFT
from raft_model.flow_utils import flow_viz
# from flow_utils.utils import InputPadder

from mm_utils import prepare_data, img_scaling, img2int, concatenate_imgs, reverse_frames, batch_multiplication


class MM(nn.Module):
    def __init__(self, args):
        super(MM, self).__init__()
        
        if args.model_type == 'raft':
            raft_args = argparse.Namespace()
            raft_args.model = args.model_dir
            raft_args.small = False
            raft_args.mixed_precision = False
            raft_args.alternate_corr = False
            # model = torch.nn.DataParallel(RAFT(raft_args)).to(device)
            model = torch.nn.DataParallel(RAFT(raft_args)).cuda()
            model.load_state_dict(torch.load(raft_args.model))
            model = model.module
            model.eval()
            self.model = model
        elif args.model_type == 'gmflow':
            self.model = model
        
        self.mm_loss = args.mm_loss
        if args.mm_loss == 'L1' or 'log':
            self.mm_loss_function = nn.L1Loss()
        elif args.mm_loss == 'MSE':
            self.mm_loss_function = nn.MSELoss()

        if args.color_loss == 'L1':
            self.color_loss_function = nn.L1Loss()
        elif args.color_loss == 'MSE':
            self.color_loss_function = nn.MSELoss()

        if args.enable_corrwise:
            from corrwise.corrwise import CorrWise
            self.corrwise = CorrWise(self.color_loss_function)
            self.enable_corrwise = True
        else:
            self.enable_corrwise = False

        self.alpha = args.alpha
        self.perceptual = args.perceptual
        self.enable_reverse_color = args.enable_reverse_color
        self.enable_reverse_mm = args.enable_reverse_mm
        self.enable_align_loss = args.enable_align_loss

        # self.color_weight = args.color_weight if args.disable_reverse_color else args.color_weight / 2
        # self.mag_weight = args.mag_weight if args.disable_reverse_mm else args.mag_weight / 2
        # self.perceptual_weight = args.perceptual_weight if args.disable_reverse_color else args.perceptual_weight / 2
        self.color_weight = args.color_weight
        self.mag_weight = args.mag_weight
        self.perceptual_weight = args.perceptual_weight
        print('color_weight: {}, mag_weight: {}, perceptual_weight: {}'.format(self.color_weight, self.mag_weight, self.perceptual_weight))

        if args.perceptual:
            self.perceptual_loss_function = VGGLoss()

        self.gaussian_blur = args.gaussian_blur
        self.flexible_alpha = args.flexible_alpha_training

        self.num_iters = args.num_iters
        self.enable_cycle_check = args.enable_cycle_check

        self.second_frame = 2

    
    def cal_delta_n_color(self, ims, detach=False):
        tracks = self.im2tracks(img_scaling(ims))
        # shape: b*N*2*H*W
        deltas = tracks[:, 1:] - tracks[:, :-1]
        # shape: b*(N-1)*2*H*W
        if detach:
            tracks = tracks.detach()
            deltas = deltas.detach()
        torch.cuda.empty_cache()

        colors = self.sample_color(tracks, ims)
        # shape: (N*b)*3*H*W
        torch.cuda.empty_cache()

        return deltas, colors

    def cal_mm_loss(self, tgt, src):
        if self.mm_loss == 'lorentzian':
            from mm_utils import lorentzian_function
            return self.mag_weight * lorentzian_function(src, tgt, s=2)
        else:
            return self.mag_weight * self.mm_loss_function(tgt, src)
    
    def cal_color_loss(self, tgt, src):
        if self.enable_corrwise:
            return self.cal_corrwise_loss(tgt, src)
        else:
            return self.color_weight * self.color_loss_function(tgt, src)

    def cal_corrwise_loss(self, tgt, src):
        batch_size, num_frames, _, h, w = tgt.shape
        return self.color_weight * self.corrwise(tgt.reshape(num_frames*batch_size, -1, h, w), src.reshape(num_frames*batch_size, -1, h, w))[0]

    def cal_perceptual_loss(self, tgt, src):
        return self.perceptual_weight * self.perceptual_loss_function(tgt, src).mean()

    def cal_align_loss(self, tgt, src):
        batch_size, num_frames, _, h, w = tgt.shape
        tgt_ = []
        loss_function = nn.MSELoss()
        for i in range(num_frames):
            with torch.no_grad():
                _, flow = self.model(tgt[:, i], src[:, i], iters=self.num_iters, test_mode=True)
                
                base = torch.meshgrid(torch.arange(h), torch.arange(w))[::-1]
                base = torch.stack(base).float().cuda()
                flow = flow + base
                # n*b*2*h*w
                grid =  -1. + 2. * flow/(-1 + torch.tensor([w, h]).to(torch.float32).cuda()).unsqueeze(-1).unsqueeze(-1)
                tgt_.append(torch.nn.functional.grid_sample(src[:, i], grid.to(torch.float32).permute(0,2,3,1), align_corners=True).detach())
        tgt_ = torch.stack(tgt_)
        loss = loss_function(tgt, tgt_)
        
        return loss

    def forward(self, out, src, alphas=None):
        if self.flexible_alpha:
            alpha = alphas
        else:
            alpha = self.alpha
        src, tgt = prepare_data(out, src, self.second_frame)
        delta_src, colors_src = self.cal_delta_n_color(src, True)
        delta_tgt, colors_tgt = self.cal_delta_n_color(tgt, False)
        torch.cuda.empty_cache()
        
        if self.enable_cycle_check:
            mag_mask, color_mask = self.get_occlusion_mask(src, thresh=1)
            mag_loss = self.cal_mm_loss(mag_mask * delta_tgt, batch_multiplication(alpha, mag_mask * delta_src))
            color_loss = self.cal_color_loss(color_mask * colors_tgt, color_mask * colors_src)
        else:
            mag_loss = self.cal_mm_loss(delta_tgt, batch_multiplication(alpha, delta_src))
            color_loss = self.cal_color_loss(colors_tgt, colors_src)

        torch.cuda.empty_cache()

        if self.enable_reverse_mm or self.enable_reverse_color:
            reversed_src = reverse_frames(src)
            reversed_tgt = reverse_frames(tgt)
            delta_reversed_src, colors_reversed_src = self.cal_delta_n_color(reversed_src, True)
            delta_reversed_tgt, colors_reversed_tgt = self.cal_delta_n_color(reversed_tgt, False)
            torch.cuda.empty_cache()
            if self.enable_reverse_mm:
                mag_loss += self.cal_mm_loss(delta_reversed_tgt, batch_multiplication(alpha, delta_reversed_src))
            if self.enable_reverse_color:
                color_loss += self.cal_color_loss(colors_reversed_tgt, colors_reversed_src)
            torch.cuda.empty_cache()

        loss = mag_loss + color_loss
        if self.perceptual:
            _, _, h, w = colors_src.shape
            loss += self.cal_perceptual_loss(colors_tgt, colors_src)
            if self.enable_reverse_color:
                loss += self.cal_perceptual_loss(colors_reversed_tgt, colors_reversed_src)
            torch.cuda.empty_cache()

        if self.enable_align_loss:
            loss += self.cal_align_loss(tgt, src)

        return loss, mag_loss.detach(), color_loss.detach()
    
    def warp(self, im, flow, padding_mode='zeros'):
        # im, flow: b*2*h*w
        # grid_sample assumes the grid is in the range in [-1, 1]
        grid =  -1. + 2. * flow/(-1 + torch.tensor([im.shape[-1], im.shape[-2]]).to(torch.float32).cuda()).unsqueeze(-1).unsqueeze(-1)

        warped = torch.nn.functional.grid_sample(
        im.to(torch.float32),
        grid.permute((0, 2, 3, 1)).to(torch.float32),
        # mode = 'bilinear', padding_mode = 'zeros'
        padding_mode = padding_mode,
        align_corners=True
        )
        return warped

    def im2tracks(self, ims):

        batch_size, num_frames, _, h, w = ims.shape
        track = torch.stack(torch.meshgrid(torch.arange(h),torch.arange(w))).flip([0]).unsqueeze(0).to(torch.float32).cuda()
        track = track.repeat(batch_size, 1, 1, 1)
        tracks = [track.clone()]
        torch.cuda.empty_cache()
        for t in range(num_frames-1):
            _, flow = self.model(ims[:, t], ims[:, t+1], iters=self.num_iters, test_mode=True)
            torch.cuda.empty_cache()
            track = self.warp(flow, track) + track
            tracks.append(track.clone())
        tracks = torch.stack(tracks, dim=1)
        
        if self.gaussian_blur:
            from scipy.ndimage import gaussian_filter1d
            # sigma=1, axis=0
            blurred_tracks = gaussian_filter1d(tracks, 1, 0)
            # import kornia
            # blurred_tracks = torch.stack((kornia.filters.gaussian_blur2d(tracks[:, :, 0], (3,3), (1,1)), kornia.filters.gaussian_blur2d(tracks[:, :, 1], (3,3), (1,1))), 2)
            blurred_tracks[0] = tracks[0].clone()
            return blurred_tracks
        else:
            return tracks


    def sample_color(self, tracks, ims):
        # tracks: T*N*2*H*W, ims: T*N*3*H*W
        batch_size, num_frames, _, h, w = ims.shape
        colors = self.warp(ims.reshape(batch_size*num_frames, -1, h, w), tracks.reshape(batch_size*num_frames, -1, h, w), padding_mode='border')
        return colors

    def get_cycle_consistency(self, im1, im2):
        '''
        Calculates F(im1, im2) \circ F(im2, im1)
        This tells us how cycle consistent the flow is
        '''
        
        # Calculate Flows        
        with torch.no_grad():
            _, flow_forward = self.model(im1, im2, iters=self.num_iters, test_mode=True)
            _, flow_backward = self.model(im2, im1, iters=self.num_iters, test_mode=True)
        # b*2*h*w

        base = torch.meshgrid(torch.arange(flow_backward.shape[-2]), torch.arange(flow_backward.shape[-1]))[::-1]
        base = torch.stack(base).float().cuda()
        cycle = self.warp(flow_backward + base, flow_forward + base, padding_mode='border')
                
        # Get deviation from 0 flow
        diff = cycle - base
        
        return diff

    def get_occlusion_mask(self, ims, thresh=1):
        '''
        occlusion mask by cycle consistency
        0 = occlusion, 1 = no occlusion
        
        optionally pass in precalculated flows
        '''
        batch_size, num_frames, _, h, w = ims.shape
        all_masks = []
        for t in range(num_frames-1):
            diff = self.get_cycle_consistency(ims[:, 0], ims[:, t+1])
            error = torch.mean(diff.abs(), dim=1, keepdim=True)
            mask = (error < thresh).float()
            all_masks.append(mask)
        
        return torch.stack(all_masks, dim=1), torch.stack([torch.ones_like(mask)] + all_masks, dim=1).reshape(batch_size*num_frames, -1, h, w)

    def calc_flows(self, out, src, alphas):
        src, tgt = prepare_data(out, src)
        src_ = img_scaling(src)
        tgt_ = img_scaling(tgt)
        with torch.no_grad():
            _, flow = self.model(src_[:, 0], src_[:, 1], iters=20, test_mode=True)
            src_data = get_metrics_value(flow.detach().cpu().numpy())
            _, flow = self.model(tgt_[:, 0], tgt_[:, 1], iters=20, test_mode=True)
            tgt_data = get_metrics_value(flow.detach().cpu().numpy())

        outputs = {}
        outputs['alpha'] = alphas.astype(float).item()
        for key, value in src_data.items():
            outputs[key] = tgt_data[key] / value

        return outputs

    def calc_avgdelta(self, out, src):
        src, tgt = prepare_data(out, src)
        with torch.no_grad():
            tracks_src = self.im2tracks(img_scaling(src))
            avg_delta_src = (tracks_src[:, 1:] - tracks_src[:, :-1]).mean()

            tracks_tgt = self.im2tracks(img_scaling(tgt))
            avg_delta_tgt = (tracks_tgt[:, 1:] - tracks_tgt[:, :-1]).mean()

        return avg_delta_src, avg_delta_tgt
    
    def flow_vis(self, frames):
        # n*3*h*w
        num_frames = frames.shape[0]
        vis = []
        with torch.no_grad():
            for t in range(num_frames-1):
                _, flow = self.model(frames[t], frames[t+1], iters=self.num_iters, test_mode=True)
                img = frames[t+1].squeeze().permute(1,2,0).cpu().detach().numpy()
                flo = flow.squeeze().permute(1,2,0).cpu().detach().numpy()

                flo = flow_viz.flow_to_image(flo)
                img_flo = np.concatenate([img, flo], axis=0)
                vis.append(img_flo[:, :, [2,1,0]])

            # last_img = np.concatenate([imgs[-1][0].permute(1,2,0).cpu().detach().numpy(), np.zeros_like(flo)], axis=0)
            # vis.append(last_img[:, :, [2,1,0]])
            vis = np.concatenate(vis, axis=1)
        return img2int(vis)

    def test_flow_vis(self, out, src):
        src, tgt = prepare_data(out, src)
        batchsize, _, _, h, w = src.shape
        random_idx = random.randint(0, batchsize-1)
        print('sampled: the image idx {} with a batchsize of {}'.format(random_idx, batchsize))
        src_ = src[random_idx]
        tgt_ = tgt[random_idx]
        vis_src = self.flow_vis(img_scaling(src_))
        vis_tgt = self.flow_vis(img_scaling(tgt_))
        src_imgs = concatenate_imgs(img_scaling(src_))
        tgt_imgs = concatenate_imgs(img_scaling(tgt_))
        # vis_src = self.calc_flow(self.img_scaling(src_))
        # vis_tgt = self.calc_flow(self.img_scaling(tgt_))
        return vis_src, vis_tgt, src_imgs, tgt_imgs

    def track_n_delta_vis(self, ims):
        num_frames, _, h, w = ims.shape
        with torch.no_grad():
            tracks = self.im2tracks(ims.unsqueeze(0))
            delta = tracks[:, 1:] - tracks[:, :-1]
            tracks = tracks.reshape(num_frames, 2*h, w).permute(0,2,1).reshape(num_frames*w,-1).permute(1,0).cpu().detach().numpy()
            delta = delta.reshape(num_frames-1, 2*h, w).permute(0,2,1).reshape((num_frames-1)*w,-1).permute(1,0).cpu().detach().numpy()
            delta_ = np.zeros_like(tracks)
            delta_[:, h:] = np.abs(delta)
            outputs = np.concatenate([tracks, 20*delta_], axis=0)[..., np.newaxis].repeat(3,-1)
        imgs_ = concatenate_imgs(ims)
        return img2int(np.concatenate([imgs_, outputs], axis=0))
    
    def color_vis(self, imgs):
        num_frames, _, h, w = imgs.shape
        with torch.no_grad():
            tracks = self.im2tracks(imgs.unsqueeze(0)).detach()
            colors = self.sample_color(tracks, imgs.unsqueeze(0))
            colors = colors.squeeze().permute(0,3,2,1).reshape(num_frames*w,h,3).permute(1,0,2).flip([-1]).cpu().detach().numpy()
        imgs_ = concatenate_imgs(imgs)
        return img2int(np.concatenate([imgs_, colors], axis=0))

    def toy_eg(self, out, src):
        src, tgt = prepare_data(out, src)
        flow = self.flow_vis(img_scaling(tgt))
        track_n_delta = self.track_n_delta_vis(img_scaling(tgt))
        colors = self.color_vis(img_scaling(tgt))
        
        return flow, track_n_delta, colors
        
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()        
        self.vgg = torch.nn.DataParallel(Vgg19()).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice_idxs = [2,7,12,21,30]
            
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(self.slice_idxs[0]):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[0], self.slice_idxs[1]):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[1], self.slice_idxs[2]):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[2], self.slice_idxs[3]):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(self.slice_idxs[3], self.slice_idxs[4]):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

def check_video_mag_loss(src_path, tgt_path, alpha, mm):
    from mm_utils import read_video
    src, _ = read_video(src_path)
    tgt, _ = read_video(tgt_path)
    with torch.no_grad():
        src = torch.stack(src).permute(0,3,1,2).to(torch.float).unsqueeze(0).cuda() / 255
        tgt = torch.stack(tgt).permute(0,3,1,2).to(torch.float).unsqueeze(0).cuda() / 255
        delta_src, _ = mm.cal_delta_n_color(src[:, :50], True)
        delta_tgt, _ = mm.cal_delta_n_color(tgt[:, :50], True)
        torch.cuda.empty_cache()
        mag_loss = mm.cal_mm_loss(delta_tgt, batch_multiplication(alpha, delta_src))
    print(' '*4+'src video: {}, tgt video: {}, alpha: {}, mag loss: {}'.format(src_path.split('/')[-2], tgt_path.split('/')[-2], alpha, mag_loss))
    print(' '*4+'max src delta: {}, max tgt delta: {}'.format(delta_src.max().item(), delta_tgt.max().item()))
    print(' '*4+'99.9% src delta: {}, 99.9% tgt delta: {}'.format(np.percentile(delta_src.cpu().detach().numpy(), 99.9), np.percentile(delta_tgt.cpu().detach().numpy(), 99.9)))

def check_delta(path, mm, path_type):
    from mm_utils import read_images, read_video
    if path_type == 'image':
        images = read_images(path)
    elif path_type == 'video':
        images, _ = read_video(path)
    else:
        raise NotImplementedError

    with torch.no_grad():
        images = torch.stack(images).permute(0,3,1,2).to(torch.float).unsqueeze(0).cuda() / 255
        delta, _ = mm.cal_delta_n_color(images, True)
    # for i in range(delta.shape[0]):
    #     print('frame {} and {}: 99.9% track {}, 0.01% track {}'.format(i, i+1, np.percentile(delta[i].cpu().detach().numpy(), 99.9), np.percentile(delta[i].cpu().detach().numpy(), 0.01)))
    #     print('frame {} and {}: max track {}, min track {}'.format(i, i+1, delta[i].cpu().detach().numpy().max(), delta[i].cpu().detach().numpy().min()))
    delta = delta.detach().cpu().abs()
    data = {'median': torch.median(delta), 'avg': torch.mean(delta), 'max': np.percentile(np.abs(np.array(delta)), 99.9)}
    return data

def check_img_flow(ims, mm):

    # image_paths = sorted(glob.glob('/n/owens-data1/mnt/big2/data/panzy/flavr/saved_models_final/youtube_vos_mm/0208_loss_img_test_gpu2/vis/loss_test/*.png'))[-1000:]
    # for path in image_paths:
    #     images = torch.tensor(cv2.imread(path).astype(np.float32)).permute(2,0,1).unsqueeze(0).unsqueeze(0).cuda() / 255
    #     print('Filename: {}'.format(os.path.basename(path)))
    #     a, b, c = check_img_flow(images, model)
    # ims = torch.cat(ims.split(256, -1), 0)

    with torch.no_grad():
        # print(t)
        _, flow = mm.model(ims[0], ims[1], iters=20, test_mode=True)
        # flow = torch.ones([batch_size, 2, h, w])
        flow = flow.detach().cpu().numpy()
    
    max_flow0 = np.percentile(np.abs(np.array(flow)), 99.9)
    max_flow1 = np.percentile(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2), 99.9)
    min_flow0 = np.percentile(np.abs(np.array(flow)), 0.1)
    min_flow1 = np.percentile(np.sqrt(flow[:, 0]**2 + flow[:, 1]**2), 0.1)
    avg_flow = flow.mean()
    print('abs flow: {}, {};\ntop sqrt of flow: {}, {};\navg flow: {}'.format(max_flow0, min_flow0, max_flow1, min_flow1, avg_flow))
    return max_flow0, max_flow1, avg_flow

# def check_video_flow(video_path, mm):
#     from mm_utils import read_video
#     frames, _ = read_video(video_path)
#     data = {'median': [], 'avg': [], 'max': []}
#     with torch.no_grad():
#         for t in range(len(frames) - 1):
#             _, flow = mm.model(frames[0].permute(2,0,1).unsqueeze(0).cuda(), frames[t+1].permute(2,0,1).unsqueeze(0).cuda(), iters=20, test_mode=True)
#             flow_ = flow.detach().cpu().abs()
#             data['median'].append(torch.median(flow_))
#             data['avg'].append(torch.mean(flow_))
#             data['max'].append(np.percentile(np.abs(np.array(flow_)), 99.9))
#     return data


def check_video_flow(video_path, mm):
    from mm_utils import read_video
    frames, _ = read_video(video_path)
    flows = []
    with torch.no_grad():
        for t in range(len(frames) - 1):
            _, flow = mm.model(frames[0].permute(2,0,1).unsqueeze(0).cuda(), frames[t+1].permute(2,0,1).unsqueeze(0).cuda(), iters=20, test_mode=True)
            flows.append(flow.detach().cpu().numpy())
    flows = np.stack(flows)
    return flows

def get_metrics_value(input_data, use_abs=True):
    # if use_abs:
    input_data = np.abs(input_data)
    from scipy.stats import hmean, gmean, mode
    output = {'median': np.median(input_data).astype(float),
              'avg': np.mean(input_data).astype(float),
              'max': np.max(input_data).astype(float),
              '99.99th': np.percentile(input_data, 99.99).astype(float),
              '99.9th': np.percentile(input_data, 99.9).astype(float),
              '99th': np.percentile(input_data, 99).astype(float),
              '95th': np.percentile(input_data, 95).astype(float),
              '90th': np.percentile(input_data, 90).astype(float),
              '80th': np.percentile(input_data, 80).astype(float),
              '70th': np.percentile(input_data, 70).astype(float),
              '60th': np.percentile(input_data, 60).astype(float)}
    return output
# def img_alpha_plot(dataloader, model, mm):

# def dataset_alpha_plot(data_root, mm):
#     pass

def video_alpha_plot(original_video, magnified_video_root, mm):
    # original_data = check_delta(original_video, mm, 'video')
    original_data = get_metrics_value(check_video_flow(original_video, mm), True)
    all_magnified_videos = sorted(glob.glob(os.path.join(magnified_video_root, '*.mp4')), key=lambda x: float(os.path.basename(x).split('x')[-1].split('_')[0]))
    outputs = {'alpha': []}
    for key, _ in original_data.items():
        outputs[key] = []
    import tqdm
    for path in tqdm.tqdm(all_magnified_videos):
        # data = check_delta(path, mm, 'video')
        data = get_metrics_value(check_video_flow(path, mm), True)
        outputs['alpha'].append(int(float(os.path.basename(path).split('x')[-1].split('_')[0])))
        for key, value in original_data.items():
            outputs[key].append(data[key] / value)
        # outputs['median'].append(data['median'] / original_data['median'])
        # outputs['avg'].append(data['avg'] / original_data['avg'])
        # outputs['max'].append(data['max'] / original_data['max'])
        # outputs['99.99th'].append(data['99.99th'] / original_data['99.99th'])
        # outputs['99.9th'].append(data['99.9th'] / original_data['99.9th'])
        # outputs['99th'].append(data['99th'] / original_data['99th'])
        # outputs['95th'].append(data['95th'] / original_data['95th'])
        # outputs['90th'].append(data['90th'] / original_data['90th'])
        # outputs['80th'].append(data['80th'] / original_data['80th'])
        # outputs['mode'].append(data['mode'] / original_data['mode'])
        # outputs['hmean'].append(data['hmean'] / original_data['hmean'])
        # outputs['gmean'].append(data['gmean'] / original_data['gmean'])
    
    return outputs


def align_loss_sanity_check():
    import argparse
    import cv2
    
    args = argparse.Namespace()
    args.mm_loss = 'L1'
    args.color_weight = 1e3
    args.mag_weight = 1e3
    args.alpha = 1
    args.model_type = 'raft'
    args.model_dir = '/home/panzy/FLAVR/utils/raft-sintel.pth'
    args.perceptual = True
    args.enable_reverse_color = False
    args.enable_reverse_mm = False
    args.enable_align_loss = True
    args.perceptual_weight = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # args.device_id = '0'
    model = MM(args)
    
    import sys
    sys.path.insert(1, '/home/panzy/FLAVR')

    single_tgt = torch.tensor(cv2.imread('/home/panzy/toy1/crop1.jpg').astype(np.float32)).permute(2,0,1).unsqueeze(0).unsqueeze(0).cuda()
    single_src = torch.tensor(cv2.imread('/home/panzy/toy1/crop2.jpg').astype(np.float32)).permute(2,0,1).unsqueeze(0).unsqueeze(0).cuda()
    out1 = model.cal_align_loss(single_tgt, single_src)
    out2 = model.cal_align_loss(single_tgt, single_tgt)
    out3 = model.cal_align_loss(single_src, single_src)
    print(out1, out2, out3)

def vis(num_samples, data_root, save_root):
    
    from PIL import Image
    from torchvision import transforms
    import torchvision.transforms.functional as F
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor()
    ])
    # -------------------------------toy example----------------------------------------
    # image_paths = sorted(glob.glob(os.path.join('/home/panzy/FLAVR/toy', '*')))
    # dataset_name = 'toy'
    # folder_name = 'dog'
    # save_root = '/home/panzy/image_test'
    # os.makedirs(os.path.join(save_root, dataset_name), exist_ok=True)
    # images = []
    # for path in image_paths:
    #     images.append(img_transforms(Image.open(path)))
    # images = torch.stack(images).cuda()
    # colors = model.color_vis(img_scaling(images))
    # track_n_delta = model.track_n_delta_vis(img_scaling(images))
    # _, mask = model.get_occlusion_mask(img_scaling(images.unsqueeze(0)))
    # mask = mask.permute(0,3,2,1).reshape(-1,256,1).permute(1,0,2).cpu().detach().numpy()
    # mask = img2int(img_scaling(torch.tensor(np.concatenate([concatenate_imgs(images), concatenate_imgs(images)*mask.repeat(3,-1), mask.repeat(3,-1)], axis=0))).numpy())

    # cv2.imwrite(os.path.join(save_root, dataset_name, '{}_track_n_delta.png'.format(folder_name)), track_n_delta)
    # cv2.imwrite(os.path.join(save_root, dataset_name, '{}_colors.png'.format(folder_name)), colors)
    # cv2.imwrite(os.path.join(save_root, dataset_name, '{}_mask.png'.format(folder_name)), mask)

    # -------------------------------dataset samples----------------------------------------
    # num_samples = 10
    # data_root = '/datac/panzy/youtube-vos-mm3'
    split = 'train'
    # save_root = '/home/panzy/image_test'
    folders = glob.glob(os.path.join(data_root, split, '*'))
    dataset_name = data_root.split('/')[-1]
    os.makedirs(os.path.join(save_root, dataset_name), exist_ok=True)
    for i in range(num_samples):
        folder_name = folders[i].split('/')[-1]
        image_paths = sorted(glob.glob(os.path.join(folders[i], '*')))
        images = []
        for path in image_paths:
            images.append(img_transforms(Image.open(path)))
        images = torch.stack(images).cuda()
        colors = model.color_vis(img_scaling(images))
        track_n_delta = model.track_n_delta_vis(img_scaling(images))
        _, mask = model.get_occlusion_mask(img_scaling(images.unsqueeze(0)))
        mask = mask.permute(0,3,2,1).reshape(5*256,256,1).permute(1,0,2).cpu().detach().numpy()
        mask = img2int(img_scaling(torch.tensor(np.concatenate([concatenate_imgs(images), concatenate_imgs(images)*mask.repeat(3,-1), mask.repeat(3,-1)], axis=0))).numpy())

        cv2.imwrite(os.path.join(save_root, dataset_name, '{}_track_n_delta.png'.format(folder_name)), track_n_delta)
        cv2.imwrite(os.path.join(save_root, dataset_name, '{}_colors.png'.format(folder_name)), colors)
        cv2.imwrite(os.path.join(save_root, dataset_name, '{}_mask.png'.format(folder_name)), mask)


if __name__ == "__main__":
    # import sys
    # sys.path.append('.')
    # import config
    # args, unparsed = config.get_args()
    # loss_function = MM(args)
    import argparse
    import glob
    import cv2
    
    args = argparse.Namespace()
    args.mm_loss = 'L1'
    args.color_loss = 'L1'
    args.color_weight = 1
    args.mag_weight = 1
    args.alpha = 1
    args.model_type = 'raft'
    args.model_dir = '/home/panzy/FLAVR/utils/raft-sintel.pth'
    args.perceptual = True
    args.enable_reverse_color = False
    args.enable_reverse_mm = False
    args.enable_align_loss = True
    args.gaussian_blur = False
    args.flexible_alpha_training = False
    args.perceptual_weight = 1
    args.enable_corrwise = False
    args.num_iters = 20
    args.nbr_frame = 2
    args.enable_cycle_check = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    # args.device_id = '0'
    model = MM(args)
    
    import sys
    sys.path.insert(1, '/home/panzy/FLAVR')

    import json
    original_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/baby_20.mp4'
    dm_video_root = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference_all_alphas/baby'
    our_video_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/all_alpha_videos/0328_flexible_alpha_mm3_test32_6/baby_20'
    # original_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/bookshelf_50.mp4'
    # dm_video_root = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference_all_alphas/bookshelf'
    # our_video_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/all_alpha_videos/0326_flexible_alpha_mm3_test32_5/bookshelf'
    # original_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/cropped_tuningfork.mp4'
    # dm_video_root = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference_all_alphas/cropped_tuningfork'
    # our_video_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/all_alpha_videos/0326_flexible_alpha_mm3_test32_5/tuningfork'
    print(os.path.basename(original_video))
    with open('/home/panzy/alpha_test/dm_baby.json', 'r') as f:
        dm_outputs = json.load(f)
    # dm_outputs = video_alpha_plot(original_video, dm_video_root, model)
    # with open('/home/panzy/alpha_test/dm_baby.json', 'w') as f:
    #     json.dump(dm_outputs, f, indent=4)
    our_outputs = video_alpha_plot(original_video, our_video_root, model)
    with open('/home/panzy/alpha_test/0328_32_6_baby.json', 'w') as f:
        json.dump(our_outputs, f, indent=4)
    
    import matplotlib.pyplot as plt
    all_keys = list(dm_outputs.keys())[1:]
    plt.figure(figsize=(5*np.ceil(len(all_keys)/2),10))
    for i in range(len(all_keys)):
        plt.subplot(2, np.ceil(len(all_keys)/2).astype(int), i+1)
        key = all_keys[i]
        plt.title(key)
        plt.scatter(dm_outputs['alpha'], dm_outputs[key], label='deep mag')
        plt.scatter(our_outputs['alpha'], our_outputs[key], label='ours')
        plt.plot(np.arange(75), np.arange(75))
        plt.grid()
        plt.legend()
    plt.savefig('/home/panzy/alpha_test/0328_32_6_baby_comp.png')

    # plt.subplot(2, 5, 1)
    # plt.title('median')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['median'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['median'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 2)
    # plt.title('mean')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['avg'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['avg'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 3)
    # plt.title('max')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['max'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['max'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 4)
    # plt.title('99.99th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['99.99th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['99.99th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 5)
    # plt.title('99.9th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['99.9th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['99.9th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 6)
    # plt.title('99th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['99.9th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['99.9th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 7)
    # plt.title('95th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['99.9th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['99.9th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

    # plt.subplot(2, 5, 8)
    # plt.title('90th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['99.9th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['99.9th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()
    
    # plt.subplot(2, 5, 9)
    # plt.title('80th percentile')
    # plt.scatter(dm_outputs['alpha'], dm_outputs['80th'], label='deep mag')
    # plt.scatter(our_outputs['alpha'], our_outputs['80th'], label='ours')
    # plt.plot(np.arange(75), np.arange(75))
    # plt.grid()
    # plt.legend()

# # -------------------------------check masks and warped images----------------------------------------
# vis(10, '/datac/panzy/youtube-vos-mm3', '/home/panzy/image_test')

# # # -------------------------------check flow for videos----------------------------------------
#     video_list = ['bookshelf_50', 'baby_20']
#     alpha_list = [1, 20, 50]
#     root_list = ['/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos', '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference', '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0326_flexible_alpha_mm3_test32_5_e470', '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0326_flexible_alpha_mm3_test32_5']
#     for video in video_list:
#         for alpha in alpha_list:
#             original_path = os.path.join(root_list[0], video+'.mp4')
#             print('{}.mp4 (x{})'.format(video, alpha))
#             data0 = check_video_flow(original_path, model)

#             deep_mag_path = os.path.join(root_list[1], '{}_x{}_static.mp4'.format(video, float(alpha)))
#             data1 = check_video_flow(deep_mag_path, model)
            
#             our_path = os.path.join(root_list[2], '{}_x{}_static.mp4'.format(video, alpha))
#             data2 = check_video_flow(our_path, model)

#             our_path = os.path.join(root_list[3], '{}_x{}_static.mp4'.format(video, alpha))
#             data3 = check_video_flow(our_path, model)

#             for i in range(len(data0['median'])):
#                 print('frame {}:'.format(i+1))
#                 print('median:')
#                 print('     frame {}: original video: {:.4f}, deep mag video: {:.4f}, epoch 470: {:.4f}, epoch 2000: {:.4f}'.format(i+1, data0['median'][i], data1['median'][i], data2['median'][i], data3['median'][i]))

#             # for i in range(len(data0['median'])):
#                 print('avg:')
#                 print('     frame {}: original video: {:.4f}, deep mag video: {:.4f}, epoch 470: {:.4f}, epoch 2000: {:.4f}'.format(i+1, data0['avg'][i], data1['avg'][i], data2['avg'][i], data3['avg'][i]))

#             # for i in range(len(data0['median'])):
#                 print('99.9% percentile:')
#                 print('     frame {}: original video: {:.4f}, deep mag video: {:.4f}, epoch 470: {:.4f}, epoch 2000: {:.4f}'.format(i+1, data0['max'][i], data1['max'][i], data2['max'][i], data3['max'][i]))

# # # -------------------------------check flow for videos----------------------------------------


    # video_path = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/bookshelf_50.mp4'
    # print('bookshelf.mp4 (x20)\noriginal video:')
    # check_video_flow(video_path, model)
    # video_path = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference/bookshelf_50_x20.0_static.mp4'
    # print('deep mag inference video:')
    # check_video_flow(video_path, model)
    # video_path = '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0326_flexible_alpha_mm3_test32_5/bookshelf_50_x20_static.mp4'
    # print('our inference video:')
    # check_video_flow(video_path, model)

    # video_path = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/baby_20.mp4'
    # print('original video:')
    # check_video_flow(video_path, model)
    # video_path = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference/baby_20_x20.0_static.mp4'
    # print('deep mag inference video:')
    # check_video_flow(video_path, model)
    # video_path = '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0326_flexible_alpha_mm3_test32_5/baby_20_x20_static.mp4'
    # print('our inference video:')
    # check_video_flow(video_path, model)

    # im1 = torch.tensor(cv2.imread('/home/panzy/toy2/crop1.jpg').astype(float)/255).permute(2,0,1).unsqueeze(0).cuda()
    # im2 = torch.tensor(cv2.imread('/home/panzy/toy2/crop2.jpg').astype(float)/255).permute(2,0,1).unsqueeze(0).cuda()
    # mask0, mask1 = model.get_occlusion_mask(torch.stack([im1, im2], dim=1).to(torch.float))

# -------------------------------check mag loss for video----------------------------------------
    # src_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/test_videos/bookshelf_50.mp4'
    # tgt_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0313_flexible_alpha_mm3_test39/bookshelf_50_x20_static.mp4'
    # deep_mag_video = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference/bookshelf_50_x20.0_static.mp4'
    # check_video_mag_loss(src_video, tgt_video, 20., model)
    # check_video_mag_loss(src_video, deep_mag_video, 20., model)

# ------------------------------find most similar video----------------------------------------
    # src_delta = check_delta(src_video, model, 'video')
    
    # our_video_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/all_alpha_videos/0305_flexible_alpha_mm3_test32'
    # our_video_paths = sorted(glob.glob(os.path.join(our_video_root, 'baby*.mp4')), key=lambda x: int(os.path.basename(x).split('x')[-1].split('_')[0]))
    # best_alpha = None
    # min_diff = float('inf')
    # for path in our_video_paths:
    #     alpha = os.path.basename(path).split('x')[-1].split('_')[0]
    #     tgt_delta = check_delta(path, model, 'video')
    #     diff = torch.sqrt(((src_delta - tgt_delta) ** 2)).mean()
    #     print('alpha: {}, diff: {}'.format(alpha, diff))
    #     if diff < min_diff:
    #         min_diff = diff
    #         best_alpha = alpha
    # print('best alpha: {}, min diff: {}'.format(best_alpha, min_diff))

# -----------------------------------------------------------------------------------
    # # tgt_video = '/n/owens-data1/mnt/big2/data/panzy/flavr/saved_videos/0124_flexible_alpha/baby_x20_static.mp4'
    # deep_mag_video = '/n/owens-data1/mnt/big2/data/panzy/deep_mag_inference/baby_20_x20.0_static.mp4'
    # alpha = 20.
    # print('Deep mag inference, alpha=20:')
    # check_video_mag_loss(src_video, deep_mag_video, alpha, model)

    # our_video_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/all_alpha_videos/0305_flexible_alpha_mm3_test32'
    # our_video_paths = sorted(glob.glob(os.path.join(our_video_root, 'baby*.mp4')), key=lambda x: int(os.path.basename(x).split('x')[-1].split('_')[0]))
    # for path in our_video_paths:
    #     alpha = os.path.basename(path).split('x')[-1].split('_')[0]
    #     print('Our inference, alpha={}:'.format(alpha))
    #     check_video_mag_loss(src_video, path, 20., model)

# -----------------------------check flow, track, delta, and sampled color for toy example----------------------------------------------
    # from dataset.toy import get_toy_loader
    # vis_loc = '/home/panzy/FLAVR'
    # data_root = '/home/panzy/toy1'
    # test_loader = get_toy_loader(data_root, args, 1, shuffle=False, num_workers=4)
    # for images, _ in test_loader:
    #     images = [img_.cuda() for img_ in images]
    #     flow = model.flow_vis(img_scaling(torch.stack(images, dim=1)))
    #     track_n_delta = model.track_n_delta_vis(img_scaling(torch.stack(images, dim=1)))
    #     colors = model.color_vis(img_scaling(torch.stack(images, dim=1)))
    #     cv2.imwrite(os.path.join(vis_loc, 'flow_test.jpg'), flow)
    #     cv2.imwrite(os.path.join(vis_loc, 'track_n_delta_test.jpg'), track_n_delta)
    #     cv2.imwrite(os.path.join(vis_loc, 'colors_test.jpg'), colors)


    # images = torch.load('/home/panzy/tgt.pt')
    # print(images.shape, images.max(), images.min())
    # flow = model.flow_vis(images)
    # cv2.imwrite(os.path.join(vis_loc, 'flow_test.jpg'), flow)

    # from dataset.deep_mag import get_loader
    # vis_loc = '/home/panzy/FLAVR'
    # data_root = '/n/owens-data1/mnt/big2/data/panzy/flavr/a10'
    # test_loader = get_loader('test', data_root, True, 2, shuffle=False, num_workers=4) 
    # for images, _ in test_loader:
    #     images = [img_.cuda() for img_ in images]
    #     flow = model.flow_vis(model.img_scaling(torch.stack(images)))
    #     track_n_delta = model.track_n_delta_vis(model.img_scaling(torch.stack(images)))
    #     colors = model.color_vis(model.img_scaling(torch.stack(images)))
    #     cv2.imwrite(os.path.join(vis_loc, 'flow_test.jpg'), flow)
    #     cv2.imwrite(os.path.join(vis_loc, 'track_n_delta_test.jpg'), track_n_delta)
    #     # cv2.imwrite(os.path.join(vis_loc, 'delta_test.jpg'), delta)
    #     cv2.imwrite(os.path.join(vis_loc, 'colors_test.jpg'), colors)