import argparse
import os
import shutil
from scipy.stats import gaussian_kde
import setproctitle
from configobj import ConfigObj
from validate import Validator
from magnet import MagNet3Frames
from tqdm import tqdm
import torch
from pytorch_msssim import ssim_matlab as calc_ssim
import json
import tensorflow as tf
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train',
                    help='train, test, run, interactive')
parser.add_argument('--config_file', dest='config_file', default='configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf',
                    help='path to config file')
parser.add_argument('--config_spec', dest='config_spec',
                    default='configs/configspec.conf',
                    help='path to config spec file')
# for inference
parser.add_argument('--vid_dir', dest='vid_dir', default=None,
                    help='Video folder to run the network on.')
parser.add_argument('--frame_ext', dest='frame_ext', default='png',
                    help='Video frame file extension.')
parser.add_argument('--out_dir', dest='out_dir', default=None,
                    help='Output folder of the video run.')
parser.add_argument('--alpha',
                    type=float, default=None,
                    help='Magnification factor for inference.')
parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true',
                    help='Whether to do velocity magnification.')
# For temporal operation.
parser.add_argument('--fl', dest='fl', type=float,
                    help='Low cutoff Frequency.')
parser.add_argument('--fh', dest='fh', type=float,
                    help='High cutoff Frequency.')
parser.add_argument('--fs', dest='fs', type=float,
                    help='Sampling rate.')
parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int,
                    help='Number of filter tap required.')
parser.add_argument('--filter_type', dest='filter_type', type=str,
                    help='Type of filter to use, must be Butter or FIR.')
parser.add_argument('--data_root', type=str)
parser.add_argument('--vid_path', type=str)
parser.add_argument('--exp_name', type=str)
parser.add_argument('--eval_dir', type=str)
parser.add_argument('--device_id', type=str, default="0")
parser.add_argument('--save_outputs', action='store_true')


arguments = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = arguments.device_id

def cal_rmse(img1, img2):
    return torch.sqrt(torch.square(img1 - img2).mean())

def main(args):
    
    configspec = ConfigObj(args.config_spec, raise_errors=True)
    config = ConfigObj(args.config_file,
                       configspec=configspec,
                       raise_errors=True,
                       file_error=True)
    # Validate to get all the default values.
    config.validate(Validator())
    # if not os.path.exists(config['exp_dir']):
    #     # checkpoint directory.
    #     os.makedirs(os.path.join(config['exp_dir'], 'checkpoint'))
    #     # Tensorboard logs directory.
    #     os.makedirs(os.path.join(config['exp_dir'], 'logs'))
    #     # default output directory for this experiment.
    #     os.makedirs(os.path.join(config['exp_dir'], 'sample'))
    network_type = config['architecture']['network_arch']
    exp_name = config['exp_name']
    mode = 'dynamic' if args.velocity_mag else 'static'

    setproctitle.setproctitle('{}_{}_{}' \
                              .format(args.phase, network_type, exp_name))
    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                              log_device_placement=False)
    tfconfig.gpu_options.allow_growth = True
    
    results_dict = {}
    with tf.compat.v1.Session(config=tfconfig) as sess:
        model = MagNet3Frames(sess, exp_name, config['architecture'])
        checkpoint = config['training']['checkpoint_dir']
        if args.phase == 'alpha_eval':
            
            mm_args = argparse.Namespace()
            mm_args.mm_loss = 'L1'
            mm_args.color_loss = 'L1'
            mm_args.color_weight = 1
            mm_args.mag_weight = 1
            mm_args.alpha = 1
            mm_args.model_type = 'raft'
            mm_args.model_dir = '/home/panzy/deep_motion_mag/m_utils/raft-sintel.pth'
            mm_args.perceptual = True
            mm_args.enable_reverse_color = False
            mm_args.enable_reverse_mm = False
            mm_args.enable_align_loss = True
            mm_args.gaussian_blur = False
            mm_args.flexible_alpha_training = False
            mm_args.perceptual_weight = 1
            mm_args.enable_corrwise = False
            mm_args.num_iters = 20
            mm_args.nbr_frame = 2
            mm_args.enable_cycle_check = True
            mm_args.data_root = '/n/owens-data1/mnt/big2/data/panzy/youtube-vos-mm2'
            mm_args.crop_type = 'center'
            mm_args.img_size = 256
            mm_args.min_alpha = 2
            mm_args.max_alpha = 50
            
            from m_utils.MM import MM
            criterion = MM(mm_args)
            
            from youtube_vos_mm import get_loader
            test_loader = get_loader('valid', mm_args, False)
            
            all_data = {}
            for i, (images, a) in enumerate(tqdm(test_loader)):
                images_ = [255*image.squeeze().permute(1,2,0) for image in images]
                out_amp = model.inference(checkpoint,
                                          images_,
                                          a.item())
                out_amp = torch.tensor((np.flip(out_amp, axis=-1) / 2 + 0.5)).permute(0,3,1,2).unsqueeze(1).cuda()
                images_ = torch.stack(images_, dim=0).permute(0,3,1,2).unsqueeze(0).cuda()/255
                data = criterion.calc_flows(out_amp, images_, a.cpu().numpy())

                for key, value in data.items():
                    if key not in list(all_data.keys()):
                        all_data[key] = [value]
                    else:
                        all_data[key].append(value)
            file_path = '/home/panzy/alpha_test/{}.json'.format(args.exp_name)
            print('saved to {}'.format(file_path))
            with open(file_path, 'w') as f:
                json.dump(all_data, f, indent=4)


            import matplotlib.pyplot as plt
            all_keys = list(all_data.keys())[1:]
            max_alpha = np.ceil(np.array(all_data['alpha']).max())

            plt.figure(figsize=(5*np.ceil(len(all_keys)/2),10))
            plt.suptitle(os.path.basename(file_path).split('.json')[0])
            for i in range(len(all_keys)):
                plt.subplot(2, np.ceil(len(all_keys)/2).astype(int), i+1)
                key = all_keys[i]
                
                x = all_data['alpha']
                y = all_data[key]
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy)

                plt.scatter(x, y, c=z, s=10, label=key)
                plt.title('{}, MSE: {}'.format(key, ((np.array(x) - np.array(y))**2).mean()))
                plt.xlabel('input alpha')
                plt.ylabel('actual alpha')
                # plt.scatter(all_data['alpha'], all_data[key], label='ours')
                plt.plot(np.arange(max_alpha), np.arange(max_alpha), color='r')
                plt.grid()
                plt.legend()
            plt.savefig('/home/panzy/alpha_test/{}.png'.format(args.exp_name))
                
        else:
            raise ValueError('Invalid phase argument. '
                             'Expected ["alpha_eval", "run"], '
                             'got ' + args.phase)


if __name__ == '__main__':
    main(arguments)
