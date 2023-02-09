import argparse
import os
import shutil
import setproctitle
from configobj import ConfigObj
from validate import Validator
from magnet import MagNet3Frames
from tqdm import tqdm
import torch
from pytorch_msssim import ssim_matlab as calc_ssim
import json
import tensorflow as tf


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
                    type=float, default=5,
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
        if args.phase == 'run':
            if int(args.alpha) == args.alpha:
                alpha = str(int(args.alpha))
            else:
                alpha = str(args.alpha).replace('.', '_')
            out_path = os.path.join(args.out_dir, '{}_x{}_{}.mp4'.format(os.path.basename(args.vid_path).split('.')[0], str(int(args.alpha)), mode))
            model.run(checkpoint,
                      args.vid_path,
                      out_path,
                      args.alpha,
                      args.velocity_mag)
        elif args.phase == 'eval':
            
            from deep_mag_test import get_loader
            test_loader = get_loader(256, args.data_root, 4)

            save_root = os.path.join(args.eval_dir, 'eval_results', args.exp_name)
            if not os.path.exists(save_root):
                # user_input = input('folder for experiment named {} already exists, press ENTER to delete:'.format(args.exp_name))
                # if user_input == '':
                #     shutil.rmtree(save_root)
                # else:
                #     exit()
                os.mkdir(save_root)

            for i, (images, gt_image, a, noise_factor) in enumerate(tqdm(test_loader)):
                images_ = [255*image.squeeze().permute(1,2,0) for image in images]
                gt_image_ = 255*gt_image[0].squeeze().permute(1,2,0)
                out_amp = model.inference(checkpoint,
                                          images_,
                                          a.item())
                ssim = calc_ssim(torch.tensor(out_amp) / 2 + 0.5, gt_image_.unsqueeze(0) / 255, val_range=1.).item()
                rmse = cal_rmse(torch.tensor(out_amp) / 2 + 0.5, gt_image_.unsqueeze(0) / 255).item()
                results_dict[i] = {
                    'noise_factor': noise_factor.item(),
                    'ssim': ssim,
                    'rmse': rmse
                }
            
            result_fn = os.path.join(args.eval_dir, 'eval_results', args.exp_name, 'deep_mag_data.json')
            with open(result_fn, 'w') as fp:
                json.dump(results_dict, fp, indent=4)
            print('save evaluation results to', result_fn)
        else:
            raise ValueError('Invalid phase argument. '
                             'Expected ["eval", "run"], '
                             'got ' + args.phase)


if __name__ == '__main__':
    main(arguments)
