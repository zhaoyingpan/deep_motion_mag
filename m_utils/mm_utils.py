import os
import math
import numpy as np
import glob
import torch
import cv2

def lorentzian_function(src, tgt, s=1):
    return torch.log(1 + s * (src - tgt)**2).mean()


def prepare_data(out, src, second_frame=None):
    # out: b*n-1*3*H*W, src: b*m*3*H*W

    if second_frame is not None:
        tgt = torch.stack(list(torch.unbind(src[:, :second_frame], dim=1))+list(torch.unbind(out, dim=1))+list(torch.unbind(src[:, second_frame+1:], dim=1)), dim=1)
        return src, tgt
    else:
        if src.shape[1] - 1 == out.shape[1]:
            tgt = torch.stack([src[:, 0]]+list(torch.unbind(out, dim=1)), dim=1)
            return src, tgt

        elif src.shape[1] == out.shape[1]:
            return src, out
        else:
            raise Exception('the shapes of the source images and target images are not matched!')
    
# def prepare_data(out, src):
#     # # out: n-1*b*3*H*W, src: [1*b*3*H*W]*n
#     # out: b*n-1*3*H*W, src: b*n*3*H*W

#     if src.shape[1] - 1 == out.shape[1]:
#         tgt = torch.stack([src[:, 0].squeeze()]+list(torch.unbind(out, dim=1)), dim=1)
#         return src, tgt

#     elif src.shape[1] == out.shape[1]:
#         return src, out
#     else:
#         raise Exception('the shapes of the source images and target images are not matched!')
    # src = (255*(torch.clamp(src, 0, 1))).to(torch.int).cuda()
    # tgt = (255*(torch.clamp(tgt, 0, 1))).to(torch.int).cuda()

def reverse_frames(ims):
    return ims.flip([1])

def img_scaling(ims):
    return 255*(torch.clamp(ims, 0., 1.))

def img2int(ims):
    return np.clip(ims, 0., 255.).astype(np.int)

def concatenate_imgs(imgs):
    # n*3*h*w
    num_frames, _, h, w = imgs.shape
    return imgs.permute(0,3,2,1).reshape(num_frames*w, h, -1).permute(1,0,2).flip([-1]).cpu().detach().numpy()

def alpha_embedding(alphas, num_frames, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param alphas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    N = alphas.shape[0]
    dim = 32*32
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=alphas.device)
    args = alphas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding.reshape(-1, 32, 32).unsqueeze(1).unsqueeze(1).repeat(1, 1, num_frames, 1, 1)

def batch_multiplication(alpha, batch_data):
    """
    alpha: batch_size, or 1
    batch_data: batch_size * num_frames * 2 * H * W (2 for x-axis and y-axis tracks)
    """
    
    if isinstance(alpha, float):
        return alpha * batch_data
    else:
        return alpha.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * batch_data

def read_images(image_root):
    paths = sorted(glob.glob(os.path.join(image_root, '*')))
    images = []
    for path in paths:
        images.append(torch.tensor(cv2.imread(path)))
    return images

def read_video(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()

    frames = []
    while ret:
        frames.append(torch.tensor(frame))
        ret, frame = cap.read()
    return frames, fps

def write_video(frames, fps, output_path):
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w,h))
    for frame in frames:
        writer.write(frame)

    writer.release()

if __name__ == "__main__":
    alpha = (50 - 1) * torch.rand(5) + 1
    print(alpha, alpha_embedding(alpha, 2).shape)