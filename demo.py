import os
import torch
import argparse
from PIL import Image
from models.mabdt import Model as MABDT
from pytorch_msssim import ssim
from utils.utils import expand2square
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms.functional as ttf


def demo(args):

    model_restoration = MABDT().cuda() if args.cuda else MABDT()

    if args.cuda:
        model_restoration.load_state_dict(torch.load(args.resume_state))
    else:
        model_restoration.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))
    model_restoration.eval()

    input_ = ttf.to_tensor(Image.open(args.input_image).convert('RGB'))
    target = ttf.to_tensor(Image.open(args.target_image).convert('RGB'))
    name = args.target_image.split('/')[-1]

    with torch.no_grad():

        input_ = input_.unsqueeze(0).cuda() if args.cuda else input_.unsqueeze(0)
        target = target.unsqueeze(0).cuda() if args.cuda else target.unsqueeze(0)

        _, _, h, w = input_.shape
        input_, mask = expand2square(input_, factor=args.expand_factor)  # to square multiplier
        if args.only_last:
            restored_ = model_restoration(input_, only_last=args.only_last)
        else:
            restored_, _,_, _ = model_restoration(input_)
        restored_ = torch.masked_select(restored_, mask.bool()).reshape(1, 3, h, w)

        restored = restored_.clamp_(-1, 1)
        restored = restored * 0.5 + 0.5
        target = target * 0.5 + 0.5

        # metrics
        mse_value = F.mse_loss(restored, target)
        psnr_value = 10 * torch.log10(1 / mse_value).item()
        _, _, H, W = restored.size()
        down_ratio = max(1, round(min(H, W) / 256))
        ssim_value = ssim(F.adaptive_avg_pool2d(restored, (int(H / down_ratio), int(W / down_ratio))),
                        F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                        data_range=1, size_average=False).item()

        print(f'Current Image: {args.input_image}')
        print('--------------------------------------')
        print('Current Metrics: \nPSNR: {:.3f}, \nSSIM: {:.5f}, \nMSE: {:.5f}'.format(psnr_value, ssim_value, mse_value))

        if args.result_save:
            save_image(restored_, os.path.join(args.result_dir, name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='./data/hazy/1.png', help='path of the input hazy image')
    parser.add_argument('--target_image', type=str, default='./data/clear/1.png', help='path of the ground truth')
    parser.add_argument('--result_dir', type=str, default='./data/result', help='path to save the result')
    parser.add_argument('--expand_factor', type=int, default=128, help='expand input to a square multiplier')
    parser.add_argument('--result_save', type=bool, default=True, help='wether to save the result')
    parser.add_argument('--resume_state', type=str, default='./Haze1k_moderate/model_best.pth', help='path of the pre-trained model')
    parser.add_argument('--only_last', type=bool, default=True, help='only output the final result')
    parser.add_argument('--cuda', type=bool, default=True, help='do you have cuda?')
    args = parser.parse_args()

    demo(args)
