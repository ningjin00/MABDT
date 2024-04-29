import os
import sys
import torch
import argparse
from tqdm import tqdm
from models.my_model import Model as EMPFNet
from utils.utils import expand2square
import torch.nn.functional as F
from pytorch_msssim import ssim
from datasets import MyTestDataSet
from torch.utils.data import DataLoader
from torchvision.utils import save_image


def test(args):

    model_restoration = EMPFNet().cuda() if args.cuda else EMPFNet()

    path_val_input, path_val_target = args.val_data + '/hazy/', args.val_data + '/clear/'
    datasetTest = MyTestDataSet(path_val_input, path_val_target)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=args.num_works, pin_memory=True)

    if args.cuda:
        model_restoration.load_state_dict(torch.load(args.resume_state))
    else:
        model_restoration.load_state_dict(torch.load(args.resume_state, map_location=torch.device('cpu')))
    model_restoration.eval()

    PSNR = 0
    SSIM = 0
    MSE = 0
    L = len(testLoader)

    with torch.no_grad():
        for index, (x, y, name) in enumerate(tqdm(testLoader, desc='Test !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_, target = (x.cuda(), y.cuda()) if args.cuda else (x, y)
            _, _, h, w = input_.shape
            input_, mask = expand2square(input_, factor=args.expand_factor)
            if args.only_last:
                restored_ = model_restoration(input_, only_last=args.only_last, mask=1-mask)
            else:
                restored_, _,_, _ = model_restoration(input_, only_last=args.only_last, mask=1-mask)
            restored_ = torch.masked_select(restored_, mask.bool()).reshape(1, 3, h, w)

            restored = restored_.clamp_(-1, 1)
            restored = restored * 0.5 + 0.5
            target = target * 0.5 + 0.5

            mse_value = F.mse_loss(restored, target)
            psnr_value = 10 * torch.log10(1 / mse_value).item()
            _, _, H, W = restored.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_value = ssim(F.adaptive_avg_pool2d(restored, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

            MSE += mse_value.item()
            PSNR += psnr_value
            SSIM += ssim_value

            if args.result_save:
                save_image(restored_, os.path.join(args.result_dir, name[0]))
        print('--------------------------------------')
        print(f'Current dataset: {args.val_data.split("/")[-2]}')
        print('--------------------------------------')
        print('Current Metrics: \nPSNR: {:.3f}, \nSSIM: {:.5f}, \nMSE: {:.5f}'.format(PSNR / L, SSIM / L, MSE / L))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--val_data', type=str, default='D:\dataset_GRS\RICE1\\test')
    parser.add_argument('--result_dir', type=str, default='E:\output_GRS\experience\Test')
    parser.add_argument('--resume_state', type=str, default='./RICE1_DF/model_best.pth')
    parser.add_argument('--expand_factor', type=int, default=128, help='expand input to a square multiplier')
    parser.add_argument('--result_save', type=bool, default=True, help='wether to save the result')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--only_last', type=bool, default=True, help='only output the final result')
    parser.add_argument('--num_works', type=int, default=0)
    args = parser.parse_args()

    test(args)
