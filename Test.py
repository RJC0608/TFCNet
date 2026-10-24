import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse, cv2
# from scipy import misc
import time
import imageio
from model.TFCNet import TFCNet
from utils1.data import test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
argument = parser.add_argument('--testsize', type=int, default=384, help='testing size')
args = parser.parse_args()
opt = args

dataset_path = './TSOD10K/Test/'

model = TFCNet()
model.load_state_dict(torch.load('./TFCNet/Net_epoch_best.pth'))

model.cuda()
model.eval()

test_datasets = ['']

for dataset in test_datasets:
    save_path = './results/TFCNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/image/'
    print(dataset)
    gt_root = dataset_path + dataset + '/mask/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()

        s1, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(image)

        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.interpolate(s1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path+name, res * 255)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))


