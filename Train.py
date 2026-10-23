import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from model.TFCNet import TFCNet

from utils.utils import adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
import eval.metrics as Measure
import pytorch_iou
import pytorch_fm

from utils.clip_grad import clip_gradient
from utils.dataset import get_loader, test_dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

CE = torch.nn.BCEWithLogitsLoss()
IOU = pytorch_iou.IOU(size_average = True)

def DiceLoss(inputs, targets):
    eps = 1e-5
    dice_loss = []
    for i in range(inputs.size(0)):
        inter = (inputs[i] * targets[i]).sum()
        dice = (2 * inter + eps) / (inputs[i].sum() + targets[i].sum() + eps)
        dice_loss.append(1 - dice)
    return torch.mean(torch.tensor(dice_loss))

floss = pytorch_fm.FLoss()


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):

            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()

            # s1, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4, s5, s5_sig, edg5 = model(images)
            s1, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(images)
            # bce+iou+fmloss
            loss1 = CE(s1, gts) + IOU(s1_sig, gts) + floss(s1_sig, gts) #+ CE(edg1, gts)
            loss2 = CE(s2, gts) + IOU(s2_sig, gts) + floss(s2_sig, gts) #+ CE(edg2, gts)
            loss3 = CE(s3, gts) + IOU(s3_sig, gts) + floss(s3_sig, gts) #+ CE(edg3, gts)
            loss4 = CE(s4, gts) + IOU(s4_sig, gts) + floss(s4_sig, gts) #+ CE(edg4, gts)
            loss5 = CE(s5, gts) + IOU(s5_sig, gts) + floss(s5_sig, gts) #+ CE(edg5, gts)

            # loss = loss1 + loss2 + loss3 + loss4 + loss5

            loss_init = loss1 + loss2 + loss3
            loss_final = loss4 + loss5

            loss = loss_init + loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 200 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} '.
                        format(epoch, opt.epoch, i, total_step, loss.data))

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_metric_dict, best_score_mae, best_epoch, best_score
    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    WFM = Measure.WeightedFmeasure()
    MAE = Measure.MAE()
    metrics_dict = dict()

    model.eval()
    test_loss = 0
    step = 0
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, gt_tensor, _, _ = test_loader.load_data()
            gts = gt_tensor.unsqueeze(1)
            gts = gts.cuda()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            # s1, s1_sig, edg1, s2, s2_sig, edg2, s3, s3_sig, edg3, s4, s4_sig, edg4, s5, s5_sig, edg5 = model(image)
            s1, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(image)
            res = s1
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
            WFM.step(pred=res, gt=gt)
            MAE.step(pred=res, gt=gt)


        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(meanEm=EM.get_results()['em']['curve'].mean().round(4))
        metrics_dict.update(wFm=WFM.get_results()['wfm'].round(4))
        metrics_dict.update(mae=MAE.get_results()['mae'].round(4))

        cur_score_mae = metrics_dict['mae']
        cur_score = metrics_dict['Sm'] + metrics_dict['meanEm'] + metrics_dict['wFm']

        if epoch == 1:
            best_score = cur_score
            best_metric_dict = metrics_dict
            print('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={}, MAE={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))
            logging.info('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},  MAE={})'.format(
                epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')

            print('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})\n[Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})'.format(
                    epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae'],
                    best_epoch, best_metric_dict['Sm'], best_metric_dict['wFm'], best_metric_dict['meanEm'],best_metric_dict['mae']))
            logging.info('[Cur Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})\n[Best Epoch: {}] Metrics ( Sm={}, wFm={}, meanEm={},MAE={})'.format(
                    epoch, metrics_dict['Sm'], metrics_dict['wFm'], metrics_dict['meanEm'], metrics_dict['mae'],
                    best_epoch, best_metric_dict['Sm'], best_metric_dict['wFm'], best_metric_dict['meanEm'],best_metric_dict['mae']))

        writer.add_scalar('Test_Sm', metrics_dict['Sm'], global_step=epoch)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='3', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/TFCNet/',
                        help='the path to save model and log')
    opt = parser.parse_args()

    # set the device for training
#    if opt.gpu_id == '0':
#       os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#       print('USE GPU 0')
#    elif opt.gpu_id == '1':
#       os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#       print('USE GPU 1')
#    cudnn.benchmark = True

    # build the model
    model = TFCNet().cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'image//',
                              gt_root=opt.train_root + 'mask//',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
    val_loader = test_dataset(image_root=opt.val_root + 'image//',
                              gt_root=opt.val_root + 'mask//',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)


