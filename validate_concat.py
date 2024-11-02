#!/usr/bin/env python3
import argparse
import os
import json
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from contextlib import suppress
from torch.utils.data.dataloader import DataLoader
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

import models
from utils.metrics_test import *
from datasets.mp_liver_dataset import MultiPhaseLiverDataset, create_loader
import yaml
import re



has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='RU-Net Test')
parser.add_argument('-c', '--config', default='exp_configs.yaml', type=str, metavar='FILE',
                           help='YAML config file specifying default arguments (default: exp_configs.yaml)')

parser.add_argument('--img_size', default=(64, 128, 128), type=int, nargs='+', help='input image size.')
parser.add_argument('--crop_size', default=(56, 112, 112), type=int, nargs='+', help='cropped image size.')
parser.add_argument('--data_dir', default='', type=str)
parser.add_argument('--val_anno_file', default='./labels/test.txt', type=str)
parser.add_argument('--val_transform_list', default=['center_crop'], nargs='+', type=str)
parser.add_argument('--model', '-m', metavar='NAME', default='uniformer',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--num-classes', type=int, default=-1,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--test_results_dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--score-dir', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation score (summary)')

parser.add_argument('--feat_csv_dir', default='', type=str)
parser.add_argument('--exp', default='BCLM_CRLM', type=str, metavar='NAME',
                    help='Name of the training experiment (default: "BCLM_CRLM")')
parser.add_argument('--date', default='0918', type=str,
                    help='Date of the experiment (default: "0918")')
parser.add_argument('--fold-index', default=1, type=int,
                    help='Fold index (default: 1)')
parser.add_argument('--fold-name', default='5-fold_0918', type=str,
                    help='Name of the fold (default: "5-fold_0918")')
parser.add_argument('--feature-fusion', action='store_true', default=False,
                    help='Enable feature fusion (default: False)')

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    if 'uniformer' in args.model:
    #### use uniformer small or base
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            pretrained_cfg=None)
    
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))


    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = MultiPhaseLiverDataset(args, is_training=False)

    loader = DataLoader(dataset, 
                        batch_size=args.batch_size, 
                        num_workers=args.workers, 
                        pin_memory=args.pin_mem,
                        shuffle=False)

    batch_time = AverageMeter()

    predictions = []
    labels = []

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        end = time.time()
        for (input, target, feat) in tqdm(loader):
            target = target.cuda()
            input = input.cuda()
            feat = feat.cuda()
            # compute output
            with amp_autocast():
                output = model(input, feat)
            predictions.append(output)
            labels.append(target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    evaluation_metrics = compute_metrics(predictions, labels, criterion, args)
    return evaluation_metrics

def compute_metrics(outputs, targets, loss_fn, args):
    
    outputs = torch.cat(outputs, dim=0).detach()
    targets = torch.cat(targets, dim=0).detach()
    pred_score = torch.softmax(outputs, dim=1)
    loss = loss_fn(outputs, targets).cpu().item()
    outputs = outputs.cpu().numpy()
    targets = targets.cpu().numpy()
    pred_score = pred_score.cpu().numpy()
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    report = cls_report(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    auc = compute_auc(outputs, targets)
    specificity = compute_specificity(outputs, targets)
    metrics = OrderedDict([
        ('Specificity',specificity),
        ('Sensitivity', recall),
        ('F1-Score', f1),
        ('AUC', auc),
        ('precision', precision),
        ('Accuracy', acc),
        ('Kappa', kappa),
        #('confusion matrix', cm),
        #('classification report', report),
    ])
    return metrics, pred_score


def write_results2txt(results_dir, results):
    results_file = os.path.join(results_dir, 'results.txt')
    file = open(results_file, 'w')
    file.write(results)
    file.close()



def write_result2csv(results_dir, results):
    # 解析 output_str
    
    results_file = os.path.join(results_dir, 'results.csv')
    with open(results_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['Class', 'Specificity', 'Sensitivity', 'F1-Score', 'AUC', 'Accuracy', 'Kappa'])

        # 遍历每个类别的结果，写入到CSV文件中
        num_classes = len(results['Specificity'])
        for class_index in range(num_classes):
            writer.writerow([class_index, results['Specificity'][class_index], results['Sensitivity'][class_index], 
                             results['F1-Score'][class_index], results['AUC'][class_index],'',''])
        writer.writerow(['Overall','','','','',results['Accuracy'], results['Kappa']])



def write_score2json(score_info, args):
    # score_info = score_info.astype(np.float)
    score_info = score_info.astype(float)
    score_list = []
    anno_info = np.loadtxt(args.val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        label = int(item[1])
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'label': label,
            'prediction': pred,
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    file = open(os.path.join(args.test_results_dir, 'score.json'), 'w')
    file.write(json_data)
    file.close()

def main():
    setup_default_logging()
    
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            for key, value in cfg.items():
                if getattr(args, key) == parser.get_default(key):
                    setattr(args, key, value)
    
    args.feat_csv_file = f"./datasets/Data/Radiomics_feat/{args.exp}/{args.fold_name}/fold_{args.fold_index}_scaled_features_dl.csv"
    args.val_anno_file = f"./datasets/Data/Label/{args.exp}/{args.fold_name}/fold_{args.fold_index}_test_tumor_label_dl.txt"
    args.checkpoint = f"./output/{args.date}/{args.exp}/train/fold_{args.fold_index}/model_best.pth.tar"
    args.test_results_dir = f"./output/{args.date}/{args.exp}/test/fold_{args.fold_index}"
    
    if args.exp == 'BCLM_CRLM':
        args.num_classes = 2
    elif args.exp == 'BCLM_PLC_HHM':
        args.num_classes = 3

    results, score = validate(args)
    output_str = 'Test Results:\n'
    for key, value in results.items():
        if key == 'confusion matrix':
            output_str += f'{key}:\n {value}\n'
        elif key == 'classification report':
            output_str += f'{key}:\n {value}\n'
        else:
            output_str += f'{key}: {value}\n'
    print(results)
    os.makedirs(args.test_results_dir, exist_ok=True)
    write_results2txt(args.test_results_dir, output_str)
    write_result2csv(args.test_results_dir, results)
    write_score2json(score, args)
    print(output_str)

if __name__ == '__main__':
    main()