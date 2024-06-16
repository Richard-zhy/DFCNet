# -*- coding: utf-8 -*-

import os.path as osp
import argparse
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import build_dataset_5
from datasets import build_dataset_20
import torch.optim as optim
import os
import pprint
from my_engine_attn import train, evaluate
from utils.save_indicator import SaveIndicator
from utils.get_event_time import get_event_time
from utils.create_model import create_model
from utils.create_lr_scheduler import create_lr_scheduler
from utils.read_yaml import read_yaml
from utils.save_attns import save_attns


def main(cfg):
    time_now = datetime.datetime.now()
    unique_comment = f'PT_{time_now.month}-{time_now.day}-{time_now.hour}-{time_now.minute}'
    out_dir = osp.join(os.getcwd().rsplit('\\',1)[0], 'results')
    final_out = osp.join(out_dir, unique_comment)
    os.makedirs(final_out, exist_ok=True)
    # 记录指标
    my_indicator = SaveIndicator(final_out)
    logger = my_indicator.make_logger(console=True, file=True)
    cfg_data = pprint.pformat(cfg, indent=4, width=80)
    logger.info(cfg_data)
    device = torch.device("cuda")


    dataset_train_5 = build_dataset_5(cfg['5_train_data'], image_set='train', is_training=True)
    dataset_val_5 = build_dataset_5(cfg['5_val_data'], image_set='test',is_training=False)
    train_loader_5 = DataLoader(dataset_train_5,
                              batch_size=cfg['batch_size'],
                              drop_last=False,
                              shuffle=False,
                              # num_workers=cfg['num_workers'],
                              num_workers=1,
                              pin_memory=True)
    val_loader_5 = DataLoader(dataset_val_5,
                            batch_size=cfg['batch_size'],
                            drop_last=False,
                            shuffle=False,
                            # num_workers=cfg['num_workers'],
                            num_workers=1,
                            pin_memory=True)

    # 加载20倍数据
    dataset_train_20 = build_dataset_20(cfg['20_train_data'], image_set='train', is_training=True)
    dataset_val_20 = build_dataset_20(cfg['20_val_data'], image_set='test',is_training=False)
    train_loader_20 = DataLoader(dataset_train_20,
                              batch_size=cfg['batch_size'],
                              drop_last=False,
                              shuffle=False,
                              # num_workers=cfg['num_workers'],
                              num_workers=4,
                              pin_memory=True)
    val_loader_20 = DataLoader(dataset_val_20,
                            batch_size=cfg['batch_size'],
                            drop_last=False,
                            shuffle=False,
                            # num_workers=cfg['num_workers'],
                            num_workers=4,
                            pin_memory=True)

    model = create_model(model_name=cfg['model_name'])
    if cfg['loss'] == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("loss function value choice error")

    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = create_lr_scheduler(optimizer, num_step=1, epochs=cfg['epochs'],
                                    warmup=True, warmup_epochs=60)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of model params:  {n_parameters}')
    model.to(device)

    best_acc = 0
    basic_acc_savemodel= 0.7
    for epoch in range(cfg['epochs']):
        train_prob_every_wsi, train_acc, attn_scores = train(model, criterion, optimizer,
                                                train_loader_5,train_loader_20, device, epoch,
                                                logger )
        val_loss, logits, truth, prob_every_wsi, attn_scores, all_tilenames = evaluate(model, criterion, val_loader_5,val_loader_20, device)

        if cfg['auto_lr']:
            scheduler.step()
        # 计算验证集上指标并保存
        event_time = get_event_time(prob_every_wsi['wsi_name'])
        val_acc, precision, recall, spec, f1, mcc, cindex = my_indicator.my_numerical(epoch, val_loss, truth, logits,
                                                                                      # writer,
                                                                                      logger, event_time,
                                                                                      prob_every_wsi['pred_0'],
                                                                                      mode='val', cal_c=True)
        # 保存权重与结果
        if val_acc > basic_acc_savemodel or val_acc > best_acc:
            if val_acc > best_acc: # 保存最优权重
                best_acc = val_acc
                logger.info("Get Best ↑  !")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(final_out, 'val_best_{}_{}.pth'.format(epoch, best_acc)))

            elif val_acc > basic_acc_savemodel:
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(obj, os.path.join(final_out, 'val_over_{}_{}.pth'.format(epoch, val_acc)))
            my_indicator.my_plot(epoch, truth, logits, val_acc, prob_every_wsi['prob'][:,1], 'val', show=False)
            my_indicator.save_prediction_to_csv(prob_every_wsi, 'prediction_result_{}_{}.csv'.format(epoch, val_acc))
            save_attns(attn_scores, all_tilenames, save_path=final_out, epoch=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="My idea test")
    parser.add_argument("--cfg",
                        default=r"\path\to\my_cfg.yaml",metavar="FILE", help="path to config file", type=str,)
    args = parser.parse_args()
    cfg = read_yaml(args.cfg)
    main(cfg)





