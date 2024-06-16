# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import torch
import sys
import torch.nn.functional as F
def train(model, criterion, optimizer, train_loader_5,train_loader_20, device, epoch, logger,
          # writer,
          mode='train' ):
    model.train(True)
    epoch_loss = 0
    correct_nums = 0
    count_iter = 0
    logits = []
    truth = []
    epoch_pred_prob = []
    all_wsiname = []
    attns = []
    train_loader = zip(train_loader_5, train_loader_20)
    train_loader = tqdm(train_loader, file=sys.stdout)
    for step, (data1, data2) in enumerate(train_loader):
        # data
        count_iter += 1
        images_5, targets = data1
        images_20,_ = data2
        images_5 = images_5.to(device)
        images_20 = images_20.to(device)
        labels = targets['wsi_label'][0].clone().detach()
        labels = labels.unsqueeze(0).to(device)  #
        truth.append(int(labels.cpu())) #
        all_wsiname.extend(targets['wsi_name']) #
        # predict
        pred_prob, attn_score = model(images_5, images_20)
        loss = criterion(pred_prob, labels)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        epoch_pred_prob.append(pred_prob.detach().cpu().numpy()) #
        pred_classes = torch.max(pred_prob, dim=1)[1]
        logits.extend(pred_classes.cpu()) #
        # cal acc and backward
        correct_nums += torch.eq(pred_classes, labels).sum()
        loss.backward()
        epoch_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()
    epoch_acc = correct_nums / len(truth)
    # record result
    mean_loss = epoch_loss / count_iter
    logger.info("[train epoch:{}]  train_loss:{:.5f}, train_acc:{:.5f}".format(epoch, mean_loss, epoch_acc))
    prob_every_wsi = {}
    prob_every_wsi['label'] = truth
    prob_every_wsi['pred'] = logits
    prob_every_wsi['prob'] = np.array(epoch_pred_prob).squeeze() # 需要变成array
    prob_every_wsi['wsi_name'] = all_wsiname
    return prob_every_wsi, epoch_acc, attns

@torch.no_grad()
def evaluate(model, criterion, val_loader_5, val_loader_20, device):
    model.eval()
    criterion.eval()
    epoch_loss = 0
    logits = []
    truth = []
    img_id = []
    epoch_pred_prob = []
    count_iter = 0
    survival_prob = []
    all_wsiname = []
    # 存储一轮的计算结果
    attn_scores = {}
    all_tilenames = {}

    val_loader = zip(val_loader_5, val_loader_20)
    for step, (data1,data2) in enumerate(val_loader):
        count_iter += 1
        images_5, targets = data1
        images_20, _ = data2
        images_5 = images_5.to(device)
        images_20 = images_20.to(device)
        labels = targets['wsi_label'][0].clone().detach()
        labels = labels.unsqueeze(0).to(device) #
        truth.append(int(labels.cpu())) #
        all_wsiname.extend(targets['wsi_name']) #
        # predict
        pred_prob, attn_score = model(images_5, images_20)
        loss = criterion(pred_prob, labels)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        epoch_loss += loss.detach()
        epoch_pred_prob.append(pred_prob.detach().cpu().numpy())
        pred_classes = torch.max(pred_prob, dim=1)[1]
        logits.extend(pred_classes.cpu())
        survival_prob.extend(pred_prob[:, 0].cpu()) # 生存时间
        # 对attn_scores做一些操作
        attn_score = attn_score.sum(dim=0)
        attn_score = attn_score.softmax(dim=0)
        attn_score.cpu().detach().numpy()*100
        flattened_list = []
        for i in targets['one_tilenames']:
            if isinstance(i, list):
                flattened_list.extend(i)
            else:
                flattened_list.append(i)
        all_tilenames[targets['wsi_name'][0]] = flattened_list
        attn_scores[targets['wsi_name'][0]] = attn_score
    # cal return values:
    mean_loss = epoch_loss / count_iter
    prob_every_wsi = {}
    prob_every_wsi['label'] = truth
    prob_every_wsi['pred'] = logits
    prob_every_wsi['prob'] = np.array(epoch_pred_prob).squeeze()
    prob_every_wsi['wsi_name'] = all_wsiname
    prob_every_wsi['pred_0'] = survival_prob
    # 保存的内容：平均损失，预测值，标签值，真实概率（用于存储csv）
    return mean_loss.cpu(), logits, truth, prob_every_wsi, attn_scores, all_tilenames