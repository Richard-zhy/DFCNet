'''
用于计算、记录、绘制模型相关指标
'''
import logging
import numpy as np
import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,matthews_corrcoef
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from torch.utils.tensorboard import SummaryWriter
from lifelines.utils import concordance_index


class SaveIndicator:
    def __init__(self, save_path):
        self.save_path = save_path
    def make_logger(self, console:bool=True, file:bool=True):
        logger = logging.getLogger('console')
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

        # 输出到文件
        if file==True:
            file_handler = logging.FileHandler(os.path.join(self.save_path, 'log.log'))
            file_handler.setLevel(level=logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        # 输出到控制台
        if console==True:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.DEBUG)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger
    def save_prediction_to_csv(self, input, filename):
        output_dict = dict()
        # output_dict['img_id'] = input['imd_id']
        output_dict['wsi_name'] = input['wsi_name']
        output_dict['label'] = input['label']
        output_dict['pred'] = input['pred']
        N, M = np.shape(input['prob'])
        for i in range(M):
            output_dict['pred_prob_class_{}'.format(i)] = input['prob'][:, i]
        df = pd.DataFrame(output_dict)
        df.to_csv(osp.join(self.save_path, filename))

    def my_plot(self, epoch, real, pred, acc,positive_prob,  mode='val', show:bool = False):
        # 1. 绘制混淆矩阵
        self._plot_cfm(real, pred, epoch, acc,mode, show)
        # 2. 绘制单ROC曲线
        self._plot_roc(real,positive_prob,epoch, mode, show)

    # def my_numerical(self, epoch, loss, real, pred, writer, logger, event_time, pred_0, mode='val', cal_c:bool=False):
    def my_numerical(self, epoch, loss, real, pred, logger, event_time, pred_0, mode='val', cal_c:bool=False):
        # 相关指标计算
        acc, precision, recall, f1, mcc = self._my_metrics(real, pred)
        spec = self._my_specificity(real, pred)
        if cal_c == True:
            cindex = self._my_cindex(event_time,pred_0,real)  # 指标内容
        if cal_c == False:
            logger.info("[val epoch:{}]  val_loss:{:.5f}, val_acc:{:.5f}, recall:{:.5f}, spec:{:.5f}, f1:{:.5f},  "
                  "precision:{:.5f}, mcc:{:.5f}".format(epoch, loss, acc, recall, spec, f1, precision, mcc))
        if cal_c == True:
            logger.info("[val epoch:{}]  val_loss:{:.5f}, val_acc:{:.5f}, recall:{:.5f}, spec:{:.5f}, f1:{:.5f}, Cindex:{:.5f}, "
                  "precision:{:.5f}, mcc:{:.5f}".format(epoch, loss, acc, recall, spec, f1, cindex, precision, mcc))


        return acc, precision, recall, spec, f1, mcc, cindex

    # c-index指数
    def _my_cindex(self, event_times, predicted_scores, event_observed):
        # 想要计算cindex 需要获得验证集的样本名称，从而获得样本的真实时间，以及样本是否发生事件。还需要获得预测为0的值
        return concordance_index(event_times, predicted_scores, event_observed=event_observed)

    # numerical indicators
    def _my_metrics(self, real, pred):

        real = np.array(real)
        pred = np.array(pred)
        acc = accuracy_score(real,pred)
        precision = precision_score(real, pred)
        recall = recall_score(real, pred)
        f1 = f1_score(real,pred)
        mcc = matthews_corrcoef(real,pred)
        return acc, precision, recall, f1, mcc

    # 特异度
    def _my_specificity(self, real, pred):
        real = np.array(real)
        pred = np.array(pred)
        tn = 0
        fp = 0
        for i in range(pred.shape[0]):
            if real[i] == 0 and pred[i] == 0:
                tn += 1
            elif real[i] == 0 and pred[i] == 1:
                fp += 1
        return tn / (tn + fp)

    def _plot_cfm(self, real, pred, epoch, acc,mode,show):
        cf_matrix = confusion_matrix(real, pred, normalize=None) # 非归一化
        disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix, display_labels=['normal', 'recurrence'])
        disp.plot(cmap='Blues')
        plt.tight_layout()
        experiment_name = osp.basename(self.save_path)
        plt.savefig(os.path.join(self.save_path, 'cfm_{}_{}_{:.3f}_{}.png'.format(epoch, mode, acc,experiment_name)))
        if show == True:
            plt.show()
        plt.close()

    def _plot_roc(self,real, positive_prob, epoch, mode, show):
        fpr, tpr, _ = roc_curve(real, positive_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2  # 线条宽度
        plt.plot(fpr, tpr, color='darkorange', lw=lw,
                 label='ROC curve of class tumor (area = {:0.3f})'
                       ''.format(roc_auc))
        # 0.5曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.tight_layout()
        experiment_name = osp.basename(self.save_path)
        plt.savefig(os.path.join(self.save_path, 'roc_{}_{}_{:.3f}_{}.png'.format(epoch, mode, roc_auc, experiment_name)))
        if show == True:
            plt.show()
        plt.close()