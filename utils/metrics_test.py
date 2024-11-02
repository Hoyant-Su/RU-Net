import numpy as np
from sklearn import metrics

def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    #return metrics.f1_score(y_true, y_pred, average='macro')
    return metrics.f1_score(y_true, y_pred, average=None)
def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    #return metrics.recall_score(y_true, y_pred, average='macro')
    return metrics.recall_score(y_true, y_pred, average=None)

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    #return metrics.precision_score(y_true, y_pred, average='macro')
    return metrics.precision_score(y_true, y_pred, average=None)

def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4)


def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)


def compute_auc(output, target):
    # 将 target 转换为 one-hot 编码
    y_true = np.eye(output.shape[1])[target.flatten()]
    
    # 计算每个类别的 AUC
    cls_aucs = []
    for i in range(output.shape[1]):
        scores_per_class = output[:, i]
        labels_per_class = y_true[:, i]
        auc_per_class = metrics.roc_auc_score(labels_per_class, scores_per_class)
        cls_aucs.append(auc_per_class)

    return cls_aucs


def compute_specificity(output, target):
    # 将预测输出转换为类别索引
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    
    # 计算混淆矩阵
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # 初始化存储特异性的列表
    specificities = []
    
    # 遍历每个类别计算特异性
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # 移除第 i 行和第 i 列计算 TN
        fp = np.sum(cm[:, i]) - cm[i, i]  # 第 i 列除对角线元素以外的元素之和为 FP
        specificity = tn / (tn + fp)
        specificities.append(specificity)
    
    return specificities