import os
import sys
sys.path.append('./code/')
import config as config
import logging

from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix,roc_curve,accuracy_score

def p_score_1(y_true,y_pred):
    return precision_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],average='macro')

def r_score_1(y_true,y_pred):
    return recall_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],average='macro')

def f1_score_1(y_true,y_pred):

    return f1_score(y_true,y_pred,labels=[1])

def f1_score_sr(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18],average='macro')

def f1_score_3(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3],average='macro')

def f1_score_377(y_true,y_pred):
    tr_token_lis = []
    with open('tr_token.txt','r') as f:
        for i in f.readlines():
            tr_token_lis.append(eval(i.strip()))
    tr_token_lis += [2498]
    return f1_score(y_true,y_pred,labels=tr_token_lis,average='macro')

def f1_score_6(y_true,y_pred):
    return f1_score(y_true,y_pred,labels=[1,2,3],average='macro')

def f1_score_2(y_true, y_pred, mode='dev'):
    """Compute the F1 score.
    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::
        F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
    Returns:
        score : float.
    Example:
        y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(y_true)
    pred_entities = set(y_pred)
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    if mode == 'dev':
        return score
    else:
        f_score = {}
        for label in config.labels:
            true_entities_label = set()
            pred_entities_label = set()
            for t in true_entities:
                if t[0] == label:
                    true_entities_label.add(t)
            for p in pred_entities:
                if p[0] == label:
                    pred_entities_label.add(p)
            nb_correct_label = len(true_entities_label & pred_entities_label)
            nb_pred_label = len(pred_entities_label)
            nb_true_label = len(true_entities_label)

            p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
            r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
            score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
            f_score[label] = score_label
        return f_score, score






