# -*- coding: utf-8 -*-

import os, sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve
from utils import load_fea_target, draw_multiclass_roc


def set_args():
    parser = argparse.ArgumentParser(description='Thyroid ROI Classification')
    parser.add_argument('--fea_dir',         type=str,   default="../data/ThyroidS5/FeasROI/L2Feas")
    parser.add_argument('--model_name',      type=str,   default="resnet50")
    parser.add_argument('--seed',            type=int,   default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.seed)

    fea_dir = os.path.join(args.fea_dir, args.model_name)
    train_fea, train_target, train_list  = load_fea_target(os.path.join(fea_dir, 'train'))
    val_fea, val_target, val_list = load_fea_target(os.path.join(fea_dir, 'val'))
    print("Length of training rois: {}".format(len(train_target)))
    print("Length of testing rois: {}".format(len(val_target)))

    dtree_model = DecisionTreeClassifier(max_depth=2).fit(train_fea, train_target)
    dtree_preds = dtree_model.predict(val_fea)
    dtree_cm = confusion_matrix(val_target, dtree_preds)
    dtree_acc = accuracy_score(val_target, dtree_preds)
    print("----Decision Tree")
    print("Confusion matrix:")
    print(dtree_cm)
    print("Accuracy is: {:.3f}".format(dtree_acc))

    svc_model = SVC(kernel='rbf').fit(train_fea, train_target)
    svc_preds = svc_model.predict(val_fea)
    svc_cm = confusion_matrix(val_target, svc_preds)
    svc_acc = accuracy_score(val_target, svc_preds)
    print("----SVM with Gaussian Kernel")
    print("Confusion matrix:")
    print(svc_cm)
    print("Accuracy is: {:.3f}".format(svc_acc))

    rf_model = RandomForestClassifier(random_state=0).fit(train_fea, train_target)
    rf_preds = rf_model.predict(val_fea)
    rf_cm = confusion_matrix(val_target, rf_preds)
    rf_acc = accuracy_score(val_target, rf_preds)
    print("----Random Forest")
    print("Confusion matrix:")
    print(rf_cm)
    print("Accuracy is: {:.3f}".format(rf_acc))

    mlp_model = MLPClassifier().fit(train_fea, train_target)
    mlp_preds = mlp_model.predict(val_fea)
    mlp_cm = confusion_matrix(val_target, mlp_preds)
    mlp_acc = accuracy_score(val_target, mlp_preds)
    print("----MLP Classifier")
    print("Confusion matrix:")
    print(mlp_cm)
    print("Accuracy is: {:.3f}".format(mlp_acc))

    rf_pred_proba = rf_model.predict_proba(val_fea)
    draw_multiclass_roc(val_target, rf_pred_proba)
