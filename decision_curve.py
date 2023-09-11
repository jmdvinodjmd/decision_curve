'''
  Decision curve analysis
  https://en.wikipedia.org/wiki/Decision_curve_analysis
'''
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_curve(y_test, y_pred, feature_names):
        thresholds = np.linspace(0, 1, 101)
        benefits = []
        all = []
        none = []
        y_all = np.ones_like(y_test)
        y_no = np.zeros_like(y_test)

        for thresh in thresholds:
                # model
                preds = (y_pred[:,1] >= thresh).astype(int)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, preds, labels=range(2)).ravel()
                bf = (tp - fp * thresh / (1 - thresh))/len(y_test)
                benefits.append(bf)

                # all
                preds = (y_pred[:,1] >= thresh).astype(int)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_all, labels=range(2)).ravel()
                all_ = (tp - fp * thresh / (1 - thresh))/len(y_test)
                all.append(all_)

                # model
                preds = (y_pred[:,1] >= thresh).astype(int)
                tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_no, labels=range(2)).ravel()
                no_ = (tp - fp * thresh / (1 - thresh))/len(y_test)
                none.append(no_)
       
        fig = plt.figure()
        plt.plot(thresholds, benefits, label='model')
        plt.plot(thresholds, all, label='all')
        plt.plot(thresholds, none, label='none')
        # plt.xlim([0.0, 1.0])
        plt.ylim([-0.01, 0.4])
        plt.xlabel('Threshold Probability')
        plt.ylabel('Net Benefit')
        # plt.title('ROC')
        plt.legend()
        plt.show()

# plot_decision_curve(y_test, y_pred, feature_names)
