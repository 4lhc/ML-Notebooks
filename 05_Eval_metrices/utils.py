#!/bin/env python3

# Author: Sreejith S
# Date: Sat 26 Sep 2020 15:21:44 IST
#
# Plotting Utilities


import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import matthews_corrcoef
import seaborn as sns


def plot_table(eval_metrics_list, ax, titles):
    ax.set_axis_off()
    table = ax.table(
      cellText = eval_metrics_list,
      rowLabels = titles,
      colLabels = ["Score", "Precison", "Recall", "F1 Score", "MCC", "AUROC", "Avg. Precision"],
      rowColours =["skyblue"] * 5,
      colColours =["skyblue"] * 7,
      cellLoc = 'center',
      loc = 'upper left'
    )
    table.set_fontsize(15)
    table.scale(1.1, 2)
    #ax.text(0.5, 0.45, "text", ha="center" )

    
def plot_confusion(cf_matrices, titles):
    """
    Plot multiple cfm side by side:
    Ref : https://stackoverflow.com/questions/61825227/plotting-multiple-confusion-matrix-side-by-side
    """
    fig, axes = plt.subplots(1,5, sharex=True, sharey=True,figsize=(10,2))

    for i, ax in enumerate(axes.flat):
        sns.heatmap(cf_matrices[i], ax=ax,cbar=i==4, 
                    cmap="YlGnBu", annot=True, fmt="d")
        ax.set_title(titles[i],fontsize=8)

    fig.text(0.45, 0.0, 'Predicted label', ha='left')
    fig.text(0.1, 0.45, 'True label', ha='left', rotation=90, rotation_mode='anchor')
    
    
def plotter(dummy, logreg, X_test, y_test, dataset_title):
    """
    V0.0
    """
    y_pred_dummy = dummy.predict(X_test)
    y_pred_logreg = logreg.predict(X_test)
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(16,16))
    ax = axs.flatten()
    ax[0].set_title("DummyClassifier")
    plot_confusion_matrix(dummy, X_test, y_test,
                                        ax=ax[0],
                                        values_format='d',
                                        cmap=plt.cm.Blues)
    ax[1].set_title("LogisticRegression")
    plot_confusion_matrix(logreg, X_test, y_test,
                                        ax=ax[1],
                                        values_format='d',
                                        cmap=plt.cm.Blues)
    ax[2].set_title("ROC")
    plot_roc_curve(dummy, X_test, y_test, ax=ax[2])
    plot_roc_curve(logreg, X_test, y_test, ax=ax[2])
    #conf_mat_disp.ax_.set_title("Dummy Classifier (strategy='most_frequent')")
    #roc_disp.ax_.set_title("ROC")


    mets = [[dummy.score(X_test, y_test), logreg.score(X_test, y_test)],
          [precision_score(y_test, y_pred_dummy),
           precision_score(y_test, y_pred_logreg)],
          [recall_score(y_test, y_pred_dummy),
           recall_score(y_test, y_pred_logreg)],
          [f1_score(y_test, y_pred_dummy),
           f1_score(y_test, y_pred_logreg)],
          [matthews_corrcoef(y_test, y_pred_dummy),
           matthews_corrcoef(y_test, y_pred_logreg)]]
    table_cells = [[f"{j:.2f}" for j in i] for i in mets]
    table_cells[1][0] = "N.D"
    table_cells[4][0] = "N.D"
    print(table_cells)

    ax[3].set_axis_off()
    table = ax[3].table(
      cellText = table_cells,
      rowLabels = ["Score", "Precison", "Recall", "F1 Score", "MCC"],
      colLabels = ["DummyClassifier", "LogisticRegression"],
      rowColours =["skyblue"] * 5,
      colColours =["skyblue"] * 2,
      cellLoc = 'center',
      loc = 'upper left'
    )
    table.set_fontsize(15)
    table.scale(1.1, 2)
    ax[3].set_title(f"Evaluation Metrices for {dataset_title}")
    plt.suptitle(dataset_title)
    plt.show()

    
    
def test_models(dataset, f_list, test_size=0.3, random_state=0):
    """
    V0.0
    """
    #dummy_first_plot = True
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 5), sharey='row')
    ax = axs.flatten()
    
    for i, f in enumerate(f_list):
        X = dataset[f]
        y = dataset[['Diab']].values.ravel()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                      random_state=random_state)
        
        
        if i == 0:
            #All dummy Classifier metrices are going to be same for splits with same random_state 
            dummy = DummyClassifier(strategy='most_frequent', random_state=0)
            dummy.fit(X_train, y_train)
            y_pred_dummy = dummy.predict(X_test)
            cfm = confusion_matrix(y_test, y_pred_dummy)
            cfm_disp = ConfusionMatrixDisplay(cfm)
            cfm_disp.plot(ax=ax[0])
            cfm_disp.ax_.set_title("DummyClassifier")
            cfm_disp.im_.colorbar.remove()
            cfm_disp.ax_.set_xlabel('')
                   
        cfm_disp.ax_.set_ylabel('')
            
            
        logreg = LogisticRegression(max_iter=200)
        logreg.fit(X_train, y_train)
        y_pred_logreg = logreg.predict(X_test)
        cfm = confusion_matrix(y_test, y_pred_logreg)
        cfm_disp = ConfusionMatrixDisplay(cfm)
        cfm_disp.plot(ax=ax[i+1])
        #cfm_disp.ax_.set_title("DummyClassifier")
        cfm_disp.im_.colorbar.remove()
        cfm_disp.ax_.set_xlabel('')
        
        
    fig.text(0.4, 0.1, 'Predicted label', ha='left')
    fig.text(0.1, 0.45, 'True label', ha='left', rotation=90, rotation_mode='anchor')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(cfm_disp.im_, ax=axs)
    plt.ylabel('')
    plt.show()

