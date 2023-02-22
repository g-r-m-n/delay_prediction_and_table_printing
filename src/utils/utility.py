# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score, make_scorer
from matplotlib import pyplot as plt
from lightgbm import LGBMClassifier, log_evaluation, early_stopping 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn import preprocessing
from pandas.api.types import is_numeric_dtype
import sklearn.metrics as metrics
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import warnings


        
def train_pred_model(df,  y_var = 'depart_delay_is_large', x_vars=[], horizon=14, model_type = 'rf', eval_var = 'carrier', TUNE = True, PLOT = False, output_folder_plots = '', title0='ROC_curve', title1='Prediction', title2= 'importance', SAVE_OUTPUT = 1):
    """function to tune, train, predict and plot model results"""
    # Copy the input data to not overwrite it:
    df1 = df.copy()
    # Cet date and 'eval_var' as index:
    #df1 = df1.set_index(['date',eval_var], drop=False, inplace=False, append=False)

    # Create the feature data set:
    X = df1[x_vars].copy() 
    
    # For non-numeric data, set them to the categorial data type: 
    for i in x_vars:
        if not is_numeric_dtype(X.loc[:,i]):
            X.loc[:,i] = X.loc[:,i].astype("category")
        

    # Create the response variable
    y = df1[y_var]
    
    # Get the dates of the dataset for validation
    inx_day = np.unique(df['date'])
    
    # Take last days of the dataset for validation
    if horizon >= 1:
        inx_day =inx_day[-horizon]
        # Get the training and test data:
        X_train, X_test = X.iloc[X.index[df['date']<inx_day],:], X.iloc[X.index[df['date']>=inx_day],:]
        y_train, y_test = y.iloc[y.index[df['date']<inx_day]], y.iloc[y.index[df['date']>=inx_day]]        
    elif (0 < horizon ) and (horizon < 1):   
        # Get the number of out-of-sample dates:
        n_oos = int(len(inx_day)*horizon)
        # Randomly sample the  out-of-sample dates:
        inx_day = np.random.choice(inx_day,n_oos , replace=False)
        # Get the training and test data:
        X_train, X_test = X.iloc[X.index[~df['date'].isin(inx_day)],:], X.iloc[X.index[df['date'].isin(inx_day)],:]
        y_train, y_test = y.iloc[y.index[~df['date'].isin(inx_day)]],   y.iloc[y.index[df['date'].isin(inx_day)]]
    
    # Use an Random Forest ML model:
    if 0 and model_type == 'rf':
        clf = RandomForestClassifier(n_estimators = 20, random_state=42 )
        tuning_dict = {
            'max_depth': [5,7,9, None],
            'min_samples_split': [2,3,5],
            'max_features': [None,'sqrt', 'log2','auto']
        },      
    # Use an LGBM ML model:    
    if model_type in ['lgbm','rf']:
        n_estimators = 20
                 
        params = {'subsample': 0.5, 'num_leaves': 10, 'max_depth': 5, 'learning_rate': 0.2} #'boosting_type' : 'dart'}
        if model_type == 'rf':
                params = params | {'boosting_type' : 'rf', 'bagging_freq' : 1, 'bagging_fraction' : 0.8  } 
                # {'subsample': 0.5, 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.01}
        clf = SilentLGBMClassifier( random_state=42, n_estimators=n_estimators, **params)
        tuning_dict = { 
               #'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                                'max_depth': [3, 5, -1],
                                'num_leaves': [5, 10, 31],
                                'subsample': [0.3, 0.5, 1]
            }
        # 'subsample': 0.8, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.01 
        #{'subsample': 1, 'num_leaves': 31, 'max_depth': -1, 'learning_rate': 0.01}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 15, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.3}
        #{'subsample': 0.5, 'num_leaves': 10, 'max_depth': 5, 'learning_rate': 0.1}
    #create, train and do inference of the model
    if TUNE:
        # Tune hyperparameters and final model using cv cross-validation with n_iter parameter settings sampled from random search. Random search can cover a larger area of the paramter space with the same number of consider setting compared to e.g. grid search.
        rs = RandomizedSearchCV(clf, tuning_dict, 
            scoring= {'F1':  make_scorer(f1_score), 'balanced_accuracy': make_scorer(balanced_accuracy_score)}, #'f1', 'balanced_accuracy' Overview of scoring parameters: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                # default: accuracy_score for classification and sklearn.metrics.r2_score for regression
            refit= 'F1',
            cv=5, 
            return_train_score=False, 
            n_iter=20,
            verbose = True
        )
        print("\nTuning hyperparameters ..")
        rs.fit(X_train, y_train, eval_set = [[X_test, y_test]], 
               eval_metric = ['accuracy','auc','f1','logloss'], 
               callbacks=[log_evaluation(n_estimators), early_stopping(10)])    # 'f1_score','accuracy' # 
        
        print("\nTuned hyperparameters :(best score)     ",rs.best_score_)
        print("\nTuned hyperparameters :(best prameters) ",rs.best_params_)
        
        model = clf
        clf.set_params(**rs.best_params_)
    else:
        model = clf
    model.fit(X_train, y_train)
    predictions_prob = model.predict_proba(X_test)[:,1]
    predictions      = model.predict(X_test)
    
   
    # Calculate precision recall fscore
    perf_results = binary_classification_performance(y_test, predictions, predictions_prob)
    print("\nOut-of-sample prediction performance:\n")
    print(perf_results)
    
    # Plot auc roc curve
    if 1:
        plot_roc_curve(y_test, predictions_prob)
        # Saving plot to pdf and png file
        if SAVE_OUTPUT:
            plt.savefig(output_folder_plots  +title0+'.pdf', dpi=100,bbox_inches="tight")
            #plt.title(title1, fontsize=20)
            plt.savefig(output_folder_plots  +title0+ '.png', dpi=100,bbox_inches="tight")    
        plt.show()
        
    # Plot Observed vs prediction for the horizon of the dataset
    if PLOT:
        items = np.unique(df1.index.get_level_values(eval_var))
        for i in items:
            fig = plt.figure(figsize=(16,8))
            inx_i = y_test.index.get_level_values(eval_var)==i
            #calculate precision recall fscore
            precision, recall, fscore, _ = np.round(precision_recall_fscore_support(y_test[inx_i], predictions[inx_i]), 3) 
            plt.title(f'Observed vs Prediction - Precision: {precision}, Recall: {recall}, fscore: {fscore}', fontsize=20)
            plt.plot(pd.Series(y_test[inx_i].values,index=y_test.index.get_level_values('date')[inx_i]), color='red')
            plt.plot(pd.Series(predictions[inx_i],  index=y_test.index.get_level_values('date')[inx_i]), color='green')
            plt.xlabel('date', fontsize=16)
            plt.ylabel(y_var+':'+str(i), fontsize=16)
            plt.legend(labels=['Real', 'Prediction'], fontsize=16)
            plt.grid()
            # Saving plot to pdf and png file
            if SAVE_OUTPUT:
                plt.savefig(output_folder_plots  +title1+'_'+str(i)+'.pdf', dpi=100,bbox_inches="tight")
                #plt.title(title1, fontsize=20)
                plt.savefig(output_folder_plots  +title1+'_'+str(i)+ '.png', dpi=100,bbox_inches="tight")
            plt.show()
            
    # Ensure that the model has the attribute feature_name_:
    if not hasattr(model, 'feature_name_'):
         model.feature_name_   = model.feature_names_in_
            
    # Create a dataframe with the variable importances of the model
    df_importances = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    # Plot variable importances of the model
    MAX_VARS = 20
    plt.title('Variable Importances', fontsize=16)
    sns.barplot(x=df_importances.importance[:MAX_VARS], y=df_importances.feature[:MAX_VARS], orient='h')
    
    # Saving plot to pdf and png file
    if SAVE_OUTPUT:
        plt.savefig(output_folder_plots  +title2+'.pdf', dpi=100,bbox_inches="tight")
        #plt.title(title1, fontsize=20)
        plt.savefig(output_folder_plots  +title2+ '.png', dpi=100,bbox_inches="tight")    
    plt.show()
    
    return model
              
    
    
def binary_classification_performance(y_test, y_pred, y_pred_prob, labels=[1,0]):
    tp, fp, fn, tn = confusion_matrix(y_test, y_pred, labels=labels).ravel()
    accuracy       = round((tp+tn)/(tp+tn+fp+fn),2)
    precision      = round(tp/(tp+fp),2)
    recall         = round(tp/(tp+fn),2)
    f1_score       = round((2*tp)/(2*tp+fp+fn),2)
    specificity    = round(tn/(tn+fp),2)
    npv            = round(tn/(tn+fn),2)
    auc_roc        = round(roc_auc_score(y_score = y_pred_prob, y_true = y_test, labels=labels),2)

    result = pd.DataFrame({'AUC_ROC' : [auc_roc],
                         'Accuracy  [=(tp+tn)/(tp+tn+fp+fn)]' : [accuracy],
                         'Precision [=tp/(tp+fp)]' : [precision],
                         'Recall    [=tp/(tp+fn)]' : [recall],
                         'f1 score  [=(2*tp)/(2*tp+fp+fn)]' : [f1_score],
                         #'Specificty (or TNR)': [specificity],
                         #'NPV' : [npv],
                         'True Positive  (tp)' : [tp],
                         'True Negative  (tn)' : [tn],
                         'False Positive (fp)':[fp],
                         'False Negative (fn)':[fn]},index=['Prediction Performance'])
    return result.T



def plot_roc_curve(y_test, preds):
    """Plot the Receiver Operating Characteristic (ROC) curve for a binary prediction scenario."""
    # Calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    

    
class SilentLGBMClassifier(LGBMClassifier):
    """In order to ingore userwarnings due to bug in lgbm. See https://github.com/microsoft/LightGBM/issues/3379"""
    def fit(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return super().fit(*args, **kwargs)
        
            

def plot_pie_delay_per_carrier(df):
    df2 = df.loc[:, ['carrier', 'dep_delay']]
    font = {'family' : 'normal', 'weight' : 'bold', 'size'   : 15}

    colors = mcolors.CSS4_COLORS
    colors= np.random.choice(list(colors.keys()),len(colors) , replace=False)
    abbr_companies = {i:i for i in  np.unique(df.carrier)}

    fig = plt.figure(1, figsize=(16,15))
    gs=plt.GridSpec(2,2)             
    ax1=fig.add_subplot(gs[0,0]) 
    ax2=fig.add_subplot(gs[0,1]) 
    ax3=fig.add_subplot(gs[1,:]) 

    # Pie chart nr 1: nb of flights
    # Function that extract statistical parameters from a grouby objet:
    def get_stats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}

    # Creation of a dataframe with statitical infos on each airline:
    global_stats = df['dep_delay'].groupby(df['carrier']).apply(get_stats).unstack()
    global_stats = global_stats.sort_values('count')
    global_stats

    labels = [s for s in  global_stats.index]
    sizes  = global_stats['count'].values
    explode = [0.3 if sizes[i] < 20000 else 0.0 for i in range(len(abbr_companies))]
    patches, texts, autotexts = ax1.pie(sizes, explode = explode,
                                    labels=labels, colors = colors,  autopct='%1.0f%%',
                                    shadow=False, startangle=0)
    for i in range(len(abbr_companies)): 
        texts[i].set_fontsize(14)
    ax1.axis('equal')
    ax1.set_title('% of flights per company', bbox={'facecolor':'midnightblue', 'pad':5},
                  color = 'w',fontsize=18)

    # Set the legend: abreviation -> airline name
    if 0:
        comp_handler = []
        for i in range(len(abbr_companies)):
            comp_handler.append(mpatches.Patch(color=colors[i],
                    label = global_stats.index[i] + ': ' + abbr_companies[global_stats.index[i]]))
        ax1.legend(handles=comp_handler, bbox_to_anchor=(0.2, 0.9), 
                   fontsize = 13, bbox_transform=plt.gcf().transFigure)

    # Pie chart nr 2: mean delay at departure
    sizes  = global_stats['mean'].values
    sizes  = [max(s,0) for s in sizes]
    explode = [0.0 if sizes[i] < 20000 else 0.01 for i in range(len(abbr_companies))]
    patches, texts, autotexts = ax2.pie(sizes, explode = explode, labels = labels,
                                    colors = colors, shadow=False, startangle=0,
                                    autopct = lambda p :  '{:.0f}'.format(p * sum(sizes) / 100))
    for i in range(len(abbr_companies)): 
        texts[i].set_fontsize(14)
    ax2.axis('equal')
    ax2.set_title('Mean delay at origin', bbox={'facecolor':'midnightblue', 'pad':5},
                  color='w', fontsize=18)

    # Striplot with all the values reported for the delays
    # Redefine the colors for correspondance with the pie charts
    colors = ['firebrick', 'gold', 'lightcoral', 'aquamarine', 'c', 'yellowgreen', 'grey',
              'seagreen', 'tomato', 'violet', 'wheat', 'chartreuse', 'lightskyblue', 'royalblue']

    ax3 = sns.stripplot(y="carrier", x="dep_delay", size = 4, palette = colors,
                        data=df2, linewidth = 0.5,  jitter=True)
    plt.setp(ax3.get_xticklabels(), fontsize=14)
    plt.setp(ax3.get_yticklabels(), fontsize=14)
    ax3.set_xticklabels(['{:2.0f}h{:2.0f}m'.format(*[int(y) for y in divmod(x,60)])
                             for x in ax3.get_xticks()])
    plt.xlabel('Departure delay', fontsize=18, bbox={'facecolor':'midnightblue', 'pad':5},
               color='w', labelpad=20)
    ax3.yaxis.label.set_visible(False)

    plt.tight_layout(w_pad=3)    
    plt.show()
    
    
def plot_pie_delay_per_carrier_s(df):    
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            plot_pie_delay_per_carrier(df)
            
def plot_num_flight_grouped_delay(df):
    """Show per carrier the number of flights within delay classes: normal, small and large delay"""
    # Function that define how delays are grouped
    delay_type = lambda x:((0,1)[x > 5],2)[x > 30]
    df['DELAY_LEVEL'] = df['dep_delay'].apply(delay_type)
    abbr_companies = {i:i for i in  np.unique(df.carrier)}

    fig = plt.figure(1, figsize=(10,7))
    ax = sns.countplot(y="carrier", hue='DELAY_LEVEL', data=df)

    # Replace the abbreviations by the full names of the companies and set the labels
    labels = [abbr_companies[item.get_text()] for item in ax.get_yticklabels()]
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);
    plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'normal', rotation = 0);
    ax.yaxis.label.set_visible(False)
    plt.xlabel('Flight count', fontsize=12, weight = 'normal', labelpad=10)

    # Set the legend
    L = plt.legend()
    L.get_texts()[0].set_text('on time (t < 5 min)')
    L.get_texts()[1].set_text('small delay (5 < t < 30 min)')
    L.get_texts()[2].set_text('large delay (t > 30 min)')
    plt.show()  
    
    
    