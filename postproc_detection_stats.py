import os
import pandas as pd
from shutil import copy2
import sys
import matplotlib.pyplot as plt
import numpy as np
import bstrap
from sklearn.metrics import average_precision_score, matthews_corrcoef
from ipdb import set_trace as bp


# define paths
PATH_EXPERIMENTS = '/scratch/groups/rubin/fdubost/project_ali/experiments'
FIRST_EXP = '48'
SECOND_EXP = '49'

def compute_F1(val_target,val_predict):
    val_target = val_target.astype('bool')
    val_predict = val_predict.astype('bool')
    tp = np.count_nonzero(val_target * val_predict)
    fp = np.count_nonzero(~val_target * val_predict)
    fn = np.count_nonzero(val_target * ~val_predict)
    return tp * 1. / (tp + 0.5 * (fp + fn) + sys.float_info.epsilon)

def compute_mF1(data):
    gt = data[[column for column in data.columns if 'gt' in column]]
    predictions = data[[column for column in data.columns if 'pred' in column]]
    nbr_classes = len(gt.columns)
    mF1 = 0
    for class_idx in range(nbr_classes):
        mF1 += compute_F1(gt['gt_'+str(class_idx)], predictions['pred_'+str(class_idx)]>0.5)
    return mF1/nbr_classes

def compute_mAP(data):
    gt = data[[column for column in data.columns if 'gt' in column]]
    predictions = data[[column for column in data.columns if 'pred' in column]]
    return average_precision_score(gt, predictions, average='weighted')
    
def compute_mMCC(data):
    gt = data[[column for column in data.columns if 'gt' in column]]
    predictions = data[[column for column in data.columns if 'pred' in column]]
    nbr_classes = len(gt.columns)
    mMCC = 0
    for class_idx in range(nbr_classes):
        mMCC += matthews_corrcoef(gt['gt_'+str(class_idx)], predictions['pred_'+str(class_idx)]>0.5)
    return mMCC/nbr_classes

def createExpFolderandCodeList(save_path,files=[]):
    #result folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    code_folder_path = os.path.join(save_path, 'code')
    if not os.path.exists(code_folder_path):
        os.makedirs(code_folder_path)
    #save code files
    for file_name in os.listdir() + files:
        if not os.path.isdir(file_name):
            copy2('./%s' % file_name, os.path.join(save_path, 'code', file_name))

def binned_scatter_plot(df,x_name,y_name,nbins,name_fig):
    df_sorted = df.sort_values(by=[x_name])
    x = df_sorted[x_name]
    range_data = x.iloc[-1] - x.iloc[0]
    threshold_multiplier = range_data / nbins

    # first pass
    y_averages = []
    current_vals = []
    current_max_x = x.iloc[0] + threshold_multiplier
    for idx, val in enumerate(x):
        if val < current_max_x and idx < len(x)-1:
            current_vals.append(df_sorted[y_name].iloc[idx])
        else:
            if len(current_vals) > 0:
                y_averages.append(np.average(np.array(current_vals)))
            else:
                y_averages.append(y_averages[-1])
            current_vals = []
            current_max_x += threshold_multiplier

    # second pass
    y = []
    current_bin = 1
    current_max_x = x.iloc[0] + threshold_multiplier
    for val in x:
        y.append(y_averages[current_bin-1])
        if val >= current_max_x:
            current_bin += 1
            current_max_x += threshold_multiplier

    # plot and save
    plt.clf()
    plt.scatter(x, y, alpha=0.2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    plt.savefig(os.path.join(save_path, name_fig+'.pdf'))

    return x, y

def scatter_plot(df,x_name,y_name,name_fig):
    plt.clf()
    plt.scatter(df[x_name], df[y_name], alpha=0.2)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()
    plt.savefig(os.path.join(save_path, name_fig + '.pdf'))

def binarize_bool_df(df):
    df = df.replace('True', 1)
    df = df.replace('False', 0)
    df['TP'] = df['TP'].replace('0.0', 0)
    return df

def plot_hist(df,x_name,y_name,lb_clip,nbins,name_fig):
    df_clipped = df[df[x_name] > lb_clip]
    plt.clf()
    plt.hist(df_clipped[df_clipped[y_name] == 1][x_name], color='red', alpha=0.5, bins=nbins, label='TP')
    plt.hist(df_clipped[df_clipped[y_name] == 0][x_name], color='blue', alpha=0.5, bins=nbins, label='FN')
    plt.xlabel('Consistency')
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig(os.path.join(save_path, name_fig+'.pdf'))

if __name__ == '__main__':

    # create exp folder and copy code
    current_exp_id = sys.argv[1]
    save_path = os.path.join(PATH_EXPERIMENTS, current_exp_id)
    createExpFolderandCodeList(save_path)

    # load dataframe
    df1 = pd.read_csv(os.path.join(PATH_EXPERIMENTS, FIRST_EXP, 'stats.csv'))
    df2 = pd.read_csv(os.path.join(PATH_EXPERIMENTS, SECOND_EXP, 'stats.csv'))

    # binarize bool
    df1 = binarize_bool_df(df1)
    df2 = binarize_bool_df(df2)

    # concatenate dataframes
    df1 = df1.set_index('image_id')
    df2 = df2.set_index('image_id')

    df1 = df1.rename(dict((c_name, c_name + '_1') for c_name in df1.columns), axis='columns')
    df2 = df2.rename(dict((c_name, c_name + '_2') for c_name in df2.columns), axis='columns')
    df = pd.concat([df1, df2], axis=1, join='inner')

    # selecting DF of interest
    df_tp_union = df[(df['TP_1'] == 1) & (df['TP_2'] == 1)]
    df_FN1_FP2 = df[(df['TP_1'] != 1) & (df['TP_2'] == 1)]
    df_TP1 = df1[df1['TP_1'] == 1]
    df_FN1 = df1[df1['TP_1'] != 1]
    df_TP2 = df2[df2['TP_2'] == 1]
    df_FN2 = df2[df2['TP_2'] != 1]

    # define dfs to loop over
    df_of_interest = [df_tp_union, df_FN1_FP2, df_TP1, df_FN1, df_TP2, df_FN2]
    df_of_interest_names = ['df_tp_union', 'df_FN1_FP2', 'df_TP1', 'df_FN1', 'df_TP2', 'df_FN2']
    corresponding_models = [[1,2], [1,2], [1], [1], [2], [2]]

    # print all metrics
    metrics = ['consistency', 'surface_bbox', 'overlap_gradcam_bbox',
               'overlap_guidedbackprop_bbox', 'overlap_mask_bbox',
               'overlap_gradcam_bbox/surface_bbox', 'overlap_bbox_gb_bbox', 'BCE']
    for idx, df_current in enumerate(df_of_interest):
        # print len
        print(df_of_interest_names[idx])
        print(len(df_current))
        # print metric means
        for metric in metrics:
            print(metric)
            for model in corresponding_models[idx]:
                print(df_current[metric + '_' + str(model)].mean())
                # if we can compare both model, make stat test
                if corresponding_models[idx] == [1,2]:
                    print(bstrap.bootstrap(np.mean,df_current[metric + '_1'],df_current[metric + '_2'], nbr_runs=1000))
            print('')

    # class-wise metrics
    df_class_wise = pd.DataFrame()
    identified_class = set(df_tp_union['pred_class_name_1'])
    for class_current in identified_class:
        df_current = df_tp_union[df_tp_union['pred_class_name_1'] == class_current]
        df_current = df_current.mean()
        df_current['class'] = class_current
        df_class_wise = df_class_wise.append(df_current,ignore_index=True)
    df_class_wise.to_csv(os.path.join(save_path, 'class_wise_metrics_union_TP.csv'))
    # save summary of df_class_wise (faster to inspect), and compute stat test for selected columns
    selected_columns = ['overlap_gradcam_bbox', 'overlap_bbox_gb_bbox']
    df_class_wise_summary = df_class_wise[['class']+[column + '_'+ str(method) for method in [1,2] for column in selected_columns]]
    for column in selected_columns:
        p_value_list = []
        for class_current in identified_class:
            df_current = df_tp_union[df_tp_union['pred_class_name_1'] == class_current]
            _, _, p_value = bstrap.bootstrap(np.mean,df_current[column + '_1'],df_current[column + '_2'], nbr_runs=1000)
            p_value_list.append(p_value)
        df_class_wise_summary[column+'_p_values'] = p_value_list
    df_class_wise_summary.to_csv(os.path.join(save_path, 'class_wise_metrics_union_TP_summary.csv'))

    # binned scatter plots
    binned_scatter_plot(df1, 'consistency_1', 'TP_1', 1300, 'sup' + '_consistency_TP')
    binned_scatter_plot(df2, 'consistency_2', 'TP_2', 1300, 'alt' + '_consistency_TP')

    # scatter plot
    scatter_plot(df1, 'consistency_1', 'TP_1', 'sup' + '_consistency_TP_scatter')
    scatter_plot(df2, 'consistency_2', 'TP_2', 'alt' + '_consistency_TP_scatter')

    # plot hist
    plot_hist(df1, 'consistency_1', 'TP_1', 0.9, 100, 'sup_hist')
    plot_hist(df2, 'consistency_2', 'TP_2', 0.9, 100, 'alt_hist')

    # show false negative with high consistency
    columns_of_interest = ['gt_class_names','pred_class_name','consistency','overlap_bbox_gb_bbox']
    df_FN1 = df_FN1.sort_values(by=['consistency_1'], ascending=False)
    df_FN2 = df_FN2.sort_values(by=['consistency_2'], ascending=False)
    print('Baseline')
    print(df_FN1[[column + '_1' for column in columns_of_interest]].head())
    print('Proposed Method')
    print(df_FN2[[column + '_2' for column in columns_of_interest]].head())

    # show true positive with low consistency
    df_TP1 = df_TP1.sort_values(by=['consistency_1'])
    df_TP2 = df_TP2.sort_values(by=['consistency_2'])
    print('Baseline')
    print(df_TP1[[column + '_1' for column in columns_of_interest]].head())
    print('Proposed method')
    print(df_TP2[[column + '_2' for column in columns_of_interest]].head())

    # bootstrap mAP------------------------------------------------------------------------------------------
    # load data
    gt = pd.read_csv(os.path.join(PATH_EXPERIMENTS, FIRST_EXP, 'gt.csv'))
    predictions_method1 = pd.read_csv(os.path.join(PATH_EXPERIMENTS, FIRST_EXP, 'predictions.csv'))
    predictions_method2 = pd.read_csv(os.path.join(PATH_EXPERIMENTS, SECOND_EXP, 'predictions.csv'))

    # reformat data to a single pandas dataframe per method with standardized column names
    gt = gt.rename(columns=dict([(column, 'gt_' + column) for column in gt.columns]))
    predictions_method1 = predictions_method1.rename(
        columns=dict([(column, 'pred_' + column) for column in predictions_method1.columns]))
    predictions_method2 = predictions_method2.rename(
        columns=dict([(column, 'pred_' + column) for column in predictions_method2.columns]))
    data_method1 = pd.concat([gt, predictions_method1], axis=1)
    data_method2 = pd.concat([gt, predictions_method2], axis=1)

    # 4. compare method 1 and 2 (same code as example 1)
    stats_method1, stats_method2, p_value = bstrap.bootstrap(compute_mMCC, data_method1, data_method2, nbr_runs=100)
    print('mMCC')
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    stats_method1, stats_method2, p_value = bstrap.bootstrap(compute_mAP, data_method1, data_method2, nbr_runs=100)
    print('mAP')
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    stats_method1, stats_method2, p_value = bstrap.bootstrap(compute_mF1, data_method1, data_method2, nbr_runs=100)
    print('mF1')
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # class-wise AP and F1
    df_f1 = pd.DataFrame()
    nbr_classes = len(gt.columns)
    for class_idx in range(nbr_classes):
            print('class '+str(class_idx))
            current_gt = gt['gt_'+str(class_idx)]
            current_pred1 = predictions_method1['pred_'+str(class_idx)]
            current_pred2 = predictions_method2['pred_'+str(class_idx)]
            current_gt = current_gt.rename('gt_0')
            current_pred1 = current_pred1.rename('pred_0')
            current_pred2 = current_pred2.rename('pred_0')
            data_method1 = pd.concat([current_gt, current_pred1], axis=1)
            data_method2 = pd.concat([current_gt, current_pred2], axis=1)
            stats_method1, stats_method2, p_value = bstrap.bootstrap(compute_mAP, data_method1, data_method2, nbr_runs=100)
            print('AP')
            print(stats_method1)
            print(stats_method2)
            print(p_value)
            stats_method1, stats_method2, p_value = bstrap.bootstrap(compute_mF1, data_method1, data_method2, nbr_runs=100)
            print('F1')
            df_f1 = df_f1.append({'class': class_idx,
                                  'F1 method 1': stats_method1['avg_metric'],
                                  'F1 method 2': stats_method2['avg_metric'],
                                  'p_value': p_value}, ignore_index=True)
            print(stats_method1)
            print(stats_method2)
            print(p_value)

    # save df f1
    df_f1.to_csv(os.path.join(save_path, 'F1.csv'))

    # save combine dataframe
    df.to_csv(os.path.join(save_path, 'combined_stats.csv'))



