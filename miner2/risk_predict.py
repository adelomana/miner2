import os
import pandas as pd
import numpy as np
import warnings

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import json

from miner2 import survival, preprocess, biclusters, miner


def generate_prediction_matrix(srv, mtrx, high_risk_cutoff=0.2):

    srv = srv.copy()
    srv.sort_values(by='GuanScore',ascending=False,inplace=True)

    highRiskSamples = list(srv.index[0:int(high_risk_cutoff*srv.shape[0])])
    lowRiskSamples = list(srv.index[int(high_risk_cutoff*srv.shape[0]):])

    hrFlag = pd.DataFrame(np.ones((len(highRiskSamples),1)).astype(int))
    hrFlag.index = highRiskSamples
    hrFlag.columns = ["HR_FLAG"]

    lrFlag = pd.DataFrame(np.zeros((len(lowRiskSamples),1)).astype(int))
    lrFlag.index = lowRiskSamples
    lrFlag.columns = ["HR_FLAG"]

    # This crashes when none of the samples in highRiskSamples are
    # found in the mtrx columns, so we fall back to return None
    try:
        hrMatrix = pd.concat([mtrx.loc[:,highRiskSamples].T,hrFlag],axis=1)
        hrMatrix.columns = np.array(hrMatrix.columns).astype(str)
    except:
        hrMatrix = None

    # This crashes when none of the samples in lowRiskSamples are
    # found in the mtrx columns, so we fall back to return None
    try:
        lrMatrix = pd.concat([mtrx.loc[:,lowRiskSamples].T,lrFlag],axis=1)
        lrMatrix.columns = np.array(lrMatrix.columns).astype(str)
    except:
        lrMatrix = None

    return hrMatrix, lrMatrix


def prediction_matrix(membership_datasets,survival_datasets,high_risk_cutoff=0.20):
    hr_matrices = []
    lr_matrices = []

    for i in range(len(membership_datasets)):
        hrmatrix, lrmatrix = generate_prediction_matrix(survival_datasets[i],
                                                        membership_datasets[i],
                                                        high_risk_cutoff=high_risk_cutoff)
        if hrmatrix is not None:
            hr_matrices.append(hrmatrix)
        if lrmatrix is not None:
            lr_matrices.append(lrmatrix)

    print('# hr_matrices: %d' % len(hr_matrices))
    print('# lr_matrices: %d' % len(lr_matrices))
    hrMatrixCombined = pd.concat(hr_matrices,axis=0)
    lrMatrixCombined = pd.concat(lr_matrices,axis=0)
    predictionMat = pd.concat([hrMatrixCombined,lrMatrixCombined],axis=0)
    print(predictionMat.shape)
    return predictionMat


def risk_stratification(lbls, mtrx, guan_srv, survival_tag, classifier,
                        resultsDirectory=None, plot_all=False, guan_rank=False,
                        high_risk_cutoffs=None, plot_any=True):
    warnings.filterwarnings("ignore")

    guan_srv = guan_srv.loc[list(set(guan_srv.index)&set(mtrx.columns)),:]
    if plot_any is True:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
        f.tight_layout(pad=1.08)
        f.set_figwidth(10)
        f.set_figheight(4)

    predicted_probabilities = classifier.predict_proba(np.array(mtrx.T))[:,1]
    predicted_probabilities_df = pd.DataFrame(predicted_probabilities)
    predicted_probabilities_df.index = mtrx.columns
    predicted_probabilities_df.columns = ["probability_high_risk"]

    srv = guan_srv.iloc[:,0:2]
    srv_observed = guan_srv[guan_srv.iloc[:,1]==1]
    srv_unobserved = guan_srv[guan_srv.iloc[:,1]==0]

    if high_risk_cutoffs is None:
        high_risk_cutoffs = np.percentile(list(srv_observed.iloc[:,0]),[10,15,20,25,30])

    aucs = []
    cutoffs = []
    tpr_list = []
    fpr_list = []
    prec = []
    rec = []
    for i in range(len(high_risk_cutoffs)):#range(0,max(list(srv.iloc[:,0]))+interval,interval):
        if guan_rank is True:
            percentile = 10+i*(20.0/(len(high_risk_cutoffs)-1))
            number_samples = int(np.ceil(guan_srv.shape[0]*(percentile/100.0)))
            cutoff = guan_srv.iloc[number_samples,0]
            true_hr = guan_srv.index[0:number_samples]
            true_lr = guan_srv.index[number_samples:]
            srv_total = guan_srv.copy()

        elif guan_rank is not True:
            cutoff = high_risk_cutoffs[i]
            srv_extended = srv_unobserved[srv_unobserved.iloc[:,0]>=cutoff]
            if srv_extended.shape[0] > 0:
                srv_total = pd.concat([srv_observed,srv_extended],axis=0)
            elif srv_extended.shape[0] == 0:
                srv_total = srv_observed.copy()
            pass_index = np.where(srv_total.iloc[:,0]<cutoff)[0]
            true_hr = []
            if len(pass_index) > 0:
                true_hr = srv_total.index[pass_index]
            true_lr = list(set(srv_total.index)-set(true_hr))


        # use predicted_probabilities_df against true_hr, true_lr to
        # compute precision and recall from sklearn.metrics
        tpr = []
        fpr = []
        precisions = []
        recalls = []
        for threshold in np.arange(0,1.02,0.01):
            model_pp = predicted_probabilities_df.copy()
            model_pp = model_pp.loc[list(set(model_pp.index)&set(srv_total.index)),:]
            predicted_hr = model_pp.index[np.where(model_pp.iloc[:,0]>=threshold)[0]]
            predicted_lr = list(set(model_pp.index)-set(predicted_hr))

            tp = set(true_hr)&set(predicted_hr)
            fp = set(true_lr)&set(predicted_hr)
            allpos = set(true_hr)
            tn = set(true_lr)&set(predicted_lr)
            fn = set(true_hr)&set(predicted_lr)
            allneg = set(true_lr)

            if len(allpos) == 0:
                tp_rate = 0
                precision = 0
                recall=0
            elif len(allpos) > 0:
                tp_rate = len(tp)/float(len(allpos))

                if len(tp) + len(fp) > 0:
                    precision = len(tp)/float(len(tp) + len(fp))
                elif len(tp) + len(fp) == 0:
                    precision = 0

                if len(tp) +len(fn) > 0:
                    recall = len(tp)/float(len(tp) +len(fn))
                elif len(tp) +len(fn) == 0:
                    recall = 0
            if len(allneg) == 0:
                tn_rate = 0
            elif len(allneg) > 0:
                tn_rate = len(tn)/float(len(allneg))

            tpr.append(tp_rate)
            fpr.append(1-tn_rate)

            precisions.append(precision)
            recalls.append(recall)

        if plot_all:
            plt.figure()
            plt.plot(fpr,tpr)
            plt.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--")
            plt.ylim(-0.05,1.05)
            plt.xlim(-0.05,1.05)
            plt.title('ROC curve, cutoff = {:d}'.format(int(cutoff)))

        area = metrics.auc(fpr,tpr)
        aucs.append(area)
        cutoffs.append(cutoff)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        prec.append(precisions)
        rec.append(recalls)

    integrated_auc = np.mean(aucs)

    #print('classifier has integrated AUC of {:.3f}'.format(integrated_auc))

    tpr_stds = np.std(np.vstack(tpr_list),axis=0)
    tpr_means = np.mean(np.vstack(tpr_list),axis=0)
    fpr_means = np.mean(np.vstack(fpr_list),axis=0)

    if plot_any:
        ax1.fill_between(fpr_means, tpr_means-tpr_stds, tpr_means+tpr_stds,color=[0,0.4,0.6],alpha=0.3)
        ax1.plot(fpr_means,tpr_means,color=[0,0.4,0.6],LineWidth=1.5)
        ax1.plot(np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),"--",color=[0.2,0.2,0.2])
        ax1.set_ylim(-0.05,1.05)
        ax1.set_xlim(-0.05,1.05)
        ax1.set_title('Integrated AUC = {:.2f}'.format(integrated_auc))
        ax1.set_ylabel('Sensitivity',FontSize=14)
        ax1.set_xlabel('1-Specificity',FontSize=14)

    hr_dt = mtrx.columns[lbls.astype(bool)]
    lr_dt = mtrx.columns[(1-lbls).astype(bool)]

    kmTag = "decision_tree"
    kmFilename = ("_").join([survival_tag,kmTag,"high-risk",".pdf"])

    groups = [hr_dt,lr_dt]
    labels = ["High-risk","Low-risk"]

    cox_vectors = []
    srv_set = set(srv.index)
    for i in range(len(groups)):
        group = groups[i]
        patients = list(set(group)&srv_set)
        tmp_df = pd.DataFrame(np.zeros(srv.shape[0]))
        tmp_df.index = srv.index
        tmp_df.columns = [labels[i]]
        tmp_df.loc[patients,labels[i]] = 1
        cox_vectors.append(tmp_df)

    pre_cox = pd.concat(cox_vectors,axis=1).T
    pre_cox.head(5)

    cox_dict = survival.parallel_member_survival_analysis(membershipDf=pre_cox,
                                                          numCores=1,
                                                          survivalPath="",
                                                          survivalData=srv)
    #print('Risk stratification of '+survival_tag+' has Hazard Ratio of {:.2f}'.format(cox_dict['High-risk'][0]))

    hazard_ratio = cox_dict['High-risk'][0]
    if plot_any is True:
        if resultsDirectory is not None:
            plotName = os.path.join(resultsDirectory,kmFilename)
            survival.kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=plotName)
            plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

        elif resultsDirectory is None:
            survival.kmplot(srv=srv,groups=groups,labels=labels,xlim_=(-100,1750),filename=None)
            plt.title('Dataset: '+survival_tag+'; HR: {:.2f}'.format(cox_dict['High-risk'][0]))

    return aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec


def generate_predictor(membership_datasets, survival_datasets, dataset_labels,
                       iterations=20, method='xgboost', n_estimators=100,
                       output_directory=None, best_state=None, test_only=True,
                       separate_results=True, metric='roc_auc', class1_proportion=0.20,
                       test_proportion=0.35,colsample_bytree=1,subsample=1):
    """
    Computes a classifier object from the specified input data sets.
    The result is either an xgboost.XGBClassifier or a sklearn.tree.DecisionTreeClassifier
    """
    if method == 'xgboost':
        # prevents kernel from dying when running XGBClassifier
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        from xgboost import XGBClassifier

    elif method=='decisionTree':
        from sklearn.tree import DecisionTreeClassifier

    predictionMat = prediction_matrix(membership_datasets,survival_datasets,
                                      high_risk_cutoff=class1_proportion)

    X = np.array(predictionMat.iloc[:,0:-1])
    Y = np.array(predictionMat.iloc[:,-1])
    X = X.astype('int')
    Y = Y.astype('int')

    samples_ = np.array(predictionMat.index)

    if best_state is None:
        mean_aucs = []
        mean_hrs = []
        pct_labeled = []
        for rs in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_proportion, random_state = rs)
            X_train_columns, X_test_columns, y_train_samples, y_test_samples = train_test_split(X, samples_, test_size = test_proportion, random_state = rs)

            train_datasets = []
            test_datasets = []
            for td in range(len(membership_datasets)):
                dataset = membership_datasets[td]
                train_members = list(set(dataset.columns)&set(y_train_samples))
                test_members = list(set(dataset.columns)&set(y_test_samples))
                train_datasets.append(dataset.loc[:,train_members])
                test_datasets.append(dataset.loc[:,test_members])

            if method=='xgboost':
                eval_set = [(X_train, y_train), (X_test, y_test)]
                clf = XGBClassifier(n_jobs=1,random_state=12,n_estimators=n_estimators,colsample_bytree=colsample_bytree,subsample=subsample)
                clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)
            elif method=='decisionTree':
                clf = DecisionTreeClassifier(criterion = "gini", random_state = 12, max_depth=6, min_samples_leaf=5)
                clf.fit(X_train, y_train)

            train_predictions = []
            test_predictions = []
            for p in range(len(membership_datasets)):
                tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
                tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
                train_predictions.append(tmp_train_predictions)
                test_predictions.append(tmp_test_predictions)

            if test_only:
                scores = []
                hrs = []
                for j in range(len(test_datasets)):
                    mtrx = test_datasets[j]
                    guan_srv = survival_datasets[j]
                    survival_tag = dataset_labels[j]
                    lbls = test_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)
                    pct_labeled.append(100*sum(lbls)/float(len(lbls)))

                mean_auc = np.mean(scores)
                mean_hr = np.mean(hrs)
                mean_aucs.append(mean_auc)
                mean_hrs.append(mean_hr)
                print(rs,mean_auc,mean_hr)

            elif test_only is False:
                scores = []
                hrs = []
                for j in range(len(test_datasets)):
                    mtrx = test_datasets[j]
                    guan_srv = survival_datasets[j]
                    survival_tag = dataset_labels[j]
                    lbls = test_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)

                    mtrx = train_datasets[j]
                    lbls = train_predictions[j]
                    aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
                    score = np.mean(aucs)
                    scores.append(score)
                    hrs.append(hazard_ratio)

                mean_auc = np.mean(scores)
                mean_hr = np.mean(hrs)
                mean_aucs.append(mean_auc)
                mean_hrs.append(mean_hr)
                print(rs,mean_auc,mean_hr)

        if metric == 'roc_auc':
            best_state = np.argsort(np.array(mean_aucs))[-1]
        elif metric == 'hazard_ratio':
            best_state = np.argsort(np.array(mean_hrs))[-1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = test_proportion, random_state = best_state)
    X_train_columns, X_test_columns, y_train_samples, y_test_samples = train_test_split(X, samples_, test_size = test_proportion, random_state = best_state)

    train_datasets = []
    test_datasets = []
    for td in range(len(membership_datasets)):
        dataset = membership_datasets[td]
        train_members = list(set(dataset.columns)&set(y_train_samples))
        test_members = list(set(dataset.columns)&set(y_test_samples))
        train_datasets.append(dataset.loc[:,train_members])
        test_datasets.append(dataset.loc[:,test_members])

    if method=='xgboost':
        eval_set = [(X_train, y_train), (X_test, y_test)]
        clf = XGBClassifier(n_jobs=1,random_state=12,n_estimators=n_estimators,colsample_bytree=colsample_bytree,subsample=subsample)
        clf.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)
    elif method=='decisionTree':
        clf = DecisionTreeClassifier(criterion = "gini", random_state = 12, max_depth=6, min_samples_leaf=5)
        clf.fit(X_train, y_train)

    train_predictions = []
    test_predictions = []
    for p in range(len(membership_datasets)):
        tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
        tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
        train_predictions.append(tmp_train_predictions)
        test_predictions.append(tmp_test_predictions)

    mean_aucs = []
    mean_hrs = []
    if test_only:
        scores = []
        hrs = []
        pct_labeled = []
        for j in range(len(test_datasets)):
            mtrx = test_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = test_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)
            pct_labeled.append(100*sum(lbls)/float(len(lbls)))

        mean_auc = np.mean(scores)
        mean_hr = np.mean(hrs)
        mean_aucs.append(mean_auc)
        mean_hrs.append(mean_hr)
        precision_matrix = np.vstack(prec)
        recall_matrix = np.vstack(rec)
        print(best_state,mean_auc,mean_hr)

    else:
        scores = []
        hrs = []
        pct_labeled = []
        for j in range(len(test_datasets)):
            mtrx = test_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = test_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)
            pct_labeled.append(100*sum(lbls)/float(len(lbls)))

            mtrx = train_datasets[j]
            lbls = train_predictions[j]
            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=False)
            score = np.mean(aucs)
            scores.append(score)
            hrs.append(hazard_ratio)

        mean_auc = np.mean(scores)
        mean_hr = np.mean(hrs)
        mean_aucs.append(mean_auc)
        mean_hrs.append(mean_hr)
        precision_matrix = np.vstack(prec)
        recall_matrix = np.vstack(rec)
        print(best_state,mean_auc,mean_hr)

    train_predictions = []
    test_predictions = []
    predictions = []
    #add print for percent labeled high-risk
    for p in range(len(membership_datasets)):
        tmp_train_predictions = clf.predict(np.array(train_datasets[p].T))
        tmp_test_predictions = clf.predict(np.array(test_datasets[p].T))
        tmp_predictions = clf.predict(np.array(membership_datasets[p].T))
        train_predictions.append(tmp_train_predictions)
        test_predictions.append(tmp_test_predictions)
        predictions.append(tmp_predictions)

    if not separate_results:
        for j in range(len(membership_datasets)):
            mtrx = membership_datasets[j]
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]
            lbls = predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'survival_predictions.pdf'])),bbox_inches='tight')

    else:
        for j in range(len(membership_datasets)):
            guan_srv = survival_datasets[j]
            survival_tag = dataset_labels[j]

            mtrx = train_datasets[j]
            lbls = train_predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of training population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'training_survival_predictions.pdf'])),bbox_inches='tight')

            mtrx = test_datasets[j]
            lbls = test_predictions[j]

            percent_classified_hr = 100*sum(lbls)/float(len(lbls))
            print('classified {:.1f} percent of test population as high-risk'.format(percent_classified_hr))

            aucs, cutoffs, tpr_list, fpr_list, hazard_ratio, prec, rec = risk_stratification(lbls,mtrx,guan_srv,survival_tag,clf,guan_rank=False,resultsDirectory=None,plot_all=False,plot_any=True)
            if output_directory is not None:
                plt.savefig(os.path.join(output_directory,('_').join([survival_tag,method,metric,'test_survival_predictions.pdf'])),bbox_inches='tight')

    nextIteration = []
    class0 = []
    class1 = []
    for p in range(len(membership_datasets)):
        tmp_predictions = clf.predict(np.array(membership_datasets[p].T))
        tmp_class_0 = membership_datasets[p].columns[(1-np.array(tmp_predictions)).astype(bool)]
        tmp_class_1 = membership_datasets[p].columns[np.array(tmp_predictions).astype(bool)]
        nextIteration.append(membership_datasets[p].loc[:,tmp_class_0])
        class0.append(tmp_class_0)
        class1.append(tmp_class_1)

    print(best_state)

    if best_state is not None:
        return clf, class0, class1, mean_aucs, mean_hrs, pct_labeled, precision_matrix, recall_matrix

    return clf, class0, class1, mean_aucs, mean_hrs, pct_labeled, precision_matrix, recall_matrix


def load_test_set(exp_path, reference_modules, conv_table_path):
    exp_data = pd.read_csv(exp_path, index_col=0,header=0)
    # WHY ???? does preprocess.identifier_conversion not work ? TODO
    exp_data, _ = miner.identifierConversion(exp_data, conv_table_path)
    exp_data = preprocess.zscore(exp_data)
    bkgd = preprocess.background_df(exp_data)

    overexp_members = biclusters.make_membership_dictionary(reference_modules, bkgd, label=2, p=0.1)
    overexp_members_matrix = biclusters.membership_to_incidence(overexp_members, exp_data)

    underexp_members = biclusters.make_membership_dictionary(reference_modules, bkgd, label=0, p=0.1)
    underexp_members_matrix = biclusters.membership_to_incidence(underexp_members, exp_data)
    return overexp_members_matrix, underexp_members_matrix


def get_survival_subset(surv_df, label):
    sub_surv = surv_df[surv_df.index==label]
    sub_surv.index = sub_surv.iloc[:,0]
    sub_surv_df = sub_surv.loc[:,["D_PFS","D_PFS_FLAG"]]
    sub_surv_df.columns = ["duration","observed"]

    km_df = survival.km_analysis(survivalDf=sub_surv_df,
                                 durationCol="duration", statusCol="observed")
    return survival.guan_rank(kmSurvival=km_df)


def read_datasets(input_spec):
    """
    Reads a datasets dictionary from the specified input specification
    """
    with open(input_spec['regulons']) as infile:
        regulon_modules = json.load(infile)

    prim_survival = pd.read_csv(input_spec['primary_survival'], index_col=0, header=0)
    prim_survival_df = prim_survival.iloc[:,0:2]
    prim_survival_df.columns = ["duration", "observed"]

    test_survival = pd.read_csv(input_spec['test_survival'], index_col=0,header=0)

    datasets = []
    for dataset in input_spec['datasets']:
        omm, umm = load_test_set(dataset['exp'], regulon_modules, dataset['idmap'])
        dataset_obj = {
            'omm': omm,
            'umm': umm,
            'label': dataset['label'],
            'primary': dataset['primary'],
            'dfr': (omm - umm)
        }
        if dataset['primary']:
            dataset_obj['km'] = survival.km_analysis(survivalDf=prim_survival_df,
                                                     durationCol="duration", statusCol="observed")
            dataset_obj['gs'] = survival.guan_rank(kmSurvival=dataset_obj['km'])
        else:
            print('subset of global survival')
            dataset_obj['gs'] = get_survival_subset(test_survival, dataset['survival_subset'])

        datasets.append(dataset_obj)
    return datasets

