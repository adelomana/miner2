from lifelines import KaplanMeierFitter
from scipy import stats
import numpy
import pandas
from lifelines import CoxPHFitter
import matplotlib.pyplot as plt
from miner2 import util


# =============================================================================
# Functions used for survival analysis
# =============================================================================

def km_analysis(survivalDf, durationCol, statusCol, saveFile=None):
    kmf = KaplanMeierFitter()
    kmf.fit(survivalDf.loc[:,durationCol],survivalDf.loc[:,statusCol])
    survFunc = kmf.survival_function_

    m, b, r, p, e = stats.linregress(list(survFunc.index),survFunc.iloc[:,0])

    survivalDf = survivalDf.sort_values(by=durationCol)
    ttpfs = numpy.array(survivalDf.loc[:,durationCol])
    survTime = numpy.array(survFunc.index)
    survProb = []

    for i in range(len(ttpfs)):
        date = ttpfs[i]
        if date in survTime:
            survProb.append(survFunc.loc[date,"KM_estimate"])
        elif date not in survTime:
            lbix = numpy.where(numpy.array(survFunc.index)<date)[0][-1]
            est = 0.5*(survFunc.iloc[lbix,0]+survFunc.iloc[lbix+1,0])
            survProb.append(est)

    kmEstimate = pandas.DataFrame(survProb)
    kmEstimate.columns = ["kmEstimate"]
    kmEstimate.index = survivalDf.index

    pfsDf = pandas.concat([survivalDf,kmEstimate],axis=1)

    if saveFile is not None:
        pfsDf.to_csv(saveFile)

    return pfsDf


def guan_rank(kmSurvival, saveFile=None):
    gScore = []
    for A in range(kmSurvival.shape[0]):
        aScore = 0
        aPfs = kmSurvival.iloc[A,0]
        aStatus = kmSurvival.iloc[A,1]
        aProbPFS = kmSurvival.iloc[A,2]
        if aStatus == 1:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]
                if bPfs > aPfs:
                    aScore+=1
                if bPfs <= aPfs:
                    if bStatus == 0:
                        aScore+=aProbPFS/bProbPFS
                if bPfs == aPfs:
                    if bStatus == 1:
                        aScore+=0.5
        elif aStatus == 0:
            for B in range(kmSurvival.shape[0]):
                if B == A:
                    continue
                bPfs = kmSurvival.iloc[B,0]
                bStatus = kmSurvival.iloc[B,1]
                bProbPFS = kmSurvival.iloc[B,2]
                if bPfs >= aPfs:
                    if bStatus == 0:
                        tmp = 1-0.5*bProbPFS/aProbPFS
                        aScore+=tmp
                    elif bStatus == 1:
                        tmp = 1-bProbPFS/aProbPFS
                        aScore+=tmp
                if bPfs < aPfs:
                    if bStatus == 0:
                        aScore+=0.5*aProbPFS/bProbPFS
        gScore.append(aScore)

    GuanScore = pandas.DataFrame(gScore)
    GuanScore = GuanScore/float(max(gScore))
    GuanScore.index = kmSurvival.index
    GuanScore.columns = ["GuanScore"]
    survivalData = pandas.concat([kmSurvival,GuanScore],axis=1)
    survivalData.sort_values(by="GuanScore", ascending=False, inplace=True)

    if saveFile is not None:
        survivalData.to_csv(saveFile)

    return survivalData



def survival_membership_analysis(task):

    start, stop = task[0]
    membershipDf,SurvivalDf = task[1]

    overlapPatients = list(set(membershipDf.columns)&set(SurvivalDf.index))
    if len(overlapPatients) == 0:
        print("samples are not represented in the survival data")
        return
    Survival = SurvivalDf.loc[overlapPatients,SurvivalDf.columns[0:2]]

    coxResults = {}
    keys = membershipDf.index[start:stop]
    ct=0
    for key in keys:
        ct+=1
        if ct%10==0:
            print(ct)
        try:
            memberVector = pandas.DataFrame(membershipDf.loc[key,overlapPatients])
            Survival2 = pandas.concat([Survival,memberVector],axis=1)
            Survival2.sort_values(by=Survival2.columns[0],inplace=True)

            cph = CoxPHFitter()
            cph.fit(Survival2, duration_col=Survival2.columns[0], event_col=Survival2.columns[1])

            tmpcph = cph.summary

            cox_hr = tmpcph.loc[key,"z"]
            cox_p = tmpcph.loc[key,"p"]
            coxResults[key] = (cox_hr, cox_p)
        except:
            coxResults[key] = (0, 1)
    return coxResults


def parallel_member_survival_analysis(membershipDf, numCores=5, survivalPath=None,
                                      survivalData=None):

    if survivalData is None:
        survivalData = pandas.read_csv(survivalPath,index_col=0,header=0)

    taskSplit = util.split_for_multiprocessing(membershipDf.index, numCores)
    taskData = (membershipDf, survivalData)
    tasks = [[taskSplit[i],taskData] for i in range(len(taskSplit))]
    coxOutput = util.multiprocess(survival_membership_analysis, tasks)
    return util.condense_output(coxOutput)


def combined_states(groups, ranked_groups, survivalDf, minSamples=4, maxStates=7):
    high_risk_indices = []
    for i in range(1, len(ranked_groups) + 1):
        tmp_group = ranked_groups[-i]
        tmp_len = len(set(survivalDf.index) & set(groups[tmp_group]))
        if tmp_len >= minSamples:
            high_risk_indices.append(tmp_group)
        if len(high_risk_indices) >= maxStates:
            break

    combinations_high = []
    for i in range(len(high_risk_indices)-1):
        combinations_high.append(high_risk_indices[0: i + 1])

    low_risk_indices = []
    for i in range(len(ranked_groups)):
        tmp_group = ranked_groups[i]
        tmp_len = len(set(survivalDf.index) & set(groups[tmp_group]))
        if tmp_len >= minSamples:
            low_risk_indices.append(tmp_group)
        if len(low_risk_indices) >= maxStates:
            break

    combinations_low = []
    for i in range(len(low_risk_indices) - 1):
        combinations_low.append(low_risk_indices[0: i + 1])

    combined_states_high = []
    for i in range(len(combinations_high)):
        tmp = []
        for j in range(len(combinations_high[i])):
            tmp.append(groups[combinations_high[i][j]])
        combined_states_high.append(numpy.hstack(tmp))

    combined_states_low = []
    for i in range(len(combinations_low)):
        tmp = []
        for j in range(len(combinations_low[i])):
            tmp.append(groups[combinations_low[i][j]])
        combined_states_low.append(numpy.hstack(tmp))

    combined_states = numpy.concatenate([combined_states_high, combined_states_low])
    combined_indices_high = ["&".join(numpy.array(combinations_high[i]).astype(str))
                             for i in range(len(combinations_high))]
    combined_indices_low = ["&".join(numpy.array(combinations_low[i]).astype(str))
                            for i in range(len(combinations_low))]
    combined_indices = numpy.concatenate([combined_indices_high, combined_indices_low])

    return combined_states, combined_indices


def kmplot(srv,groups,labels,xlim_=None,filename=None,color=None,lw=1):

    for group in groups:
        try:
            patients = list(set(srv.index)&set(group))
            kmDf = kmAnalysis(survivalDf=srv.loc[patients,["duration","observed"]],
                              durationCol="duration",statusCol="observed")
            subset = kmDf[kmDf.loc[:,"observed"] == 1]
            duration = numpy.concatenate([numpy.array([0]), numpy.array(subset.loc[:,"duration"])])
            kme = numpy.concatenate([numpy.array([1]), numpy.array(subset.loc[:,"kmEstimate"])])
            if color is not None:
                plt.step(duration, kme, color=color, LineWidth=lw)
            elif color is None:
                plt.step(duration, kme, LineWidth=lw)
        except:
            continue

    try:
        axes = plt.gca()
        axes.set_axis_bgcolor([1,1,1])
        axes.grid(False)
        axes.spines['bottom'].set_color('0.25')
        axes.spines['top'].set_color('0.25')
        axes.spines['right'].set_color('0.25')
        axes.spines['left'].set_color('0.25')
        #axes.set_xlim([xmin,xmax])
        axes.set_ylim([0,1.09])
        if xlim_ is not None:
            axes.set_xlim([xlim_[0],xlim_[1]])
        #axes.set_title("Progression-free survival",FontSize=16)
        axes.set_ylabel("Disease-free",FontSize=14)
        axes.set_xlabel("Days",FontSize=14)
        if labels is not None:
            axes.legend(labels = labels,fontsize='x-small',ncol=3,loc='upper right')
        if filename is not None:
            plt.savefig(filename,bbox_inches="tight")
    except:
        print('Could not complete Kaplan-Meier analysis and plotting')
