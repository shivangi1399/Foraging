import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from parse_logfile import TextLog

import sys

def end_of_day(logfile="/as/projects/OWzeronoise/Reward_Training/001/20230123/RWD/024/2023_01_23-11_17_54_001_RWD_024_RNT_GrassyLandscapeWithBackgroundDark.log", save_data = True, plotting = True, rob_plots=True):
    
    TrialInfo = make_trial_info(logfile)
    
    if save_data:
        data_folder = create_data_folder(logfile)
        save_trial_info(logfile, TrialInfo, data_folder)
        
        
    if plotting or rob_plots:
        data_folder = create_data_folder(logfile)
        figure_folder = create_figure_folder(logfile, data_folder)
            
    if plotting:
        make_plots(TrialInfo, figure_folder)
        
    if rob_plots:
        make_rob_plots(TrialInfo, figure_folder)
    
    
    

def create_data_folder(logfile, root_save = '/cs/departmentN5/behaviour/'):

    # make save folder
    if "/OWzeronoise/".lower() in logfile.lower() or "/ZeroNoise_MOUSE/".lower() in logfile.lower():
        root_save = os.path.join(root_save, 'mouse')
    elif "/MWzeronoise/".lower() in logfile.lower() or "/ZeroNoise_MONKEY/".lower() in logfile.lower():
        root_save = os.path.join(root_save, 'monkey')
    else:
        print('species unrecognised')
        
    data_folder = root_save
    
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)
    
        
    return data_folder

def create_figure_folder(logfile, data_folder):
    head, tail = os.path.split(logfile)
    figure_folder= os.path.join(data_folder, tail[:-4])
    if not os.path.isdir(figure_folder):
        os.makedirs(figure_folder)
    
    return figure_folder
        
def make_trial_info(logfile):
    
    # parse logfile
    with TextLog(logfile) as log_AC:
        log_evt, unused, unused, unused = log_AC.parse_eventmarkers()
        if np.any(log_evt == 3042):
            start = 3042
        else:
            start = 3000
        TrialInfo = log_AC.get_info_per_trial(return_loc=True,choose_trials=False,return_eventmarkers=True, start=start)
        #TrialInfo = log_AC.get_info_per_trial(return_loc=True,choose_trials=['BaseBlock=4'],return_eventmarkers=True, start=3042)
        
    return TrialInfo
        
def save_trial_info(logfile, TrialInfo, data_folder):

    # save parsed data
    head, tail = os.path.split(logfile)

    with open(os.path.join(data_folder, tail[:-4]+'.pickle'), 'wb') as f:
        pickle.dump(TrialInfo, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
        
        
def make_plots(TrialInfo, figure_folder):
    
    # make Overview figure
    fig, axs = plt.subplots(2,3, figsize = [10, 5])
    axs = axs.ravel()
    
    n_corr = np.sum(TrialInfo['Correct'])
    n_trials = len(TrialInfo['Correct'])
    n_wrong = np.sum(TrialInfo['Wrong'])
    
    axs[0].bar(1, n_corr/n_trials, color='green')
    axs[0].bar(2, n_wrong/n_trials, color='red')
    axs[0].set_xticks([1,2])
    axs[0].set_xticklabels(['Correct','Wrong'])
    axs[0].set_title('Correct, Wrong')
    
    conds = np.unique(TrialInfo['BaseConditionTarget'])
    morphs = np.unique(TrialInfo['MorphTarget'])
    
    if len(conds) > 1:
        corr_wrong_barplot(TrialInfo, TrialInfo['BaseConditionTarget'], axs[1])
        axs[1].plot()
    else:
        axs[1].text(0,0, 'No conditions found')
    axs[1].set_title('Conditions- Distractor')
        
    if len(morphs) > 1:
        corr_wrong_barplot(TrialInfo, TrialInfo['MorphTarget'], axs[2])
    else:
        axs[2].text(0,0, 'No morphs found')
    axs[2].set_title('Morphs- Target')

    conds = np.unique(TrialInfo['BaseConditionDistractor'])
    morphs = np.unique(TrialInfo['MorphDistractor'])
    
    if len(conds) > 1:
        corr_wrong_barplot(TrialInfo, TrialInfo['BaseConditionDistractor'], axs[2])
        axs[2].plot()
    else:
        axs[2].text(0,0, 'No conditions found')
    axs[2].set_title('Conditions- Distractor')
        
    if len(morphs) > 1:
        corr_wrong_barplot(TrialInfo, TrialInfo['MorphDistractor'], axs[3])
    else:
        axs[3].text(0,0, 'No morphs found')
    axs[3].set_title('Morphs- Distractor')
    
    for itrl, locs in enumerate(TrialInfo['Location']):
        if TrialInfo['Correct'][itrl]:
            axs[4].plot(locs[:,1]-locs[0,1], locs[:,0]-locs[0,0], 'g', alpha=0.2)
        else:
            axs[4].plot(locs[:,1]-locs[0,1], locs[:,0]-locs[0,0], 'r', alpha=0.2)
    axs[4].set_title('Paths')
            
    fig.savefig(os.path.join(figure_folder, 'Overview.svg'), transparent = True)
    fig.savefig(os.path.join(figure_folder, 'Overview.png'))
    
    # plot paths split by right or left
    
    fig, axs = plt.subplots(1,2, figsize = [10, 5], sharex=True, sharey=True)
    axs = axs.ravel()
    
    inc = TrialInfo['Right'] # 1 if right, 0 if left
    for itrl, locs in enumerate(TrialInfo['Location']):
        if TrialInfo['Right'][itrl]:
            ax = axs[0]
        else:
            ax = axs[1]
            
        if TrialInfo['Correct'][itrl]:
            ax.plot(locs[:,1]-locs[0,1], locs[:,0]-locs[0,0], 'g', alpha=0.2)
        else:
            ax.plot(locs[:,1]-locs[0,1], locs[:,0]-locs[0,0], 'r', alpha=0.2)
            
    axs[0].set_title('Right')
    axs[1].set_title('Left')
    
    fig.savefig(os.path.join(figure_folder, 'RightLeft_paths.svg'), transparent = True)
    fig.savefig(os.path.join(figure_folder, 'RightLeft_paths.png'))

    # plot paths split by trial number
    subplot_rowcol = np.ceil(np.sqrt(n_trials//20 + 1)).astype(int)
    fig, axs = plt.subplots(subplot_rowcol, subplot_rowcol, figsize=[20,20], sharex=True, sharey=True)
    axs = axs.ravel()

    for itrl, locs in enumerate(TrialInfo['Location']):
        ax = axs[itrl//20]

        x = locs[:,1]-locs[0,1]
        y = locs[:,0]-locs[0,0]

        if TrialInfo['Correct'][itrl]:
            ax.plot(x,y, 'g', alpha = 1, markersize=1)
        elif TrialInfo['Wrong'][itrl]:
            ax.plot(x,y, 'r', alpha = 1, markersize=1)
        else:
            ax.plot(x,y, 'k', alpha = 0.4, markersize=1)

    ax.set_ylim([-50, 500])
    ax.set_xlim([-500, 500])
    fig.savefig(os.path.join(figure_folder, 'Trial_paths.png'))
    
    
def corr_wrong_barplot(TrialInfo, split, ax):
    corr_wrong_bar = []
    names = []
    uniq_splits = np.unique(split)
    width = 0.35
    for ibar, isplit in enumerate(uniq_splits):
        inc = split == isplit
        corr = np.sum(TrialInfo['Correct'][inc])
        trials = len(TrialInfo['Correct'][inc])
        wrong = np.sum(TrialInfo['Wrong'][inc])

        corr_wrong_bar.append(np.array([corr/trials, wrong/trials]))
        names.append(isplit)
    
        ax.bar(float(ibar)-width/2, corr/trials, width, color="green")
        ax.bar(float(ibar)+width/2, wrong/trials, width, color="red")
        
    ax.set_xticks(np.arange(len(uniq_splits)))
    ax.set_xticklabels(names)
    

def make_rob_plots(TrialInfo, figure_folder):
    # make plots

    triallen = []
    Av = []


    for i in TrialInfo['Location']:
        x = (len(i)/60)
        triallen = np.append(x, triallen)
        Av = np.mean(triallen)


    TrialLR = []
    TrialLR.append(TrialInfo['Right'])
    # 1 if right, 0 if left

    TrialCorrIncorr = []
    TrialCorrIncorr.append(TrialInfo['Correct'])
    #Correct is 1, anything else is 0





    TrialNonAttend = []
    x = []
    for i in TrialInfo['Event']:
        for z in i:
            if z == 3000:
                x = 1
            if ((z == 1) or (z == 2) or (z == 115) or (z == 117) or (z == 116) or (z == 104) or (z == 105) or (z == 107) or (z == 998) or (z == 3063) and (x == 1)):
                TrialNonAttend.append(z)
                x = 0
    if len(TrialNonAttend) != len(TrialInfo['Location']):
    TrialNonAttend.append(998) 

    x = 0
    transp = []
    for i in TrialInfo['Event']:
        for z in i:
            if 13000 <= z <= 13100 and (x%2) == 0:
                transp.append(z)
            x = x+1


    OutcomeLeft1 = []
    OutcomeRight1 = []
    OutcomeLeft = []
    OutcomeRight = []
    Outcome = []
    for key, decision in enumerate(TrialLR[0]):
        if decision == 1:
            OutcomeRight1.append(TrialCorrIncorr[0][key])
        else:
            OutcomeLeft1.append(TrialCorrIncorr[0][key])
    for i in OutcomeLeft1:
        if (i == 1):
            OutcomeLeft.append(100)
        else:
            OutcomeLeft.append(0)

    for i in OutcomeRight1:
        if (i == 1):
            OutcomeRight.append(100)
        else:
            OutcomeRight.append(0)

    for i in TrialCorrIncorr[0]:
        if (i == 1):
            Outcome.append(100)
        else:
            Outcome.append(0)

    AvarageLeft = []
    AvarageRight = []
    AvarageOutcome = []

    AvarageLeft = np.mean(OutcomeLeft)
    AvarageRight = np.mean(OutcomeRight)
    AvarageOutcome = np.mean(Outcome)

    ListLeft = []
    ListRight = []
    for key, decision in enumerate(TrialLR[0]):
        if decision == 1:
            ListRight.append(TrialInfo['Location'][key])
        else:
            ListLeft.append(TrialInfo['Location'][key])



    # for i in range (len(ListLeft)):
    #     ListLeft[i] = np.delete(ListLeft[i], -1, 1)

    # for i in range (len(ListRight)):
    #     ListRight[i] = np.delete(ListRight[i], -1, 1)

    ListLeft = np.array(ListLeft, dtype=object)
    ListRight = np.array(ListRight, dtype=object)

    PercListLeft = []
    for i,x in enumerate(ListLeft):
        z = x[-1,0]
        PercListLeft.append(x)
        PercListLeft[i] = np.true_divide(PercListLeft[i][0:-1,0], z)
        PercListLeft[i] = np.append(PercListLeft[i], [1])
        PercListLeft[i] = np.insert(x, 0, PercListLeft[i], axis=1)
        

    PercListRight = []
    for i,x in enumerate(ListRight):
        c = x[-1,0]
        PercListRight.append(x)
        PercListRight[i] = np.true_divide(PercListRight[i][0:-1,0], c)
        PercListRight[i] = np.append(PercListRight[i], [1])
        PercListRight[i] = np.insert(x, 0, PercListRight[i], axis=1)


    n = 100
    PercChunkL = [[0,0,0] for i in range(n)]
    PercChunkR = [[0,0,0] for i in range(n)]

    for i in PercListLeft:
        x = 0
        LB = 0.00
        UB = 0.01 
        for z in range(100):
            PercChunkL[z] = np.vstack((PercChunkL[z], i[np.logical_and(i[:,0] > LB, i[:,0] < UB)]))
            x = (x+1)
            LB = (LB+0.01)
            UB = (UB+0.01)

    for i in PercListRight:
        x = 0
        LB = 0.00
        UB = 0.01 
        for z in range(100):
            PercChunkR[z] = np.vstack((PercChunkR[z], i[np.logical_and(i[:,0] > LB, i[:,0] < UB)]))
            x = (x+1)
            LB = (LB+0.01)
            UB = (UB+0.01)

    n = 100
    AvYL = [[0] for i in range(n)]
    StdYL = [[0] for i in range(n)]
    for i in range(100):
        AvYL[i] = np.mean(PercChunkL[i][:,2])
        StdYL[i] = np.std(PercChunkL[i][:,2])
    n = 100
    AvYR = [[0] for i in range(n)]
    StdYR = [[0] for i in range(n)]
    for i in range(100):
        AvYR[i] = np.mean(PercChunkR[i][:,2])
        StdYR[i] = np.std(PercChunkR[i][:,2])

    AvYL = np.array(AvYL)
    StdYL = np.array(StdYL)
    AvYR = np.array(AvYR)
    StdYR = np.array(StdYR)




    LRDisp = 150
    LRDisp = (LRDisp/100)

    OptDispPos = []

    for i in range(100):
        OptDispPos.append((i*LRDisp))

    OptDispNeg = np.negative(OptDispPos)


    LocAbs = np.absolute(TrialInfo['Location'])
    CorrAvg = []
    IncorrAvg = []
    MissAvg = []

    for key, i in enumerate(LocAbs):
        if TrialNonAttend[key] == 1:
            CorrAvg.append(i)
        elif TrialNonAttend[key] == 2:
            IncorrAvg.append(i)
        else:
            MissAvg.append(i)

    CorrAvg = np.array(CorrAvg, dtype=object)
    IncorrAvg = np.array(IncorrAvg, dtype=object)
    MissAvg = np.array(MissAvg, dtype=object)

    PercListCorr = []
    for i,x in enumerate(CorrAvg):
        z = x[-1,0]
        PercListCorr.append(x)
        PercListCorr[i] = np.true_divide(PercListCorr[i][0:-1,0], z)
        PercListCorr[i] = np.append(PercListCorr[i], [1])
        PercListCorr[i] = np.insert(x, 0, PercListCorr[i], axis=1)
        

    PercListIncorr = []
    for i,x in enumerate(IncorrAvg):
        c = x[-1,0]
        PercListIncorr.append(x)
        PercListIncorr[i] = np.true_divide(PercListIncorr[i][0:-1,0], c)
        PercListIncorr[i] = np.append(PercListIncorr[i], [1])
        PercListIncorr[i] = np.insert(x, 0, PercListIncorr[i], axis=1)

    PercListMiss = []
    for i,x in enumerate(MissAvg):
        c = x[-1,0]
        PercListMiss.append(x)
        PercListMiss[i] = np.true_divide(PercListMiss[i][0:-1,0], c)
        PercListMiss[i] = np.append(PercListMiss[i], [1])
        PercListMiss[i] = np.insert(x, 0, PercListMiss[i], axis=1)


    n = 100
    PercChunkC = [[0,0,0] for i in range(n)]
    PercChunkI = [[0,0,0] for i in range(n)]
    PercChunkM = [[0,0,0] for i in range(n)]

    for i in PercListCorr:
        x = 0
        LB = 0.00
        UB = 0.01 
        for z in range(100):
            PercChunkC[z] = np.vstack((PercChunkC[z], i[np.logical_and(i[:,0] > LB, i[:,0] < UB)]))
            x = (x+1)
            LB = (LB+0.01)
            UB = (UB+0.01)

    for i in PercListIncorr:
        x = 0
        LB = 0.00
        UB = 0.01 
        for z in range(100):
            PercChunkI[z] = np.vstack((PercChunkI[z], i[np.logical_and(i[:,0] > LB, i[:,0] < UB)]))
            x = (x+1)
            LB = (LB+0.01)
            UB = (UB+0.01)

    for i in PercListMiss:
        x = 0
        LB = 0.00
        UB = 0.01 
        for z in range(100):
            PercChunkM[z] = np.vstack((PercChunkM[z], i[np.logical_and(i[:,0] > LB, i[:,0] < UB)]))
            x = (x+1)
            LB = (LB+0.01)
            UB = (UB+0.01)

    n = 100
    AvYC = [[0] for i in range(n)]
    StdYC = [[0] for i in range(n)]
    for i in range(100):
        AvYC[i] = np.mean(PercChunkC[i][:,2])
        StdYC[i] = np.std(PercChunkC[i][:,2])
    n = 100
    AvYI = [[0] for i in range(n)]
    StdYI = [[0] for i in range(n)]
    for i in range(100):
        AvYI[i] = np.mean(PercChunkI[i][:,2])
        StdYI[i] = np.std(PercChunkI[i][:,2])

    if(len(PercListMiss) != 0):
        n = 100
        AvYM = [[0] for i in range(n)]
        StdYM = [[0] for i in range(n)]
        for i in range(100):
            AvYM[i] = np.mean(PercChunkM[i][:,2])
            StdYM[i] = np.std(PercChunkM[i][:,2])
        

    AvYC = np.array(AvYC)
    StdYC = np.array(StdYC)
    AvYI = np.array(AvYI)
    StdYI = np.array(StdYI)
    AvYM = np.array(AvYM)
    StdY = np.array(StdYM)

    corr = []
    incorr = []
    miss = []

    for i in TrialNonAttend:
        if i == 1:
            corr.append(100)
            incorr.append(0)
            miss.append(0)
        if i == 2:
            incorr.append(100)
            corr.append(0)
            miss.append(0)
        else:
            miss.append(100)
            corr.append(0)
            incorr.append(0)


    window_size = 10
    i = 0
    moving_averages_corr = []
    while i < len(corr) - window_size + 1:
        this_window = corr[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages_corr.append(window_average)
        i += 1

    i = 0
    moving_averages_incorr = []
    while i < len(incorr) - window_size + 1:
        this_window = incorr[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages_incorr.append(window_average)
        i += 1

    i = 0
    moving_averages_miss = []
    while i < len(miss) - window_size + 1:
        this_window = miss[i : i + window_size]

        window_average = sum(this_window) / window_size
        moving_averages_miss.append(window_average)
        i += 1


    import matplotlib.pyplot as plt
    x = np.arange(len(AvYL))
    fig, axs = plt.subplots(2,3, figsize=[10, 10])
    axs = axs.ravel()
    
    axs[0].plot(x, AvYL, 'b-', label='Average Left')
    axs[0].fill_between(x, AvYL - StdYL, AvYL + StdYL, color='b', alpha=0.2)
    axs[0].plot(x, AvYR, 'r-', label='Average Right')
    axs[0].fill_between(x, AvYR - StdYR, AvYR + StdYR, color='r', alpha=0.2)
    axs[0].plot(x, OptDispPos, 'g:', label='Optimal Path')
    axs[0].plot(x, OptDispNeg, 'g:')
    axs[0].legend()
    axs[0].set_title('Average Paths', y = 1, fontsize=16, weight='bold')


    Av1 = []

    x = np.arange(len(triallen))
    for i in range(len(triallen)):
        Av1.append(Av)

    axs[1].plot(x, triallen, 'g-', label='Trial Length')
    axs[1].plot(x, Av1, 'r:', label='Average')
    axs[1].set_title('Trial Lengths', y = 1, fontsize=16, weight='bold')
    axs[1].legend()
    
    for key, i in enumerate(TrialInfo['Location']):
        if TrialNonAttend[key] == 1:
            axs[2].plot(i[:,0],i[:,1],'g-' )
        elif TrialNonAttend[key] == 2:
            axs[2].plot(i[:,0],i[:,1],'r-' )
        else:
            axs[2].plot(i[:,0],i[:,1],'#ffa600' )
    axs[2].set_title('All Trial Paths', y = 1, fontsize=16, weight='bold')



    axs[3].plot(AvYC,'g-' )
    axs[3].plot(AvYI,'r-' )
    axs[3].plot(AvYM,'#ffa600' )
    axs[3].set_title('Average Trial Paths- Correct, incorrect and Miss', y = 1, fontsize=16, weight='bold')




    axs[4].plot(moving_averages_corr, '-g', label='Hit')
    axs[4].plot(moving_averages_incorr, '-r', label='Wrong')
    axs[4].plot(moving_averages_miss, '#ffa600', label='Miss')
    axs[4].set_title('Session Performance', y = 1, fontsize=16, weight='bold')
    axs[4].legend()
 
    fig.savefig(os.path.join(figure_folder, 'RobPlots.svg'), transparent = True)

    print('Average Performance on the Right-sided stimulus:',(AvarageRight))
    print('Average Performance on the Left-sided stimulus:',(AvarageLeft))
    print('Average Performance:',(AvarageOutcome))
    print('Number of trials:',(len(TrialInfo['Location'])))
    print('Length of session (min):',((TrialInfo['EventTime'][-1][-1])/60))
    print('Hello')





if __name__ == "__main__":
    if len(sys.argv) == 1:
        end_of_day()
    else:
        logfile = sys.argv[1]
        end_of_day(logfile)
