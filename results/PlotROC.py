import matplotlib.pyplot as plt
import numpy as np
import os
import math
from scipy.io import loadmat
from ast import literal_eval
import scienceplots

def OrderData(theta, tput_meas):
    Theta = sorted(set(theta))
    TPUT_meas = []
    for k in range(len(Theta)):
        indexes=[i for i, x in enumerate(theta) if x == Theta[k]]
        Tput_meas = sum(tput_meas[i] for i in indexes)
        TPUT_meas.append(Tput_meas)
    return Theta, TPUT_meas

if __name__ == "__main__":

    path = 'results/ROC/'
    pathOut = 'results/plots/'
    files_plots = [i for i in os.listdir(path) if i.endswith('.mat')]

    PD = []
    PFA = []
    Type = []
    AUCs = []
    labels_latex = []

    for fname in files_plots:
        file_path = os.path.join(path, fname)
        r = loadmat(file_path)
        pfa_range = r['pfa_range.mat'][0].tolist()
        PD_type = literal_eval(r['PD_type.mat'].tolist()[0])
        FAR_type = literal_eval(r['FAR_type.mat'].tolist()[0])

        if 'MSAR_' in fname:
            label = r'MS-AR($6,1,0$)'
        elif 'MSARk_' in fname:
            label = r'MS-AR($18,2,0$)'
        elif 'MSARkh_' in fname:
            label = r'MS-AR($18,2,1$)'
        elif 'GPalm2020' in fname:
            label = r'AR($n$)-GSP\textsuperscript{[15]}'
        elif 'CNN-GSP' in fname:
            label = r'CNN-GSP\textsuperscript{[17]}'
        else:
            label = 'Unknown'

        Type.append(label)
        PD.append(PD_type)
        PFA.append([max(f, 1e-4) for f in FAR_type])  # Avoid log(0)

    # --- Plot Setup --- #
    plt.rc('text', usetex=True)
    plt.style.use(['science', 'ieee', 'std-colors'])

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=600)

    color_list = ['royalblue', 'darkred', 'mediumseagreen', 'darkorange', 'gray']
    marker_list = ['o', 's', '^', 'D', '*']
    linestyle_list = ['-', '--', '-.', ':', '-']

    for i in range(len(PD)):
        pd_curve = np.array(PD[i])
        pfa_curve = np.clip(np.array(PFA[i]), 1e-4, 1)  # Ensure finite values

        # Sort and normalize
        idx = np.argsort(pfa_curve)
        pd_sorted = pd_curve[idx]
        far_sorted = pfa_curve[idx]
        far_norm = (far_sorted - far_sorted[0]) / (far_sorted[-1] - far_sorted[0])
        auc_score = np.trapz(pd_sorted, far_norm)

        AUCs.append(auc_score)
        labels_latex.append(r'%s (AUC=%.3f)' % (Type[i], auc_score))

        log_far = np.log10(far_sorted)
        ax.plot(log_far, pd_sorted,
                label=labels_latex[-1],
                color=color_list[i % len(color_list)],
                linestyle=linestyle_list[i % len(linestyle_list)],
                marker=marker_list[i % len(marker_list)],
                linewidth=2, markersize=5, alpha=0.9)

    # --- Axis Labels and Grid --- #
    ax.set_xlabel(r'$\log_{10}$ FAR', fontsize=16)
    ax.set_ylabel(r'$P_D$', fontsize=16)
    ax.set_xlim([-1.65, 0])
    ax.set_ylim([0.94, 1.005])
    ax.grid(True, which='both', linestyle='-', linewidth=0.4, alpha=0.4)
    ax.minorticks_on()
    ax.yaxis.set_tick_params(labelleft=False, labelright=True)

    # --- Legend Sorted by AUC --- #
    handles, labels = ax.get_legend_handles_labels()
    auc_sorted = sorted(zip(AUCs, handles, labels), reverse=True)
    auc_handles = [x[1] for x in auc_sorted]
    auc_labels = [x[2] for x in auc_sorted]

    ax.legend(auc_handles, auc_labels, loc='lower center', bbox_to_anchor=(0.5, 1.0),
              fancybox=True, shadow=True, ncol=2, fontsize=10, title="Methods")

    fig.subplots_adjust(bottom=0.2, top=0.85)
    os.makedirs(pathOut, exist_ok=True)
    fig.savefig(os.path.join(pathOut, 'ROC_MSAR_All.png'), dpi=600)
    plt.show()
