import os
from datetime import datetime
import lib
import numpy as np
import scipy.signal as sig
import scipy.stats as stat
import matplotlib.pyplot as plt
import pickle

main_path = "/media/ivan/Seagate Backup Plus Drive/Data/2DG in txt/"

path4diapasons = "/media/ivan/Seagate Backup Plus Drive/Data/2DG in txt/diapasons_of_processing/"
path4results = "/media/ivan/Seagate Backup Plus Drive/Data/2DG in txt/results_scaled/"
rats = []

frequency_rhythms = {
    "theta" : [4, 12],
    "slow gamma" : [30, 50],
    "fast gamma" : [50, 90],
}


for file in os.listdir(main_path):
    filename, ext = os.path.splitext(file)
    if ext != ".txt": continue
    rat_number, date = filename.split("_")
    date = datetime.strptime(date, '%d.%m.%y')
    rat_number = int(rat_number[-1])
    rats.append(
        {
        "rat_number" : rat_number,
        "date" : date,
        "file" : file,
        }
    )

rats = sorted(rats, key=lambda val: (val["rat_number"], val["date"]) )
fs = 1000

zero2base_ratio = {}

for idx, rat in enumerate(rats):
    if (idx < 0):
        continue

    print (idx, rat["file"])


    data = lib.read_datapack_txt_file(main_path + rat["file"])
    lfp = data["4"]
    thresh = 5 * np.std(lfp)
    indexes = np.argwhere(lfp > thresh).ravel()

    if indexes.size > 2:

        intervals = np.diff(indexes) / fs
        max_interv_idx = np.argmax(intervals)

        min_idx_lfp = int( indexes[max_interv_idx] + 10*fs)

        if max_interv_idx == intervals.size-1:
            max_idx_lfp = intervals.size-1
        else:
            max_idx_lfp = int( indexes[max_interv_idx+1] - 10*fs)
    else:
        min_idx_lfp = 0
        max_idx_lfp = lfp.size



    # ymin = np.min(lfp)
    # ymax = np.max(lfp)
    # fig, ax = plt.subplots()
    # ax.plot(lfp)
    # ax.axvline(min_idx_lfp, ymin=ymin, ymax=ymax, color="red", linewidth=5)
    # ax.axvline(max_idx_lfp, ymin=ymin, ymax=ymax, color="red", linewidth=5)
    # fig.savefig(path4diapasons+rat["file"][:-4] + ".png")
    #  plt.close(fig=fig)
    # plt.show()


    lfp = lfp[min_idx_lfp:max_idx_lfp]

    fig, axx = plt.subplots(ncols=1, nrows=len(frequency_rhythms.keys()), constrained_layout=True)
    fig.suptitle(rat["file"])


    ripples_episodes = lib.get_ripples_episodes_indexes(lfp, fs)
    if ripples_episodes.size == 0:
        continue


    if not rat["rat_number"] in zero2base_ratio.keys():
        zero2base_ratio[rat["rat_number"]] = {}

    zero2base_ratio[rat["rat_number"]][rat["date"]] = {}

    # for rhythm_name in frequency_rhythms.keys():
    #     zero2base_ratio[rat["rat_number"]][rat["date"]][rhythm_name] = []

    # print(zero2base_ratio)


    for axx_idx, (rhythm_name, freqs_diaps) in enumerate( sorted( frequency_rhythms.items() ) ):

        rhythm_rtigered = lib.get_abs_rhythm_trigered(ripples_episodes, lfp, fs,  lowcut=freqs_diaps[0], highcut=freqs_diaps[1])
        # print(rhythm_rtigered.shape)
        t = np.linspace(-0.5 * rhythm_rtigered.shape[1] / fs, 0.5 * rhythm_rtigered.shape[1] / fs, rhythm_rtigered.shape[1])

        rhythm_rtigered_mean = np.median(rhythm_rtigered, axis=0)


        ratio = rhythm_rtigered_mean[int(rhythm_rtigered_mean.size*0.5)] / np.median(rhythm_rtigered_mean)


        zero2base_ratio[rat["rat_number"]][rat["date"]][rhythm_name] = ratio
        # rhythm_rtigered_std = np.std(rhythm_rtigered, axis=0)
        # rhythm_rtigered_mean_zscore = stat.zscore(rhythm_rtigered, axis=0)

        axx[axx_idx].plot(t, rhythm_rtigered_mean ) # , yerr=rhythm_rtigered_std
        axx[axx_idx].set_title(rhythm_name)

        axx[axx_idx].set_xlim(t[0], t[-1])
        # axx[axx_idx].set_ylim(-6.5, 6.5)

    fig.savefig(path4results + rat["file"][:-4] + ".png")
    plt.close(fig=fig)

f = open("zero2base_ratio.pkl","wb")
pickle.dump(zero2base_ratio, f)
f.close()


# fig, axx = plt.subplots(ncols=1, nrows=len(frequency_rhythms.keys()), constrained_layout=True)
#
# for idx, (rat, rhythm) in enumerate(zero2base_ratio.items()):
#
#     axx[idx].plot( zero2base_ratio[rat][rhythm], label=rat )
#
#
# plt.show()


