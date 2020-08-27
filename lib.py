
import numpy as np
# import sys
# sys.path.append("/home/ivan/coding/septo-hippocampal-model/cython_code/")
from processingLib import *
import scipy.signal as sig

def read_datapack_txt_file(filepath):
    data = []
    channel_names = []
    with open(filepath) as file:
        isstartdata = False
        for idx, line in enumerate(file.readlines()):
            line = line[:-1]  # удаляем \n
            if line.find("#") != -1: # читаем данные после строки с #
                for linedata in line.split(" "):
                    if linedata == "" or linedata == "#" or linedata == "Msec.":
                        continue
                    channel_names.append( linedata )
                    data.append([])
                isstartdata = True
                continue

            if isstartdata:
                idx_line = -3 # выбрасываем два первых столбца
                for linedata in line.split(" "):
                    try:
                        sample = float(linedata)
                        idx_line += 1
                        if idx_line >= 0:
                            data[idx_line].append(sample)

                    except ValueError:
                        continue

    datadir = {}
    for idx in range(len(channel_names)):
        datadir[channel_names[idx]] = np.asarray(data[idx])

    return datadir

def get_ripples_episodes_indexes(lfp, fs):
    ripples_lfp = butter_bandpass_filter(lfp, lowcut=90, highcut=200, fs=fs, order=3)

    ripples_lfp_th = 5 * np.std(ripples_lfp)
    ripples_lfp_relmax_idx = sig.argrelextrema(ripples_lfp, np.greater)[0]
    ripples_lfp_idx = ripples_lfp_relmax_idx[ripples_lfp[ripples_lfp_relmax_idx] > ripples_lfp_th]

    if ripples_lfp_idx.size == 0:
        return ripples_lfp_idx
    ripples_lfp_idx = ripples_lfp_idx[np.append(True, np.diff(ripples_lfp_idx) > 0.4 * fs)]


    return ripples_lfp_idx


def get_abs_rhythm_trigered(episodes, lfp, fs, lowcut, highcut):


    rhythm_rtigered = []
    for rp in episodes:
        min_inx = int(rp - 0.2 * fs)
        max_inx = int(rp + 0.2 * fs)
        if max_inx > lfp.size - 2 or min_inx < 0:
            continue
        # print(min_inx, max_inx, lfp.size)
        # print(lfp[min_inx:max_inx].shape)
        rhythm_lfp = butter_bandpass_filter(lfp[min_inx:max_inx], lowcut=lowcut, highcut=highcut, fs=fs, order=3)
        rhythm_abs = np.abs(sig.hilbert(rhythm_lfp))
        rhythm_rtigered.append(rhythm_abs)

    rhythm_rtigered = np.asarray(rhythm_rtigered)
    return rhythm_rtigered
