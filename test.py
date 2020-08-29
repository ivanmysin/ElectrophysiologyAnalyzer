import numpy as np
import matplotlib.pyplot as plt
from elephant import signal_processing as sigp 
from scipy.stats import vonmises, zscore
from scipy.signal import hilbert, coherence
import 


fd = 1000
dur = 25
N = int(dur*fd)
t = np.linspace(0, dur,  N)


phases = 2 * np.pi * 7 * t 

# phases = phases%(2 * np.pi)
# freqs = np.fft.fftfreq(t.size, 1/fd)
# lfp_sp = np.random.rand(t.size) + 1j*np.random.rand(t.size)
# lfp_sp[lfp_sp == 0] = 0 + 1j * 0
# lfp_sp[freqs < 0].real = lfp_sp[freqs > 0].real
# lfp_sp[freqs < 0].imag = -lfp_sp[freqs > 0].imag

# lfp = np.fft.ifft(lfp_sp).real
# lfp[lfp > 0.1] = 0
# lfp *= 50

#decres_phases = (phases > 0.5) & (phases < np.pi)

lfp = np.cos(phases) + np.random.normal(0, 0.5, t.size)
lfp += 0.7 * np.cos( 2 * np.pi * 45 * t  ) * vonmises.pdf(phases, 3.0, loc=1.5)


plt.plot(t, lfp)
plt.show()

theta = sigp.butter(lfp, highpass_freq=4, lowpass_freq=20, order=3, fs=fd )


gamma_freqs = np.arange(25, 90, 1)

W = sigp.wavelet_transform(lfp, gamma_freqs, nco=6, fs=fd)

# W = W[:, (t>0.4)&(t<1.6)]
# t = t[(t>0.4)&(t<1.6)]

mi = []



for fr_idx in range(gamma_freqs.size):
    
    gamma_amples = np.abs(W[fr_idx, :])
    
    f, Coh = coherence(theta, gamma_amples, fs=fd, nperseg=4096)
    Coh = Coh[ (f>1)&(f<20) ]
    
    mi.append(Coh)

print(f)
theta_freqs = f[ (f>1)&(f<20) ]

mi = np.vstack(mi)



plt.pcolor(theta_freqs, gamma_freqs, mi, cmap="rainbow" )
plt.colorbar()
plt.show()



"""
import numpy as np
import scipy.signal as sig
import scipy.stats as stat
import os
import lib
import pandas as pd
import matplotlib.pyplot as plt

import pickle


frequency_rhythms = {
    "theta" : [4, 12],
    "slow gamma" : [30, 50],
    "fast gamma" : [50, 90],
}

controlVSexp = {
    1 : "2-DG",
    3 : "ACSF",
    4 : "2-DG",
    5 : "2-DG",
    6 : "ACSF",
    7 : "ACSF",
    8 : "2-DG",
    9 : "2-DG",

}


f = open("zero2base_ratio.pkl","rb")
zero2base_ratio = pickle.load(f)
f.close()

fig, axes = plt.subplots(ncols=1, nrows=3)

axes[0].set_title("Theta")
axes[1].set_title("Slow gamma")
axes[2].set_title("Fast gamma")

for rat_number, val in sorted( zero2base_ratio.items() ):

    print(rat_number)
    dates = []
    ratios_theta = []
    ratios_slow_gamma = []
    ratios_fast_gamma = []
    for date in sorted( val.keys() ):
        dates.append(date)
        ratios_theta.append(val[date]["theta"])
        ratios_slow_gamma.append(val[date]["slow gamma"])
        ratios_fast_gamma.append(val[date]["fast gamma"])


        # print(rhythms["slow gamma"])

    group = controlVSexp[rat_number]
    if group == "ACSF":
        color = "g"
    elif  group == "2-DG":
        color = "r"

    axes[0].plot(ratios_theta, label = rat_number, color=color)
    axes[1].plot(ratios_slow_gamma, label = rat_number, color=color)
    axes[2].plot(ratios_fast_gamma, label = rat_number, color=color)

for ax in axes:
    ax.legend()
plt.show()
"""
