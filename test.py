import numpy as np

angles = np.linspace(-10, 10, 10)

norm_anles = np.arctan2(np.sin(angles), np.cos(angles))

my_norm_anles = angles%(2*np.pi)
# my_norm_anles[my_norm_anles < -np.pi] += 2*np.pi
# my_norm_anles[my_norm_anles >= np.pi] -= 2*np.pi

print(norm_anles)            
print(my_norm_anles)            
            
            
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
