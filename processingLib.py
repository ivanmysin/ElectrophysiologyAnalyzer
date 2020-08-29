# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 14:10:11 2014
library for data proceccing 
@author: ivan
"""
import numpy as np
from scipy.signal import argrelmax
import scipy.stats as stat
import scipy.signal as sig 

import matplotlib.pyplot as plt



#####################################################################
def circular_distribution(amples, angles, angle_step, nkernel=15, density=True):
    from scipy.ndimage.filters import convolve1d
    from scipy.signal.windows import parzen

    kernel = parzen(nkernel)
    bins = np.arange(-np.pi, np.pi+angle_step, angle_step)
    distr, _ = np.histogram(angles, bins=bins, weights=amples, density=density)


    distr = convolve1d(distr, kernel, mode="wrap")
    bins = np.convolve(bins, [0.5, 0.5], mode="valid")

    return bins, distr
    
def get_angles_in_range(angles):
    two_pi = 2*np.pi
    anles_in_range = angles%(two_pi)
    anles_in_range[anles_in_range < -np.pi] += two_pi
    anles_in_range[anles_in_range >= np.pi] -= two_pi
    
    return anles_in_range



def cossfrequency_phase_amp_coupling(phase_signal, fd, hi_freq_W, phasebins=20):

    phase_signal = np.angle( sig.hilbert(phase_signal), deg=False ) 

    coefAmp = np.abs( hi_freq_W )
    coefAmp = stat.zscore(coefAmp, axis=1)
    
    coupling = np.empty( shape=(phasebins, coefAmp.shape[0]), dtype=np.float)
    
    for freq_idx in range(hi_freq_W.shape[0]):
        coup, _ = np.histogram(phase_signal, bins=phasebins, weights=coefAmp[freq_idx, :], range=[-np.pi, np.pi])
        coupling[:, freq_idx]  = coup
        
    return coupling.T
    
def cossfrequency_phase_phase_coupling (signal1, signal2, nmarray, thresh_std=None, circ_distr=False):

    coupling = np.zeros(nmarray.shape[1])
    hilbert1 = sig.hilbert(signal1)
    hilbert2 = sig.hilbert(signal2)

    if not thresh_std is None:

        abs_signal = np.abs(hilbert1 * hilbert2 ) 

        signif_poits = abs_signal > (np.mean(abs_signal) + 1.0*np.std(abs_signal))

        hilbert1 = hilbert1[signif_poits]
        hilbert2 = hilbert2[signif_poits]
        
        if hilbert1.size == 0:
            return coupling
        
            

    angles1 = np.angle( hilbert1, deg=False )
    angles2 = np.angle( hilbert2, deg=False )
    

    distrs = []
    bins = []
    for i in range(nmarray.shape[1]):
        
        vects_angle = angles1 * nmarray[0, i] - angles2 * nmarray[1, i]
        x_vect = np.cos( vects_angle )
        y_vect = np.sin( vects_angle )
        mean_resultant_length = np.sqrt( np.sum(x_vect)**2 + np.sum(y_vect)**2 ) / vects_angle.size
        coupling[i] = mean_resultant_length
        
        if circ_distr:
            vects_angle = get_angles_in_range(vects_angle)
            bins_anles, distr = circular_distribution(np.ones_like(vects_angle), vects_angle, angle_step=0.1, nkernel=15)
            
            distrs.append(distr)
            bins.append(bins_anles)


    return coupling, bins, distrs

def get_2d_phase_hist(signal1, signal2, nbins=30):
    hilbert1 = sig.hilbert(signal1)
    hilbert2 = sig.hilbert(signal2)

    angles1 = np.angle(hilbert1)
    angles2 = np.angle(hilbert2)
    
    weights = np.abs(hilbert1 * hilbert2)
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(angles1, angles2)
    plt.show()
    """
    
    
    
    ranges = [[-np.pi, np.pi], [-np.pi, np.pi]]
    hist, xedges, yedges = np.histogram2d(angles2, angles1, bins=nbins, range=ranges, weights=weights, density=True)
    
    
    xedges = np.convolve(xedges, [0.5, 0.5], mode="valid")
    yedges = np.convolve(yedges, [0.5, 0.5], mode="valid")
    return hist, xedges, yedges

def get_modulation_index(W4phase, W4ampls, nbins=20):
    
    Nampls = W4ampls.shape[0]
    Nphs = W4phase.shape[0]
    
    modulation_index = np.zeros( [Nampls, Nphs], dtype=np.float )
    
    unif = 1.0 / (2*np.pi)
    
    ampls = np.abs(W4ampls)
    phases = np.angle(W4phase)
    for idx4ampl in range(Nampls):
        a = ampls[idx4ampl, :]
        for idx4phase in range(Nphs):
            p = phases[idx4phase, :]
            
            
            distr, _ = np.histogram(p, bins=nbins, weights=a, range=[-np.pi, np.pi], density=True)
            mi = np.sum(distr * np.log(distr / unif) )
            # distr = distr / np.sum(distr)
            # mi = -np.mean(distr * np.log(distr) )
            ##########
            # x = a * np.cos(p)
            # y = a * np.sin(p)
            # mi = np.sqrt(np.sum(x)**2 + np.sum(y)**2) / np.sum(a)
            
            modulation_index[idx4ampl, idx4phase ] = mi
    
    return modulation_index

def get_mi_by_coherence(phase_band, ampl_w, fd, ph_fr_range=[4, 12], nperseg=4096):
  
    mi = []

    for fr_idx in range(ampl_w.shape[0]):
        
        amples = np.abs(ampl_w[fr_idx, :])
        
        f, Coh = sig.coherence(phase_band, amples, fs=fd, nperseg=nperseg)
        Coh = Coh[ (f>ph_fr_range[0])&(f<ph_fr_range[1]) ]
        
        mi.append(Coh)
    
    f = f[ (f>ph_fr_range[0])&(f<ph_fr_range[1]) ]
    mi = np.vstack(mi)
    return mi, f
    
########################################################################
def get_units_phase_coupling(lfp, firing, fd):
    angles = np.array([])
    length = np.array([])
    analitic_signal = sig.hilbert(lfp)
    amp = np.abs(analitic_signal)
    amp /= np.linalg.norm(amp) 
    lfp_phases = np.angle(analitic_signal, deg=False )
    
    """       
    if (firing[1, :].size == 0):
        return 0, 0
    """    
    start = np.min(firing[1, :]).astype(int)
    end = np.max(firing[1, :]).astype(int) + 1
    for idx in range(start, end):
        spike_times = firing[0, firing[1, :]  == idx]
        if (spike_times.size == 0):
            continue
        spike_ind = np.floor(spike_times*fd).astype(int)
        spike_phases = lfp_phases[spike_ind]
        tmp_x = np.sum( amp[spike_ind] * np.cos(spike_phases) )
        tmp_y = np.sum( amp[spike_ind] * np.sin(spike_phases) )
        mean_phase = np.arctan2(tmp_y, tmp_x)
        length_vect = np.sqrt(tmp_x**2 + tmp_y**2) / spike_phases.size
        angles = np.append(angles, mean_phase)
        length = np.append(length, length_vect)
        
    return angles, length


    
    
    
def get_units_disrtibution(lfp, fd, firing, firing_slices):
    analitic_signal = sig.hilbert(lfp)
    lfp_phases = np.angle(analitic_signal, deg=False )
    neurons_phases = {}
    for key, sl in firing_slices.items():
        fir = firing[:, sl]
        neurons_phases[key] = np.array([], dtype=float)
        if (fir.size == 0):
            continue
        indexes = (fir[0, :] * fd).astype(int)
        neurons_phases[key] = lfp_phases[indexes]
    
    return neurons_phases


def get_phase_disrtibution(train, lfp, fs):
    if train.size == 0:
        return np.empty(0, dtype=np.float), np.empty(0, dtype=np.float)
    
    
    
    analitic_signal = sig.hilbert(lfp)
    lfp_phases = np.angle(analitic_signal, deg=False)
    lfp_ampls = np.abs(analitic_signal)
    
    train = np.round(train * fs * 0.001).astype(np.int) # multiply on 0.001 because train in ms, fs in Hz 

    train_phases = lfp_phases[train]
    train_ampls = lfp_ampls[train]

    count, bins = np.histogram(train_phases, bins=10, density=True, range=[-np.pi, np.pi], weights=train_ampls)
    bins = bins[:-1]
    return bins, count
