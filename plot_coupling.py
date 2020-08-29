import numpy as np
import h5py
import processingLib as plib
import matplotlib.pyplot as plt
filepath = "./data/test.hdf5"


with h5py.File(filepath, 'r') as h5file:
    
    for elecr in h5file["extracellular"].keys():
        electrod_group = h5file["extracellular"][elecr]
        print(electrod_group.attrs["area"])
        
        fd = electrod_group["lfp/origin_data"].attrs["SamplingRate"]
        # theta_band = electrod_group["lfp/processing/channel_1_bands/theta"][:]
        
        # slow_gamma_band = electrod_group["lfp/processing/channel_1_bands/slow gamma"][:]
        # fast_gamma_band = electrod_group["lfp/processing/channel_1_bands/fast gamma"][:]
        
        
        
        """
        freqs = electrod_group["lfp/processing/channel_1_wavelet/frequencies"][:]
        
        gamma_freqs_0 = np.argmin( freqs>=30 )
        gamma_freqs_1 = np.argmin( freqs<=90 )
        gamma_freqs = freqs[gamma_freqs_0:gamma_freqs_1]
        
        hi_freq_W = electrod_group["lfp/processing/channel_1_wavelet/channel_1wavelet_coeff"][gamma_freqs_0:gamma_freqs_1, :]

       
        coupling = plib.cossfrequency_phase_amp_coupling(theta_band, fd, hi_freq_W, phasebins=50)
        phases = np.linspace(-np.pi, np.pi, coupling.shape[0])
        
        plt.pcolor(phases, gamma_freqs, coupling)
        """
        """
        nmarray = np.ones( [2, 9] )
        nmarray[0, :] = np.arange(1, 10)
        

        cross_thythms_coh, bins, distrs = plib.cossfrequency_phase_phase_coupling(theta_band, slow_gamma_band , nmarray, 1.0, circ_distr=True)
        # cross_thythms_coh2 = plib.cossfrequency_phase_phase_coupling(theta_band, fast_gamma_band , nmarray, 1.0)
        
        fig, axes = plt.subplots()
        plt.plot(nmarray[0, :], cross_thythms_coh, color="blue", label="slow gamma")
        # plt.plot(nmarray[0, :], cross_thythms_coh2, color="green", label="fast gamma")
        plt.legend()
        
        fig, axes = plt.subplots(ncols=len(bins), subplot_kw={'projection': "polar"} )
        
        for idx, (bins_engles, distr) in enumerate(zip(bins, distrs)):
            axes[idx].plot(bins_engles, distr)
        """
        
        """
        hist, xedges, yedges = plib.get_2d_phase_hist(theta_band, slow_gamma_band, nbins=50)
        plt.pcolor(xedges, yedges, hist)
        """
        
        """
        freqs = electrod_group["lfp/processing/channel_1_wavelet/frequencies"][:]
        
        gamma_freqs_0 = np.argmin( np.abs(freqs-25) )
        gamma_freqs_1 = np.argmin( np.abs(freqs-45) )
        gamma_freqs = freqs[gamma_freqs_0:gamma_freqs_1]
        
        theta_freqs_0 = np.argmin( np.abs(freqs-4) )
        theta_freqs_1 = np.argmin( np.abs(freqs-10) )
        theta_freqs = freqs[theta_freqs_0:theta_freqs_1]
        
        gamma_W = electrod_group["lfp/processing/channel_1_wavelet/channel_1wavelet_coeff"][gamma_freqs_0:gamma_freqs_1, :]
        theta_W = electrod_group["lfp/processing/channel_1_wavelet/channel_1wavelet_coeff"][theta_freqs_0:theta_freqs_1, :]

        MI = plib.get_modulation_index(theta_W, gamma_W, nbins=20)
        plt.pcolor(theta_freqs, gamma_freqs, MI, vmin=0, vmax=np.max(MI), cmap="rainbow")
        
        plt.colorbar()
        # plt.imshow(np.abs(theta_W[:, :1000]))
        """
        
        freqs = electrod_group["lfp/processing/channel_1_wavelet/frequencies"][:]
        
        gamma_freqs_0 = np.argmin( np.abs(freqs-25) )
        gamma_freqs_1 = np.argmin( np.abs(freqs-90) )
        gamma_freqs = freqs[gamma_freqs_0:gamma_freqs_1]
        
        
        gamma_W = electrod_group["lfp/processing/channel_1_wavelet/channel_1wavelet_coeff"][gamma_freqs_0:gamma_freqs_1, :]
        theta = electrod_group["lfp/processing/channel_1_bands/theta"]
        
        MI, theta_freqs = plib.get_mi_by_coherence(theta, gamma_W, fd, ph_fr_range=[4, 12], nperseg=4096)
        
        plt.pcolor(theta_freqs, gamma_freqs, MI, vmin=0, vmax=1.0, cmap="rainbow")
        plt.colorbar()
        plt.show()

