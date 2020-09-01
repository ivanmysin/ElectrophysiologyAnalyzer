import numpy as np
import matplotlib.pyplot as plt
import h5py
import lib
import os

"""
t = np.linspace(0, 10, 20000)
phases = 2 * np.pi * 6 * t + np.random.vonmises(0, 8.0, t.size)

phases = phases%(2 * np.pi)

decres_phases = (phases > 0.5) & (phases < np.pi)

dg_lfp = np.cos(phases)
dg_lfp[decres_phases] += 0.7 * np.cos( 2 * np.pi * 35 * t[decres_phases]  )

ms_lfp = dg_lfp + np.random.randn(dg_lfp.size)
"""

sourse_path = "/home/ivan/Data/2DG_DATA/txt/"
target_path = "/home/ivan/Data/2DG_DATA/hdf5/"


areas = {
    "1" : "MS", 
    "2" : "stim", 
    "4" : "DG", 
    "7" : "photo",     
}



for file in os.listdir(sourse_path):
    filename, ext = os.path.splitext(file)
    if ext != ".txt": continue

    print(file)
    elecrtodes = lib.read_datapack_txt_file(sourse_path + file)
    
    print(elecrtodes.keys())


    with h5py.File(target_path+filename+".hdf5", 'w') as h5file:
        
        extracellular_group = h5file.create_group("extracellular")
        
        for el_idx, (el_number, lfp) in enumerate(elecrtodes.items()):
            ele_group = extracellular_group.create_group('electrode_' + str(el_idx+1) )
            
            ele_group.attrs["area"] = areas[el_number]
            
            lfp_group = ele_group.create_group('lfp')
                    
            lfp_group_origin = lfp_group.create_group('origin_data')
            lfp_group_origin.attrs['SamplingRate'] = 1000
            lfp_group_origin.create_dataset("channel_1", data = lfp)



#plt.plot(t, dg_lfp)
# plt.show()
