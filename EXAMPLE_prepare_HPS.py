import numpy as np
import obspy, os, gc, re
import matplotlib
# matplotlib.use('Agg')

from obspy import read, read_inventory

from obspy.imaging.spectrogram import _nearest_pow_2
from scipy import signal
from matplotlib import mlab, transforms

from func import hps_algo, pitch_detection

## set paths

path_project = '/home/essingd/AWI/knipovich/'
path_waveforms = path_project+'data/'
path_waveforms_test =path_waveforms+'test_folder/'
inv_path = path_project+'CAL/'
    
## set experiment     
# experiment='KNIPA_NEW'
# network = '1B'
# station = 'KNR18'
# component = 'BHZ'


## set preprocessing parameter
pre_filt=(.05,.009,20,24)       # filtercorners for filter during response remove
bandpass_min = .1               # filtercorners for bandpass filter
bandpass_max = 10               # filtercorners for bandpass filter
decimate_factor = 2             # factor to downsample waveform data

## set parameter for spectrum
add_time = 1                    # hours to add before/after day (day to calculate spectrogram) in [hours]; this prevents amplitude drop at beginnig/end of spectrogram
nfft = 2000                     # number of datapoints used in each block for FFT, might be faster with _nearest_pow_2 but due to wanted 80s of win_len in specgram I set it to 2000

## set fundamental frequency detection parameter
nr_downsamp_HPS = 4             # Number to downsample the spectrum for fundamental frequency detection (pitch detection) with HarmonicProductSpectrum-Alogrithm
threshold_HSI = 2               # threshold for ratio of HarmonicStrengthIndex (ratio between amplitude of fundamental_frequency_value and maximum  amplitdue of Interharmonic)
                                # Automated detection and characterization of harmonic tremor in continuous seismic data; Roman D; 2017

trace = read(path_waveforms_test+'1B.KNR08..BHZ.D.2016.325')
trace += read(path_waveforms_test+'1B.KNR08..BHZ.D.2016.326')
trace += read(path_waveforms_test+'1B.KNR08..BHZ.D.2016.327')
trace.merge(method=1) # merges together three days
inv = read_inventory(inv_path+'RESP.1B.KNR08..BHZ')   
print(trace)
## calculate values to trim for 26 hours of data
print(trace)
trace.decimate(factor=decimate_factor) 
trace.remove_response(inventory=inv, pre_filt=pre_filt, output="ACC",water_level=60, plot=False)
trace.filter('bandpass',freqmin=bandpass_min,freqmax=bandpass_max,zerophase=True)
trace.detrend(type='simple') # remove mean
trace.detrend(type='linear') # remove trend
# trace.trim(starttime=trim_start,endtime=trim_end,nearest_sample=True,pad=True, fill_value=0)


    
ff_value_signal, ff_amplitude_signal, ff_value_NOsignal, ff_amplitude_NOsignal, specgram, specgram_time_vec ,specgram_freq_vec,trace_data_one_day,trace_data_one_day_time_line =pitch_detection(trace,nfft,threshold_HSI,nr_downsamp_HPS)


## save arrays to csv

np.savetxt("ff_values_sig.csv", ff_value_signal, delimiter=";") # save the detected fundamental frequency values classified as signal
np.savetxt("ff_values_nosig.csv", ff_value_NOsignal, delimiter=";") # save the detected fundamental frequency values classified as no signal
np.savetxt("spectrogram.csv", specgram, delimiter=";") # save the spectrogram



## for plotting 

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

vc_time_domain = np.linspace(0,24,np.shape(trace_data_one_day_time_line)[0])
vc_time_spec = np.linspace(0,24,np.shape(specgram)[1])

gridsize = (1, 1)

fig = plt.figure(figsize=(18, 9))
fig.subplots_adjust(hspace=0, wspace=0)
plt.rcParams.update({'font.size': 18})
plt.set_cmap('inferno')
## v_component

ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=7, rowspan=2)


im = ax1.pcolormesh(vc_time_spec,specgram_freq_vec,10*np.log10(specgram),vmin=-40,vmax=-340)
ax1.plot(vc_time_spec,ff_value_signal,'.',markersize= 3.0, color='navy',label='harmonic tremor')
ax1.plot(vc_time_spec,ff_value_NOsignal,'.',markersize= 3.0, color='forestgreen', label='rejected detection')
ax1.set_xlabel('Time [h]')
ax1.set_xticks([3,6,9,12,15,18,21]) 
ax1.set_ylabel('Frequency [Hz]')
ax1.legend(facecolor='white',markerscale=3)



axins = inset_axes(ax1,
                   width="5%",  # width = 5% of parent_bbox width
                   height="100%",  # height : 50%
                   loc='lower left',
                   bbox_to_anchor=(1.05, 0., 1, 1),
                   bbox_transform=ax1.transAxes,
                   borderpad=0,
                   )
fig.colorbar(im, cax=axins,label='Amplitude [dB]')

plt.savefig(path_project+'test.jpeg', dpi=100, facecolor='w', edgecolor='w',
        orientation='portrait', format=None,
        transparent=False, bbox_inches='tight',pad_inches=0.04)
plt.clf()



