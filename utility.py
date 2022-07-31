import numpy as np
import scipy.signal as sg
from scipy.fft import fft, fftfreq

from ssqueezepy import ssq_cwt, Wavelet, ssq_stft, cwt
from ssqueezepy.utils import cwt_scalebounds, make_scales, p2up
from ssqueezepy.wavelets import center_frequency, freq_resolution, time_resolution
from ssqueezepy.utils import logscale_transition_idx
from ssqueezepy import extract_ridges


def background_sub(data, method, tparams, back_parms):
    
    indx =np.logical_and(data[:,0]> float(tparams['tmin']), data[:,0]< float(tparams['tmax']))
    t = data[:,0][indx]
    x=  data[:,1][indx]
    dt =t[1] -t[0]

    if method == 'smooth':
        smth =(sg.savgol_filter(x,  int(back_parms['smthn']),  int(back_parms['degree']),delta = dt))
        return np.array([t, smth]).T, np.array([t, x-smth]).T
    elif method == 'polyfit':
        poly = np.polyfit(t, x,int(back_parms['degree']))
        fnc = np.poly1d(poly)
        return np.array([t, fnc(t)]).T, np.array([t, x-fnc(t)]).T
    elif method == 'bandpass':

        b, a = sg.butter(int(back_parms['order']), 
                            [float(back_parms['freq-min']), float(back_parms['freq-max'])],fs=1/dt, btype='band')
        band = sg.filtfilt(b, a, x)

        return np.array([t, x-band]).T, np.array([t, band]).T
    elif method == 'smooth+bandpass':
        smth =(sg.savgol_filter(x,  int(back_parms['smthn']),  int(back_parms['degree']),delta = dt))
        b, a = sg.butter(int(back_parms['order']), 
                            [float(back_parms['freq-min']), float(back_parms['freq-max'])],fs=1/dt, btype='band')
        smth_band = sg.filtfilt(b, a, x-smth)
        return np.array([t, x-smth_band]).T, np.array([t, smth_band]).T

    elif method == 'polyfit+bandpass':
        poly = np.polyfit(t, x,int(back_parms['degree']))
        b, a = sg.butter(int(back_parms['order']), 
                            [float(back_parms['freq-min']), float(back_parms['freq-max'])],fs=1/dt, btype='band')
        fnc = np.poly1d(poly)
        poly_band = sg.filtfilt(b, a, x-fnc(t))
        return np.array([t, x-poly_band]).T, np.array([t, poly_band]).T

def bandfilt_design(data,back_parms):
    dt =data[:,0][1] -data[:,0][0]
    b, a = sg.butter(int(back_parms['order']), 
                            [float(back_parms['freq-min']), float(back_parms['freq-max'])],fs=1/dt, btype='band')
    w, h = sg.freqz(b, a, fs=1/dt,worN=2000)
    indx =np.logical_and(w>float(back_parms['freq-min'])*0.5, w<float(back_parms['freq-max'])*1.5)
    return w[indx], h[indx]

def fft_fn(data, tparams, fparams):
    indx =np.logical_and(data[:,0]>= float(tparams['tmin']), data[:,0]<= float(tparams['tmax']))
    t = data[:,0][indx]
    x=  data[:,1][indx]
    dt =t[1] -t[0]

    nextpow2 = int(fparams['nextpow2'])
    alpha =  float(fparams['alpha'])
    fmin =  float(fparams['fmin'])
    fmax =  float(fparams['fmax'])
    window =sg.windows.tukey(len(t), alpha=alpha)

    
    nfft=int(2**(np.ceil(np.log2(len(t)))+nextpow2)) 
    resu_fft = 2.0/nfft * np.abs( fft(x*window, n=nfft )[0:nfft//2])
    freq = fftfreq(n=int(nfft), d=dt)[:nfft//2]
    findx =  np.logical_and(freq>=fmin, freq<=fmax)

    return freq[findx], resu_fft[findx]


def stft_fn(data, params, boundary_zeroes =True):

    fc =params['fc']
    ncycles = params['ncycles']
    overlap = params ['overlap']
    nextpow2 = params ['nextpow2']
    fmin =  params ['fmin']
    fmax =  params ['fmax']

    x=  data[:,1]
    dt =data[:,0][1] -data[:,0][0];  fs=1/dt
    nperseg = int((ncycles/fc)*fs)
    noverlap = int(nperseg* overlap)

    nfft =int(2**(np.ceil(np.log2(nperseg))  + nextpow2))
    t=0  # starting of time axis

    step = nperseg - noverlap
    if boundary_zeroes is True:
        # add zeros to boundary
        zeros_shape = list(x.shape)
        zeros_shape[-1] = nperseg//2
        zeros = np.zeros(zeros_shape, dtype=x.dtype)
        x = np.concatenate((zeros, x, zeros), axis=-1)
        # padd zeros to make integer number of segments
        nadd = (-(x.shape[-1]-nperseg) % step) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        t -= nperseg/2/fs

    # Data slicing
   
    shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
    strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
    x_slice_2D = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    x_slice_2D = x_slice_2D*sg.get_window('hann', nperseg)

    Xfft = ((fft(x_slice_2D,nfft,axis=1)[:,:nfft//2])/(sg.get_window('hann', nperseg).sum())).T
    f =fftfreq(n=nfft, d=dt)[:nfft//2]
    t += (np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1, step))/fs #add initial time
    t +=  data[:,0][0]

    f_indx = np.logical_and(f>=fmin, f<=fmax)
    return t, f[f_indx], np.abs(Xfft)[f_indx,:]
    # return t, f, np.abs(Xfft)



def wavelet_fn(data, params):
    if isinstance(params['TF-options']['select'], str):
        params['TF-options']['select'] = eval(params['TF-options']['select'])

    method =params['TF-options']['select']['val']

    t = data[:,0]
    x=  data[:,1]
    dt =t[1] -t[0]

    mu= float(params[method]['mu'])
    nv= int(params[method]['nv'])
    fmin= int(params[method]['fmin'])
    fmax= int(params[method]['fmax'])
    s= float(params['wavelet-const']['s'])
    gamma = float(params['wavelet-const']['gamma'])
    scaletype =params['wavelet-const']['scaletype']
    if isinstance(params[method]['wavelets'], str):
        params[method]['wavelets'] = eval(params[method]['wavelets'])
    wlet = params[method]['wavelets']['val']       
   
    selector = {'bump': {'mu': mu, 's':s},
            'gmw': {'beta': mu, 'gamma':gamma}, 
            'morlet':{'mu': mu}, 
            'cmhat': {'mu': mu, 's':s}, 
            'hhhat':{'mu': mu}  }


    wavelet = (wlet, selector[wlet])
    padtype = 'reflect'
    # scaletype ='log-piecewise' #'linear'
    preset = 'maximal'
    # downsampling factor for higher scales (used only if `scaletype='log-piecewise'`)
    downsample = None# 4
    N= len(t)
    M = p2up(N)[0]
    wavelet = Wavelet(wavelet, N=M)

    Ncycle = (wavelet.std_t/(2*np.pi))*6   #window resolution

    min_scale, max_scale = cwt_scalebounds(wavelet, N=N, preset=preset)
    scales = make_scales(N, min_scale, max_scale, nv=nv, scaletype=scaletype,
                        wavelet=wavelet, downsample=downsample)
    f_c   = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        f_c[i]    = (center_frequency(wavelet, float(scale), M)/(2*np.pi))/dt

    fc_inx = np.logical_and(f_c>fmin, f_c<fmax)
    scales = scales[fc_inx]
    f_c   = np.zeros(len(scales))
    for i, scale in enumerate(scales):
        f_c[i]    = (center_frequency(wavelet, float(scale), M)/(2*np.pi))/dt


    td = wavelet.std_t/(f_c*2*np.pi)

    Tx, Wx, ssq_freqs,*_ = ssq_cwt(x, wavelet, scales=scales, padtype=padtype, fs=1/dt)

    if method == "cwt":
        return t, f_c, np.abs(Wx), Ncycle
    else:
        return t, ssq_freqs, np.abs(Tx), Ncycle

    


            

    


