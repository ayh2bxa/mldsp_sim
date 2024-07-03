import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import kl_div

mix, sr = librosa.load("mixed.wav", sr=None)
hop_size = 256
win_size = 2048
spectrogrammix = librosa.stft(mix, n_fft=win_size, hop_length=hop_size, center=False, win_length=win_size)
V = np.abs(spectrogrammix)
phase = np.angle(spectrogrammix)
max_iter = 2000
n_fft, t = V.shape
num_src = 30

def nmf(n_iter, thres=1e-4):
    W = np.genfromtxt("W_init.csv",delimiter=",", dtype=float)
    H = np.genfromtxt("H_init.csv",delimiter=",", dtype=float)
    W = W[:, 0:num_src]
    H = H[0:num_src, 0:t]
    ones = np.ones((n_fft, t))
    i = 0
    for _ in range(n_iter):
        W = W*((V/(W@H)@H.T)/(ones@H.T))
        H = H*((W.T@(V/(W@H)))/(W.T@ones))
        loss = np.sum(kl_div(V, W@H))
        i += 1
        if loss < thres:
            break
    return W, H

if __name__ == '__main__':
    W, H = nmf(max_iter)
    rows = 5
    cols = num_src//rows
    f, axs = plt.subplots(nrows=rows, ncols=cols,figsize=(25,10))
    indiv_srcs = [None]*num_src
    for i in range(num_src):
        row_idx = i // cols
        col_idx = i % cols
        axs[row_idx, col_idx].set_title(f"Source {i}")
        srci_mag_spec = (W[:,[i]]@H[[i],:])
        graph = librosa.amplitude_to_db(srci_mag_spec)
        librosa.display.specshow(graph, y_axis = 'hz', sr = sr, hop_length = hop_size, x_axis = 's', cmap= matplotlib.cm.jet, ax = axs[row_idx, col_idx])
        indiv_srcs[i] = srci_mag_spec*phase
    for i in range(num_src):
        src_i = (W[:, [i]]@H[[i], :])*phase 
        si_td = librosa.istft(src_i, hop_length = hop_size, center = False, win_length = win_size)
        sf.write(f"indiv_sources/source{i}.wav", si_td, sr)
    speech_src_ind = [0, 1, 2, 3, 4, 5, 7, 8, 9, 12, 13, 16, 17, 18, 19, 20, 21, 24, 26, 27]
    difference = set(range(num_src)) - set(speech_src_ind)
    music_src_ind = list(difference)
    src_speech = (W[:, speech_src_ind]@H[speech_src_ind, :])*phase
    speech_td = librosa.istft(src_speech, hop_length = hop_size, center = False, win_length = win_size)
    sf.write("speech.wav", speech_td, sr)
    src_mus = (W[:, music_src_ind]@H[music_src_ind, :])*phase 
    mus_td = librosa.istft(src_mus, hop_length = hop_size, center = False, win_length = win_size)
    sf.write("music.wav", mus_td, sr)