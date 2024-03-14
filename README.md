from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import librosa
from librosa.feature import melspectrogram
import scipy
import numpy as np
astream, sr = librosa.load('RecordedAudio_0010.WAV', sr=None)
dt = 1.0 / sr
tstream = np.arange(len(astream))*dt

# filter the time range
t0 = 668
t1 = 670
idx = (tstream > t0) & (tstream < t1)
astream = astream[idx]

nfft = 400  
hl = 32

fig, ax = plt.subplots(figsize=(12,5))

D = librosa.stft(astream, n_fft=nfft, hop_length=hl)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

img = librosa.display.specshow(S_db, x_axis='s', y_axis='linear', sr=sr, ax=ax, hop_length=hl, cmap='jet')
fig.colorbar(img, ax=ax, format='%+2.0f dB')
plt.show()
