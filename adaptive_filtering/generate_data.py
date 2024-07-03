import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

# Parameters
duration = 20.0  # duration of the audio in seconds
sample_rate = 16000  # sample rate in Hz
amplitude = 0.3  # amplitude of the white noise

# Generate white noise
num_samples = int(duration * sample_rate)
white_noise = amplitude * np.random.randn(num_samples)

# Write the white noise to a mono audio file
output_file = 'reference.wav'
sf.write(output_file, white_noise, sample_rate)

print(f"White noise audio file written to {output_file}")

primary = np.zeros(white_noise.size)
speech, sr = sf.read("speech.wav")
# Design a low-pass filter using a Butterworth filter
def design_lowpass_filter(cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the low-pass filter
def apply_filter(data, cutoff, fs, order=2):
    b, a = design_lowpass_filter(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Apply the low-pass filter to the white noise
filtered_noise = 0.5*apply_filter(white_noise, 3000, sample_rate)
sf.write("filtered_noise.wav", filtered_noise, sample_rate)
primary += filtered_noise 
start = 2*sample_rate
primary[start:start+len(speech)] += 0.8*speech 
sf.write("primary.wav", primary, sample_rate)