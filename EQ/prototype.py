import numpy as np 
import scipy
from scipy import signal
from scipy.signal import tf2zpk
import matplotlib.pyplot as plt

# class EQFilter:
#     def __init__(self, N, type="lowpass"):
N = 4
ks = np.arange(1, N+1)
gamma_ks = np.pi*((2*ks-1)/N-1)/4
c = 1
for i in range(len(ks)):
    c /= (2*np.cos(gamma_ks[i]))
numerators_roots = (-1, )*N
numerators = c*np.poly(numerators_roots)
print(numerators)
denominators_roots = np.array([]) if N == 1 else 1j*np.tan((2*ks-N-1)*np.pi/4/N)
denominators = np.array([1]) if N == 1 else np.poly(denominators_roots) 
print(denominators_roots)
w, h = signal.freqz(numerators, denominators)
# print(tf2zpk(numerators, denominators))
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response')
ax1.plot(w/np.pi, abs(h)**2, 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
plt.show()

# import numpy as np
# import scipy.signal as signal

# # Define the order of the filter
# N = 4  # Example order

# # Calculate the gamma values
# k = np.arange(1, N + 1)
# gamma_k = np.pi * ((2 * k - 1) / (N - 1)) / 4

# # Calculate the coefficients of the numerator and denominator
# numerator_coeffs = [1] + [1]
# denominator_coeffs = []

# for gamma in gamma_k:
#     a1 = 2 * np.cos(gamma)
#     a0 = -1j * np.tan(gamma)
#     denominator_coeffs.append([a1, a0, 1])

# # Combine the sections
# num_combined = numerator_coeffs
# den_combined = denominator_coeffs[0]

# for i in range(1, N):
#     den_combined = np.convolve(den_combined, denominator_coeffs[i])

# # Normalize the coefficients
# den_combined = np.real(den_combined / den_combined[0])

# # Print the time-domain coefficients
# print("Numerator coefficients:", num_combined)
# print("Denominator coefficients:", den_combined)
# import matplotlib.pyplot as plt

# # Frequency response
# w, h = signal.freqz(num_combined, den_combined)

# # Plot frequency response
# plt.plot(w, 20 * np.log10(abs(h)))
# plt.title('Frequency response')
# plt.xlabel('Frequency [radians / second]')
# plt.ylabel('Amplitude [dB]')
# plt.grid()
# plt.show()

# # Impulse response
# impulse = np.zeros(100)
# impulse[0] = 1
# response = signal.lfilter(num_combined, den_combined, impulse)

# # Plot impulse response
# plt.stem(response)
# plt.title('Impulse response')
# plt.xlabel('Samples')
# plt.ylabel('Amplitude')
# plt.grid()
# plt.show()
