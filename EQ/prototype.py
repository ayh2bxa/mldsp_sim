import numpy as np 
import scipy
from scipy import signal
from scipy.signal import tf2zpk
import matplotlib.pyplot as plt
import argparse

class CanonicalLPF:
    def __init__(self, order):
        self.N = order
        ks = np.arange(1, self.N+1)
        gamma_ks = np.pi*((2*ks-1)/self.N-1)/4
        self.canon_g = 1
        for i in range(len(ks)):
            self.canon_g /= (2*np.cos(gamma_ks[i]))
        self.canon_numerators_roots = (-1, )*self.N
        numerators = self.canon_g*np.poly(self.canon_numerators_roots)
        self.canon_denominators_roots = 1j*np.tan((2*ks-self.N-1)*np.pi/4/self.N)
        denominators = np.poly(self.canon_denominators_roots)

    def set_cutoff(self, fc):
        alpha = (np.tan(fc/2)-1)/(np.tan(fc/2)+1)
        beta = (1-np.tan(fc/2))/(1+np.tan(fc/2))
        # beta = np.sin((np.pi/2-fc)/2)/np.sin((np.pi/2+fc)/2)
        denominators_roots = np.array([(p+beta)/(1+beta*p) for p in self.canon_denominators_roots])
        print(self.canon_denominators_roots)
        print(denominators_roots)
        denominators = np.poly(denominators_roots)
        print(denominators)
        g = self.canon_g 
        for _ in range(self.N):
            g *= (1-beta)
        numerators = g*np.poly(self.canon_numerators_roots)
        
        w, h = signal.freqz(numerators, denominators)
        fig, ax1 = plt.subplots()
        ax1.set_title('Digital filter frequency response')
        ax1.plot(w/np.pi, abs(h)**2, 'b')
        ax1.set_ylabel('Amplitude [dB]', color='b')
        ax1.set_xlabel('Frequency [rad/sample]')
        plt.show()
        
        
        
if __name__ == '__main__':
    lpf = CanonicalLPF(2)
    lpf.set_cutoff(np.pi/4)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--type', type=str, default='lowpass', help='Filter type: lowpass, bandpass, highpass, shelf')
    # parser.add_argument('-o', '--order', type=int, default='1', help='Enter a filter order between 1 and 8 inclusive')