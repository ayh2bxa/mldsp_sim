import numpy as np
import soundfile as sf

class AF_LMS:
    def __init__(self, step_size, filt_ord, primary, reference):
        self.N = min(len(primary), len(reference))
        self.W = np.zeros(filt_ord)
        self.u = step_size
        self.pri = primary[:self.N]
        self.ref = reference[:self.N]
        self.L = filt_ord 
        self.out = np.zeros(self.N)
        self.Xs = np.zeros((self.N, self.L))
    def process(self):
        for k in range(self.N):
            Xk = np.append(self.ref[:k], np.array([0]*(self.L-k))) if k < self.L else self.ref[k-self.L:k]
            Xk = np.flip(Xk)
            self.Xs[k, :] = Xk
            yk = np.dot(Xk, self.W)
            dk = self.pri[k]
            self.W -= 2*self.u*Xk*(yk-dk)
            self.out[k] = dk-yk 
        return self.out
        # for k in range(self.N):
        #     self.out[k] = np.dot(self.Xs[k, :], self.W)
        # return self.pri-self.out
            
if __name__ == '__main__':
    pri, sr = sf.read("primary.wav")
    ref, sr = sf.read("reference.wav")
    lms = AF_LMS(0.02, 20, pri, ref) 
    output = lms.process()
    sf.write("output_lms.wav", output, sr)
    
        