import numpy as np
import soundfile as sf

class AF_LMS:
    def __init__(self, step_size, filt_ord, primary, reference):
        self.N = min(len(primary), len(reference))
        self.W = np.zeros(filt_ord)
        self.W[(filt_ord-1)//2] = 1
        self.pri = primary[:self.N]
        self.ref = reference[:self.N]
        self.L = filt_ord 
        self.u = step_size/self.L/np.var(self.ref)
        self.out = np.zeros(self.N)
    def process(self):
        for k in range(self.N):
            Xk = np.append(np.array([0]*(self.L-k-1)), self.ref[:k+1]) if k < self.L else self.ref[k-self.L+1:k+1]
            Xk = np.flip(Xk)
            yk = np.dot(Xk, self.W)
            eps = self.pri[k]-yk 
            gradient = -2*eps*Xk
            self.W -= self.u*gradient
            self.out[k] = eps
        return self.out

# Recursive Least Square, aka LMS-Newton
class AF_RLS:
    def __init__(self, step_size, filt_ord, primary, reference):
        self.N = min(len(primary), len(reference))
        self.W = np.zeros(filt_ord) 
        self.W[(filt_ord-1)//2] = 1
        self.u = step_size 
        self.pri = primary[:self.N]
        self.ref = reference[:self.N] 
        self.L = filt_ord
        self.out = np.zeros(self.N)
        self.R_inv_hat = np.identity(self.L)/np.var(self.ref)
    def process(self, alpha = 1e-4):
        for k in range(self.N):
            Xk = np.append(np.array([0]*(self.L-k-1)), self.ref[:k+1]) if k < self.L else self.ref[k-self.L+1:k+1]
            Xk = np.flip(Xk)
            Sk_hat = self.R_inv_hat@Xk
            Sk_hat = Sk_hat.reshape(-1, 1)
            self.R_inv_hat = (self.R_inv_hat-alpha*(Sk_hat@Sk_hat.T)/(1-alpha+alpha*np.dot(Xk, Sk_hat)))/(1-alpha)
            dk = self.pri[k]
            ek = dk-np.dot(Xk, self.W)
            self.W += 2*self.u*ek*(self.R_inv_hat@Xk)
            self.out[k] = ek
        return self.out
        
if __name__ == '__main__':
    pri, sr = sf.read("primary.wav")
    ref, sr = sf.read("reference.wav")
    lms = AF_LMS(0.06, 60, pri, ref)
    output_lms = lms.process()
    sf.write("output_lms.wav", output_lms, sr)
    rls = AF_RLS(1e-4, 60, pri, ref)
    output_rls = rls.process()
    sf.write("output_rls.wav", output_rls, sr)
    
    
        