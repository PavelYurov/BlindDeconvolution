import numpy as np
from scipy.sparse import diags, eye, issparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import convolve2d, fftconvolve
from sklearn.linear_model import Lasso
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import gc
import pywt
from scipy.sparse import random as sparse_random
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

class VariationalBayesianBID:
    def __init__(self, p=1.2, lambda1=0.01, eta=0.1, max_outer_iter=5, max_inner_iter=2, patch_size=256):
        self.p = p
        self.lambda1 = lambda1
        self.eta = eta
        self.max_outer_iter = max_outer_iter
        self.max_inner_iter = max_inner_iter
        self.patch_size = patch_size
        self.dtype = np.float32
        
    def _compute_derivatives(self, x):
        dx = np.empty_like(x, dtype=self.dtype)
        dy = np.empty_like(x, dtype=self.dtype)
        
        dx[:, :-1] = x[:, 1:] - x[:, :-1]
        dx[:, -1] = 0
        
        dy[:-1, :] = x[1:, :] - x[:-1, :]
        dy[-1, :] = 0
        
        dxx = np.empty_like(x, dtype=self.dtype)
        dxx[:, 1:-1] = x[:, 2:] + x[:, :-2] - 2*x[:, 1:-1]
        dxx[:, 0] = dxx[:, -1] = 0
        
        dyy = np.empty_like(x, dtype=self.dtype)
        dyy[1:-1, :] = x[2:, :] + x[:-2, :] - 2*x[1:-1, :]
        dyy[0, :] = dyy[-1, :] = 0
        
        dxy = np.empty_like(x, dtype=self.dtype)
        dxy[:-1, :-1] = dx[1:, :-1] - dx[:-1, :-1]
        dxy[-1, :] = dxy[:, -1] = 0
        
        return {'h': dx, 'v': dy, 'hh': dxx, 'vv': dyy, 'hv': dxy}
    
    def _compute_laplacian(self, h):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=self.dtype)
        return convolve2d(h, kernel, mode='same', boundary='symm')
    
    def _compute_alpha(self, x):
        derivs = self._compute_derivatives(x)
        numerator = self.lambda1 * (x.size / max(self.p, 1e-6)) + 1
        denominator = 1e-6
        
        weights = {'h': 0.5, 'v': 0.5, 'hh': 0.25, 'vv': 0.25, 'hv': 0.25}
        for d in derivs:
            denominator += weights[d] * np.sum(np.abs(derivs[d])**self.p)
            
        return min(numerator / denominator, 1e6)
    
    def _compute_gamma(self, h):
        Ch = self._compute_laplacian(h)
        denominator = np.sum(Ch**2) + 1e-6
        return min((h.size + 2) / denominator, 1e6)
    
    def _compute_v(self, x):
        derivs = self._compute_derivatives(x)
        return {d: derivs[d]**2 + 1e-6 for d in derivs}
    
    def _apply_convolution(self, x, h):
        if x.shape[0] > 512 or x.shape[1] > 512:
            result = np.zeros_like(x, dtype=self.dtype)
            pad = h.shape[0] // 2
            x_padded = np.pad(x, ((pad, pad), (pad, pad)), mode='reflect')
            
            for i in range(0, x.shape[0], self.patch_size):
                for j in range(0, x.shape[1], self.patch_size):
                    patch = x_padded[i:i+self.patch_size+2*pad, j:j+self.patch_size+2*pad]
                    conv_patch = fftconvolve(patch, h, mode='valid')
                    result[i:i+self.patch_size, j:j+self.patch_size] = conv_patch
            return result
        else:
            return fftconvolve(x, h, mode='same')
    
    def _update_x_patch(self, a_patch, h, u_patch, alpha, v_patch, W_patch, Phi_patch):
        patch_shape = (self.patch_size, self.patch_size)
        N_patch = self.patch_size * self.patch_size
        
        Wha_patch = np.zeros(N_patch, dtype=self.dtype)
        for i in range(W_patch.shape[0]):
            Wha_patch[i] = np.dot(W_patch[i].toarray().ravel() if issparse(W_patch) else W_patch[i], a_patch)
        
        def conv_operator(x):
            x_img = x.reshape(patch_shape)
            return self._apply_convolution(x_img, h).ravel()
        
        H = LinearOperator((N_patch, N_patch), matvec=conv_operator, rmatvec=conv_operator)
        
        B_diag = v_patch['h'].ravel()**(self.p/2 - 1) + v_patch['v'].ravel()**(self.p/2 - 1)
        
        def A_operator(x):
            Hx = H.matvec(x)
            HT_Hx = H.rmatvec(Hx)
            Bx = B_diag * x
            return self.eta * HT_Hx + alpha * self.p * Bx
        
        A = LinearOperator((N_patch, N_patch), matvec=A_operator)
        b = self.eta * H.rmatvec(Wha_patch + u_patch.ravel())
        
        x, _ = cg(A, b, maxiter=30, tol=1e-2)
        return x.reshape(patch_shape)
    
    def _update_h(self, a, x, u, gamma):
        x_fft = np.fft.fft2(x, s=(x.shape[0]+self.kernel_shape[0]-1, 
                                 x.shape[1]+self.kernel_shape[1]-1))
        
        def objective(h_vec):
            h = h_vec.reshape(self.kernel_shape)
            conv = np.fft.ifft2(x_fft * np.fft.fft2(h, s=x_fft.shape)).real
            conv = conv[:x.shape[0], :x.shape[1]] 
            
            data_term = 0.5 * self.eta * np.sum((conv.ravel() - (self.W @ a + u))**2)
            
            laplacian = convolve2d(h, np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=self.dtype), mode='same')
            reg_term = 0.5 * gamma * np.sum(laplacian**2)
            
            return data_term + reg_term
        
        h0 = np.zeros(self.kernel_shape, dtype=self.dtype)
        center = (self.kernel_shape[0]//2, self.kernel_shape[1]//2)
        h0[center] = 1
        
        res = minimize(objective, h0.ravel(), method='L-BFGS-B', 
                      options={'maxiter': 5, 'maxls': 5, 'maxfun': 10})
        
        return res.x.reshape(self.kernel_shape)
    
    def _process_image_in_patches(self, img, func, **kwargs):
        result = np.zeros_like(img, dtype=self.dtype)
        pad = self.kernel_shape[0] // 2 
        
        for i in range(0, img.shape[0], self.patch_size):
            for j in range(0, img.shape[1], self.patch_size):
                i_start = max(0, i - pad)
                j_start = max(0, j - pad)
                i_end = min(img.shape[0], i + self.patch_size + pad)
                j_end = min(img.shape[1], j + self.patch_size + pad)
                
                patch = img[i_start:i_end, j_start:j_end]

                processed_patch = func(patch, **kwargs)
                
                update_i_start = i if i_start == i else pad
                update_j_start = j if j_start == j else pad
                update_i_end = update_i_start + min(self.patch_size, img.shape[0] - i)
                update_j_end = update_j_start + min(self.patch_size, img.shape[1] - j)
                
                result[i:i+(update_i_end-update_i_start), 
                       j:j+(update_j_end-update_j_start)] = processed_patch[update_i_start:update_i_end, 
                                                                           update_j_start:update_j_end]
        return result
    
    def deconvolve(self, y, Phi, W, img_shape, kernel_shape):
        self.img_shape = img_shape
        self.kernel_shape = kernel_shape
        self.W = W
        self.Phi = Phi
        
        a = np.random.randn(W.shape[1]).astype(self.dtype)
        x = (W @ a).reshape(img_shape).astype(self.dtype)
        h = np.zeros(kernel_shape, dtype=self.dtype)
        center = (kernel_shape[0]//2, kernel_shape[1]//2)
        h[center] = 1  
        u = np.zeros(img_shape[0] * img_shape[1], dtype=self.dtype)
        
        results = {'x': [], 'h': [], 'a': []}
        
        for k in tqdm(range(self.max_outer_iter), desc="Outer iterations"):
            def x_update_func(patch):
                alpha = self._compute_alpha(patch)
                v_patch = self._compute_v(patch)
                
                i, j = 0, 0  
                u_patch = u[i*patch.shape[0]:(i+1)*patch.shape[0], 
                           j*patch.shape[1]:(j+1)*patch.shape[1]].ravel()
                
                return self._update_x_patch(a, h, u_patch, alpha, v_patch, 
                                          W[i*patch.shape[0]:(i+1)*patch.shape[0]], 
                                          Phi[:, i*patch.shape[0]:(i+1)*patch.shape[0]])
            
            x = self._process_image_in_patches(x, x_update_func)
            
            h = self._update_h(a, x, u, self._compute_gamma(h))
            
            conv = self._apply_convolution(x, h)
            residual = y - Phi @ (W @ a)
            beta = (len(y) + 2) / (np.sum(residual**2) + 1e-6)
            tau = (len(a) + 1) / (np.sum(np.abs(a)) + 1e-6)
            
            lasso = Lasso(alpha=tau/(2*beta), max_iter=50, selection='random', 
                         tol=1e-2, warm_start=True)
            lasso.fit(Phi @ W, y)
            a = lasso.coef_.astype(self.dtype)
            
            u = u + (W @ a - conv.ravel())
            
            gc.collect()
            
            if k % 1 == 0:
                results['x'].append(x.copy())
                results['h'].append(h.copy())
                results['a'].append(a.copy())
                
                plt.figure(figsize=(12, 4))
                plt.subplot(131)
                plt.imshow(x, cmap='gray')
                plt.title(f'Reconstructed (Iter {k+1})')
                
                plt.subplot(132)
                plt.imshow(h, cmap='gray')
                plt.title('Estimated Kernel')
                
                plt.subplot(133)
                plt.plot(a)
                plt.title('Sparse Coefficients')
                
                plt.tight_layout()
                plt.show()
                plt.close()
        
        return x, h, a, results

def create_measurement_matrix(M, N, dtype=np.float32):
    density = min(0.05, 500000/(M*N))  # Reduced density
    Phi = sparse_random(M, N, density=density, dtype=dtype)
    return Phi

def create_wavelet_matrix(N, dtype=np.float32):
    size = int(np.sqrt(N))
    wavelet = 'haar'
    
    def wavelet_transform(x):
        x_img = x.reshape(size, size)
        coeffs = pywt.wavedec2(x_img, wavelet, level=2)
        arr, _ = pywt.coeffs_to_array(coeffs)
        return arr.ravel()
    
    return LinearOperator((N, N), matvec=wavelet_transform, rmatvec=wavelet_transform, dtype=dtype)

def load_and_prepare_image(image_path, dtype=np.float32):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return (img.astype(dtype) / 255.0)

if __name__ == "__main__":
    img_path = 'image2.jpeg'
    img = load_and_prepare_image(img_path)
    img_shape = img.shape
    print(f"Original image size: {img_shape}")
    
    kernel_shape = (15, 15)
    M = min(2000, img_shape[0]*img_shape[1]//4)  
    
    try:
        true_kernel = np.zeros(kernel_shape, dtype=np.float32)
        center = (kernel_shape[0]//2, kernel_shape[1]//2)
        true_kernel[center] = 1
        true_kernel = convolve2d(true_kernel, np.ones((3,3), dtype=np.float32)/9, mode='same')
        
        blurred_img = convolve2d(img, true_kernel, mode='same', boundary='symm')
        
        print("Creating measurement matrices...")
        Phi = create_measurement_matrix(M, img_shape[0]*img_shape[1])
        W = create_wavelet_matrix(img_shape[0]*img_shape[1])
        
        print("Creating measurements...")
        y = Phi.dot(blurred_img.flatten()).astype(np.float32)
        y += 0.01 * np.random.randn(M).astype(np.float32)  
        
        print("Starting deconvolution...")
        bid = VariationalBayesianBID(max_outer_iter=5, max_inner_iter=1, patch_size=128)
        x_est, h_est, a_est, results = bid.deconvolve(y, Phi, W, img_shape, kernel_shape)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(141)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        
        plt.subplot(142)
        plt.imshow(blurred_img, cmap='gray')
        plt.title('Blurred Image')
        
        plt.subplot(143)
        plt.imshow(x_est, cmap='gray')
        plt.title('Reconstructed Image')
        
        plt.subplot(144)
        plt.imshow(h_est, cmap='gray')
        plt.title('Estimated Kernel')
        plt.colorbar()
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {str(e)}")