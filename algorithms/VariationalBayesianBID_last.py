import numpy as np
from scipy.sparse import diags, eye, issparse
from scipy.sparse.linalg import LinearOperator, cg
from scipy.signal import convolve2d, fftconvolve
import cv2
import matplotlib.pyplot as plt
import warnings
import gc

# Optional research dependencies (guarded so the framework import doesn't fail)
try:
    from sklearn.linear_model import Lasso
except Exception:  # pragma: no cover
    Lasso = None
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x
try:
    import pywt  # type: ignore
except Exception:  # pragma: no cover
    pywt = None
try:
    from scipy.sparse import random as sparse_random  # type: ignore
except Exception:  # pragma: no cover
    sparse_random = None
try:
    from scipy.optimize import minimize  # type: ignore
except Exception:  # pragma: no cover
    minimize = None

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


# === Integrated, framework-ready implementation ===
# Lightweight VB-style blind deconvolution adapter compatible with the framework.
# Alternates Wiener deconvolution for the latent image with Richardson–Lucy-style
# kernel updates. Works on grayscale or color (single kernel estimated on grayscale).

from .base import DeconvolutionAlgorithm

class VariationalBayesianBIDAlgorithm(DeconvolutionAlgorithm):
    """
    Practical VB-style Blind Deconvolution (framework-ready).

    - Alternates:
      1) Latent image update via Wiener deconvolution
      2) Kernel update via RL-style multiplicative rule

    Notes:
    - Returns uint8 image with original shape.
    - For color images, estimates a single blur kernel on the grayscale proxy
      and deconvolves each channel with that kernel.
    """

    def __init__(self,
                 kernel_size: int = 15,
                 outer_iters: int = 10,
                 x_wiener_reg: float = 1e-2,
                 kernel_update_iters: int = 5,
                 eps: float = 1e-6):
        super().__init__('VariationalBayesianBID')
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.outer_iters = max(1, outer_iters)
        self.x_wiener_reg = float(x_wiener_reg)
        self.kernel_update_iters = max(1, kernel_update_iters)
        self.eps = float(eps)

    # ---- helpers ----
    def _normalize_kernel(self, k: np.ndarray) -> np.ndarray:
        k = np.clip(k, 0, None)
        s = k.sum()
        if s <= 0:
            # fallback: delta kernel
            k[:] = 0
            c = k.shape[0]//2, k.shape[1]//2
            k[c] = 1.0
            return k
        return k / s

    def _fft_conv_same(self, img: np.ndarray, k: np.ndarray) -> np.ndarray:
        return fftconvolve(img, k, mode='same')

    def _wiener_deconv(self, y: np.ndarray, k: np.ndarray, reg: float) -> np.ndarray:
        # y, k in float64, y in [0,1]
        H = np.fft.fft2(k, s=y.shape)
        Y = np.fft.fft2(y)
        denom = (np.abs(H) ** 2) + reg
        X = (np.conj(H) * Y) / np.maximum(denom, self.eps)
        x = np.fft.ifft2(X).real
        return np.clip(x, 0.0, 1.0)

    def _flip_kernel(self, k: np.ndarray) -> np.ndarray:
        return np.flip(np.flip(k, axis=0), axis=1)

    def _center_crop(self, arr: np.ndarray, h: int, w: int) -> np.ndarray:
        H, W = arr.shape
        sh = max((H - h) // 2, 0)
        sw = max((W - w) // 2, 0)
        return arr[sh:sh+h, sw:sw+w]

    def _update_kernel_RL(self, y: np.ndarray, x: np.ndarray, k: np.ndarray, iters: int) -> np.ndarray:
        # Multiplicative update: k <- k * (flip(x) * (y / (x*k)))
        k = k.copy()
        for _ in range(iters):
            xk = self._fft_conv_same(x, k)
            ratio = y / np.maximum(xk, self.eps)
            corr_full = self._fft_conv_same(ratio, self._flip_kernel(x))
            # Map adjoint result to kernel support (center crop)
            corr = self._center_crop(corr_full, k.shape[0], k.shape[1])
            k = k * corr
            k = self._normalize_kernel(k)
        return k

    def _estimate_single_kernel(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # y in [0,1], grayscale
        k = np.zeros((self.kernel_size, self.kernel_size), dtype=np.float64)
        k[self.kernel_size//2, self.kernel_size//2] = 1.0  # delta init
        x = y.copy()
        for _ in range(self.outer_iters):
            x = self._wiener_deconv(y, k, self.x_wiener_reg)
            k = self._update_kernel_RL(y, x, k, self.kernel_update_iters)
        return x, k

    def _deconv_channel(self, ch: np.ndarray, k: np.ndarray) -> np.ndarray:
        ch_f = ch.astype(np.float64) / 255.0
        x = self._wiener_deconv(ch_f, k, self.x_wiener_reg)
        return np.clip(x * 255.0, 0, 255).astype(np.uint8)

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError('Input image is None')

        # Handle grayscale or color
        if image.ndim == 2:
            y_gray = image.astype(np.float64) / 255.0
            x_est, k_est = self._estimate_single_kernel(y_gray)
            out = np.clip(x_est * 255.0, 0, 255).astype(np.uint8)
            return out
        elif image.ndim == 3 and image.shape[2] == 3:
            # Estimate kernel on grayscale proxy
            y_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            _, k_est = self._estimate_single_kernel(y_gray)
            # Deconvolve each channel with the estimated kernel
            b, g, r = cv2.split(image)
            b_r = self._deconv_channel(b, k_est)
            g_r = self._deconv_channel(g, k_est)
            r_r = self._deconv_channel(r, k_est)
            return cv2.merge([b_r, g_r, r_r])
        else:
            raise ValueError('Unsupported image shape for VariationalBayesianBID')

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
