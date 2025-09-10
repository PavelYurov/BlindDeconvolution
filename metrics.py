import cv2 as cv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def PSNR(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Вычисляет отношение пикового сигнала к шуму (PSNR) между изображениями.
    
    Аргументы:
        original: Исходное изображение
        restored: Восстановленное/обработанное изображение
        
    Возвращает:
        Значение PSNR в децибелах (dB)
    """
    return peak_signal_noise_ratio(original, restored)

def SSIM(original: np.ndarray, restored: np.ndarray) -> float:
    """
    Вычисляет индекс структурного сходства (SSIM) между изображениями.
    
    Аргументы:
        original: Исходное изображение
        restored: Восстановленное/обработанное изображение
        
    Возвращает:
        Значение SSIM в диапазоне от 0 до 1
    """
    return structural_similarity(original, restored)