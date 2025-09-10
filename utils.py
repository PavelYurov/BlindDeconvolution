import cv2 as cv
import numpy as np
from typing import List, Optional

class Image:
    """
    Класс для управления путями к изображениям и метриками качества.
    
    Атрибуты:
        original_path (str): Путь к исходному изображению
        blurred_path (Optional[str]): Путь к размытому изображению
        blurred_array (np.array): несколько вариантов одной размытого изображения
        restored_paths (List[str]): Список путей к восстановленным изображениям
        is_color (bool): Цветное или черно-белое изображение
        psnr (np.ndarray): Значения PSNR для восстановленных изображений
        ssim (np.ndarray): Значения SSIM для восстановленных изображений
        algorithm (np.ndarray): Названия использованных алгоритмов восстановления
        filters (np.ndarray): Названия использованных фильров зашумления
        curent_filter(str): текущий фильтр
    """
        
    def __init__(self, original_path: str, is_color: bool) -> None:
        """
        Инициализация с путем к исходному изображению.
        
        Аргументы:
            original_path: Путь к исходному изображению
            is_color: Флаг цветного изображения
        """
        self.original_path = original_path
        self.blurred_path = None
        
        self.restored_paths = []
        self.is_color = is_color
        self.psnr = np.array([])
        self.ssim = np.array([])
        self.algorithm = np.array([])

        self.filters = np.array([])
        self.blurred_array = np.array([])
        self.current_filter = None
        
    def save_filter(self):
        if(self.blurred_path is not None):
            self.blurred_array = np.append(self.blurred_array,self.blurred_path)
        # else:
            # self.blurred_array = np.append(self.blurred_array,self.original_path)
            # raise(Exception("No filtered image not allowed!"))
        if(self.current_filter is not None):
            self.filters = np.append(self.filters,self.current_filter)
        # else:
        #     self.filters = np.append(self.filters,None)#?

        self.current_filter = None
        self.blurred_path = None

    def load(self, index):
        if (index >= self.get_len_filter()):
            # raise(Exception("index out of bounds"))
            if self.current is not None:
                print("current blurred image is not empty, you will lose it")
                self.current_filter = None
                self.blurred_path = None
                return
        if self.current_filter is not None:
            print("current blurred image is not empty, seve it if you don't want to lose it")
        self.current_filter = self.filters[index]
        self.blurred_path = self.blurred_array[index]
        self.filters = np.delete(self.filters,index,0)
        self.blurred_array = np.delete(self.blurred_array,index,0)
    
    def get_len_filter(self):
        if(len(self.filters)!=len(self.blurred_array)):
            raise(Exception("filters and blured images are not same amount"))
        return len(self.filters)
    
    def get_len_algorithms(self):
        return len(self.algorithm) #+1 current

    def get_blurred_array(self):
        return self.blurred_array
    
    def set_blurred_array(self, array):
        self.blurred_array = array

    def get_filters(self):
        return self.filters
    
    def set_filters(self, filters):
        self.filters = filters

    def set_current_filter(self, filter_str):
        self.current_filter = filter_str

    def get_current_filter(self):
        return self.current_filter

    def add_to_current_filter(self, filter_str):
        if self.current_filter is None:
            self.current_filter = filter_str
        else:
            self.current_filter = f"{self.current_filter}{filter_str}"

    def set_original(self, original_path) -> None:
        self.original_path = original_path
    
    def set_blured(self, blurred_path) -> None:
        self.blurred_path = blurred_path

    def set_restored(self, restored_paths) -> None:
        self.restored_paths = restored_paths

    def add_restored(self, restored_paths) -> None:
        self.restored_paths = np.append(self.restored_paths, restored_paths)

    def get_original(self) -> str:
        return self.original_path
    
    def get_blured(self) -> str:
        return self.blurred_path

    def get_restored(self) -> str:
        return self.restored_paths              #.reshape(len(self.blurred_array)+1,len(self.algorithm))
    
    def get_color(self) -> bool:
        return self.is_color
    
    def set_PSNR(self, psnr: float) -> None:
        self.psnr = psnr

    def set_SSIM(self, ssim: float) -> None:
        self.ssim = ssim

    def add_PSNR(self, psnr: float) -> None:
        self.psnr = np.append(self.psnr,psnr)

    def add_SSIM(self, ssim: float) -> None:
        self.ssim = np.append(self.ssim,ssim)
    
    def get_PSNR(self) -> float:
        return self.psnr                #.reshape(len(self.blurred_array)+1,len(self.algorithm))

    def get_SSIM(self) -> float:
        return self.ssim                #.reshape(len(self.blurred_array)+1,len(self.algorithm))

    def set_algorithm(self, algorithm) -> None: #название алгоритма, которым обрабатывалось
        self.algorithm = algorithm

    def add_algorithm(self, algorithm) -> None:  #название алгоритма, которым обрабатывалось
        self.algorithm = np.append(self.algorithm, algorithm)
    
    def get_algorithm(self) -> str: #название алгоритма, которым обрабатывалось
        return self.algorithm

    def get_original_image(self) -> np.ndarray:
        return cv.imread(
            self.original_path, 
            cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
        )
    
    def get_blured_image(self) -> Optional[np.ndarray]:
        if self.blurred_path is None:
            return None
        return cv.imread(
            self.blurred_path,
            cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
        )
    
    def get_all_blurred_images(self):
        res = [
            cv.imread(
                path,
                cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
            )
            for path in self.blurred_array
            ]
        if self.blurred_path is not None:
            res = np.append(res, cv.imread(self.blurred_path,cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE))
        return res
    
    def get_all_filters(self):
        res = self.filters
        if self.current_filter is not None:
            res = np.append(res,self.current_filter)
        return res

    
    def get_restored_image(self) -> List[np.ndarray]:
        return np.array([
            cv.imread(
                path,
                cv.IMREAD_COLOR if self.is_color else cv.IMREAD_GRAYSCALE
            )
            for path in self.restored_paths
        ])
    
    
    
    
