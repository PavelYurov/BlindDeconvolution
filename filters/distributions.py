import numpy as np

def gaussian_distribution(x: np.ndarray, std: float) -> np.ndarray:
    """
    Гауссовская функция распределения.
    
    Применение:
    - Для DefocusBlur: передаем 2D радиус (x = sqrt(x² + y²))
    - Для MotionBlur: передаем 1D координаты вдоль направления движения
    
    Параметры:
        r: Входной массив расстояний/координат
        std: Стандартное отклонение
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-x**2 / (2 * std**2))

def uniform_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Равномерная функция распределения.
    
    Применение:
    - Для DefocusBlur: создает диск (disk_psf)
    - Для MotionBlur: создает прямоугольное размытие
    
    Параметры:
        x: Входной массив расстояний/координат
        radius: Радиус/полуширина распределения
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return (x <= radius).astype(float)

def linear_decay_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Универсальная линейно убывающая функция распределения.
    
    Применение:
    - Для DefocusBlur: создает конус (cone_psf)
    - Для MotionBlur: создает треугольное размытие
    
    Параметры:
        x: Входной массив расстояний/координат
        radius: Радиус/полуширина распределения
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.clip(1 - x/radius, 0, None)

def ring_distribution(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Кольцевое распределение (специфично для размытия вне фокуса).
    
    Параметры:
        x: Входной массив расстояний
        radius: Радиус кольца
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-(x - radius)**2 / (0.1 * radius**2))

def exponential_decay_distribution(x: np.ndarray, scale: float) -> np.ndarray:
    """
    Экспоненциально убывающее распределение (специфично для размытия в движении).
    
    Параметры:
        x: 1D массив координат вдоль направления движения
        scale: Параметр масштаба
        
    Возвращает:
        Ненормализованные значения распределения
    """
    return np.exp(-np.abs(x)/scale)