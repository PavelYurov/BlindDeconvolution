import abc
import numpy as np
from typing import Any

class DeconvolutionAlgorithm(abc.ABC):
    """
    Абстрактный базовый класс для алгоритмов деконволюции.
    
    Атрибуты:
        name (str): Название алгоритма
        params (Any): Параметры алгоритма
    """
    name = 'default' 
    param = None
    def __init__(self, name: str) -> None: 
        """
        Инициализация алгоритма деконволюции.
        
        Аргументы:
            params: Дополнительные параметры алгоритма
            name: Название алгоритма (должно быть уникальным)
        """
        super().__init__()
        self.name = name
        pass
    
    @abc.abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Обработка изображения с использованием алгоритма деконволюции.
        
        Аргументы:
            image: Входное размытое изображение в виде numpy массива
            
        Возвращает:
            Восстановленное изображение в виде numpy массива
        """
        pass

    def get_name(self) -> str:
        """Получение названия алгоритма."""
        return self.name