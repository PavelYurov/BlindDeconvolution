from .base import DeconvolutionAlgorithm
import cv2 as cv
import numpy as np

class TestAlgorithm3(DeconvolutionAlgorithm):

    def __init__(self):
        super().__init__('TEST3 NOT FOR PUBLIC')
    
    def process(self,images):
        # processed_image = cv.flip(image,0)
        sum_image = np.zeros_like(images[0])
        for img in images:
            sum_image += img
    
        # Делим на количество изображений для получения среднего
        average_image = sum_image / len(images)
        return average_image
        pass