from .base import DeconvolutionAlgorithm
import cv2 as cv


class TestAlgorithm2(DeconvolutionAlgorithm):

    def __init__(self):
        super().__init__('TEST2 NOT FOR PUBLIC')
    
    def process(self,image):
        processed_image = cv.flip(image,1)
        return processed_image