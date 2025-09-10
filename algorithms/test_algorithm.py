from .base import DeconvolutionAlgorithm
import cv2 as cv


class TestAlgorithm(DeconvolutionAlgorithm):

    def __init__(self):
        super().__init__('TEST NOT FOR PUBLIC')
    
    def process(self,image):
        processed_image = cv.flip(image,0)
        return processed_image
    