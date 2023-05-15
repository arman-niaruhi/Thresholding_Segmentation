import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage.morphology import binary_opening, binary_closing
from skimage import measure
import math

class KMeans_thresh():


    '''
    for initialization we just need some current value wich can be passed 
    between the functions
    '''
    def __init__(self) -> None:
        self.back = None
        self.fore = None
        self.segmentationMap = None

    def apply(self, path1, path2, threshhold = 0.65, channel_num = 0 , closing = 2, opening= 23):

        '''
        path1 : the path belongs to the foreground image
        path2 : the path belongs to the Bakground image
        threshold : this should be used in thresholding process of one of the channels
        channel_num : Red or Green ir Blue for image
        closing & opening: size of kernel in closing & opening

        '''
        # import image
        img=Image.open(path1)
        x, y = img.size
        ratio = x/y
        img = img.resize((math.floor(400*ratio), 400), Image.LANCZOS)
        Img=np.array(img)/255.0


        # normalize the image 
        ImgNormalized = Img / np.maximum(Img.mean(axis=2)[:,:,np.newaxis],2/255)
        ImgNormalized = ImgNormalized / np.max(ImgNormalized)
        ImgNormalized2 = Img / np.maximum(np.sqrt(np.sum(Img**2,axis=2))[:,:,np.newaxis],5/255)


        # Consider a specific threshold and ignore the values under the thres in one channel
        segmentationMap = ImgNormalized2[:,:,channel_num]< threshhold
        se = np.ones((closing,closing), dtype=bool)
        segmentationMap = binary_closing(segmentationMap,se)
        se = np.ones((opening,opening), dtype=bool)
        segmentationMap = binary_opening(segmentationMap,se)


        # Inner functiob to analyze the biggest parts that are involved in a region
        def getLargestCC(segmentation):
            labels = measure.label(segmentation)
            assert( labels.max() != 0 ) # assume at least 1 CC
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
            return largestCC

        self.segmentationMap= getLargestCC(segmentationMap)

        # Save an plot the Image
        plt.imshow(Img * ~(self.segmentationMap[:,:,np.newaxis]))
        # plt.imshow(segmentationMap, alpha=0.8)#.mean(axis=2)<0.3)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/for_img.png')

        
        img2=Image.open(path2)
        x, y = img2.size
        ratio = x/y
        img2 = img2.resize((math.floor(1000*ratio), 1000), Image.LANCZOS)
        backImg=np.array(img2)/255.0

        plt.imshow(backImg)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/back_img.png')

        self.back = backImg
        self.fore = Img



    def combine(self):
        # merge the manipulated image and the given background
        manipulatedImg = self.back.copy()
        Img_foreground_after_thresholding = self.fore[25:-20, 25:-20,:]
        segmentationMap = self.segmentationMap[25:-20, 25:-20]

        ny,nx,nc = Img_foreground_after_thresholding.shape
        manipulatedImg[-ny-1:-1, math.floor(nx/2):math.floor(6*nx/4),:] = Img_foreground_after_thresholding*~segmentationMap[:,:,np.newaxis] + (1-~segmentationMap[:,:,np.newaxis])*self.back[-ny-1:-1, math.floor(nx/2):math.floor(6*nx/4) ,:]

        plt.imshow(manipulatedImg)
        plt.style.use(['dark_background'])
        plt.box(False)
        plt.savefig('./././static/images/combine_img.png')
