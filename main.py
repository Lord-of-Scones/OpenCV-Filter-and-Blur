import cv2
import numpy as np


# This is a function that is designed to convolve an image matrix passed to it alongside the kernel
# of the filter you want to apply.
def convolveFunction(Original_Image, filterKernel, kRadius, isColor):

    #  Image is padded with n rows and columns of pixels where n is the radius of the kernel
    padded_Image = cv2.copyMakeBorder(Original_Image, kRadius, kRadius, kRadius, kRadius, borderType=cv2.BORDER_REPLICATE, value=0)


    if(isColor == True):
        # Seperates the height, width, and amount of color channels into seperate variables from the original image.
        Height, Width, Channels = padded_Image.shape

        # Initializes copies of the padded image channels that will be overwritten by our convolution algorithim
        New_ImageB, New_ImageG, New_ImageR = cv2.split(padded_Image)

        # splits our padded image into its 3 color channels to be used in the convolution algorithim
        b, g, r = cv2.split(padded_Image)

        # Nested for loop that convolves the padded image and then overwrites the initialized channels we created earlier
        for i in range(kRadius, Height - kRadius - 1):
            for j in range(kRadius, Width - kRadius - 1):
                New_ImageB[i][j] = np.sum(b[i - kRadius:i + kRadius + 1, j - kRadius:j + kRadius + 1] * filterKernel)
                New_ImageG[i][j] = np.sum(g[i - kRadius:i + kRadius + 1, j - kRadius:j + kRadius + 1] * filterKernel)
                New_ImageR[i][j] = np.sum(r[i - kRadius:i + kRadius + 1, j - kRadius:j + kRadius + 1] * filterKernel)

        New_Image = cv2.merge([New_ImageB, New_ImageG, New_ImageR])
        Blurred_Image = np.array(New_Image[kRadius:Height - kRadius, kRadius:Width - kRadius], dtype=np.uint8)

        return Blurred_Image
    elif(isColor == False):

        # Seperates the height, width, and amount of color channels into seperate variables from the original image.
        Height, Width = Original_Image.shape

        New_ImageGray = padded_Image.copy()

        # Nested for loop that convolves the padded image and then overwrites the initialized channels we created earlier
        for i in range(kRadius, Height - kRadius - 1):
            for j in range(kRadius, Width - kRadius - 1):
                New_ImageGray[i][j] = np.sum(padded_Image[i - kRadius:i + kRadius + 1, j - kRadius:j + kRadius + 1] * filterKernel)

        Blurred_Image = np.array(New_ImageGray[kRadius:Height - kRadius, kRadius:Width - kRadius], dtype=np.uint8)
        return Blurred_Image





inputImage = cv2.imread("lena30.jpg", cv2.IMREAD_COLOR)
inputImage2 = cv2.imread("lena_gray.bmp", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture("lena_gray.gif", cv2.IMREAD_GRAYSCALE)
ret, inputImage3 = cap.read()
cap.release()


# Create a kernel for the 7by7 box blur
boxBlurBase = np.array([[1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1]], dtype='float32')
boxBlur_7by7 = boxBlurBase / 49


# A kernel for a 15by15 gaussian blur
GaussianBase = np.array([[1.9008004993123049, 2.847435993362043, 4.008365594694693, 5.302468291429146, 6.59154412469871, 7.700054525843593, 8.452716851189532, 8.719634165060608, 8.452716851189532, 7.700054525843593, 6.59154412469871, 5.302468291429146, 4.008365594694693, 2.847435993362043, 1.9008004993123049],
                         [2.847435993362043, 4.2655143131680875, 6.004609359591951, 7.943200284374287, 9.874260870244768, 11.534830938685987, 12.662333691769742, 13.062181001912489, 12.662333691769742, 11.534830938685987, 9.874260870244768, 7.943200284374287, 6.004609359591951, 4.2655143131680875, 2.847435993362043],
                         [4.008365594694693, 6.004609359591951, 8.452751746722004, 11.181726579940275, 13.900100875874816, 16.237702825641726, 17.824900309244597, 18.387769572976495, 17.824900309244597, 16.237702825641726, 13.900100875874816, 11.181726579940275, 8.452751746722004, 6.004609359591951, 4.008365594694693],
                         [5.302468291429146, 7.943200284374287, 11.181726579940275, 14.791752207442995, 18.387754909269105, 21.480052735851487, 23.579677665318872, 24.324269532664694, 23.579677665318872, 21.480052735851487, 18.387754909269105, 14.791752207442995, 11.181726579940275, 7.943200284374287, 5.302468291429146],
                         [6.59154412469871, 9.874260870244768, 13.900100875874816, 18.387754909269105, 22.857976922654117, 26.702039055675183, 29.312100937661754, 30.237709518185483, 29.312100937661754, 26.702039055675183, 22.857976922654117, 18.387754909269105, 13.900100875874816, 9.874260870244768, 6.59154412469871],
                         [7.700054525843593, 11.534830938685987, 16.237702825641726, 21.480052735851487, 26.702039055675183, 31.192563197671394, 34.24156331462672, 35.322832954150996, 34.24156331462672, 31.192563197671394, 26.702039055675183, 21.480052735851487, 16.237702825641726, 11.534830938685987, 7.700054525843593],
                         [8.452716851189532, 12.662333691769742, 17.824900309244597, 23.579677665318872, 29.312100937661754, 34.24156331462672, 37.5885960637284, 38.7755572822511, 37.5885960637284, 34.24156331462672, 29.312100937661754, 23.579677665318872, 17.824900309244597, 12.662333691769742, 8.452716851189532],
                         [8.719634165060608, 13.062181001912489, 18.387769572976495, 24.324269532664694, 30.237709518185483, 35.322832954150996, 38.7755572822511, 40, 38.7755572822511, 35.322832954150996, 30.237709518185483, 24.324269532664694, 18.387769572976495, 13.062181001912489, 8.719634165060608],
                         [8.452716851189532, 12.662333691769742, 17.824900309244597, 23.579677665318872, 29.312100937661754, 34.24156331462672, 37.5885960637284, 38.7755572822511, 37.5885960637284, 34.24156331462672, 29.312100937661754, 23.579677665318872, 17.824900309244597, 12.662333691769742, 8.452716851189532],
                         [7.700054525843593, 11.534830938685987, 16.237702825641726, 21.480052735851487, 26.702039055675183, 31.192563197671394, 34.24156331462672, 35.322832954150996, 34.24156331462672, 31.192563197671394, 26.702039055675183, 21.480052735851487, 16.237702825641726, 11.534830938685987, 7.700054525843593],
                         [6.59154412469871, 9.874260870244768, 13.900100875874816, 18.387754909269105, 22.857976922654117, 26.702039055675183, 29.312100937661754, 30.237709518185483, 29.312100937661754, 26.702039055675183, 22.857976922654117, 18.387754909269105, 13.900100875874816, 9.874260870244768, 6.59154412469871],
                         [5.302468291429146, 7.943200284374287, 11.181726579940275, 14.791752207442995, 18.387754909269105, 21.480052735851487, 23.579677665318872, 24.324269532664694, 23.579677665318872, 21.480052735851487, 18.387754909269105, 14.791752207442995, 11.181726579940275, 7.943200284374287, 5.302468291429146],
                         [4.008365594694693, 6.004609359591951, 8.452751746722004, 11.181726579940275, 13.900100875874816, 16.237702825641726, 17.824900309244597, 18.387769572976495, 17.824900309244597, 16.237702825641726, 13.900100875874816, 11.181726579940275, 8.452751746722004, 6.004609359591951, 4.008365594694693],
                         [2.847435993362043, 4.2655143131680875, 6.004609359591951, 7.943200284374287, 9.874260870244768, 11.534830938685987, 12.662333691769742, 13.062181001912489, 12.662333691769742, 11.534830938685987, 9.874260870244768, 7.943200284374287, 6.004609359591951, 4.2655143131680875, 2.847435993362043],
                         [1.9008004993123049, 2.847435993362043, 4.008365594694693, 5.302468291429146, 6.59154412469871, 7.700054525843593, 8.452716851189532, 8.719634165060608, 8.452716851189532, 7.700054525843593, 6.59154412469871, 5.302468291429146, 4.008365594694693, 2.847435993362043, 1.9008004993123049]], dtype='float32')

gaussianBlur_15x15 = GaussianBase/(np.sum(GaussianBase))


# Creating the kernel for the Motion blur function

MotionBlurBase = np.array([[1,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,0 ],
                           [0,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ]], dtype='float32')

MotionBlur_15x15 = (MotionBlurBase / 15)


# Opening all of the sample images to be filtered
inputImage = cv2.imread("lena5.jpg", cv2.IMREAD_COLOR)
inputImage2 = cv2.imread("lena10.jpg", cv2.IMREAD_COLOR)
inputImage3 = cv2.imread("lena15.jpg", cv2.IMREAD_COLOR)
inputImage4 = cv2.imread("lena30.jpg", cv2.IMREAD_COLOR)
inputImage5 = cv2.imread("lena_gray.bmp", cv2.IMREAD_GRAYSCALE)

cap = cv2.VideoCapture("lena_gray.gif", cv2.IMREAD_GRAYSCALE)
ret, inputImage6 = cap.read()
cap.release()



# Running convolution function while passing in a 7by7 box blur kernel

imgColor = True
# Designating kernel radius for this filter
kernalRadius = 3

Filtered_Image = convolveFunction(inputImage, boxBlur_7by7, kernalRadius, imgColor)
Filtered_Image2 = convolveFunction(inputImage2, boxBlur_7by7, kernalRadius, imgColor)
Filtered_Image3 = convolveFunction(inputImage3, boxBlur_7by7, kernalRadius, imgColor)
Filtered_Image4 = convolveFunction(inputImage4, boxBlur_7by7, kernalRadius, imgColor)
Filtered_Image6 = convolveFunction(inputImage6, boxBlur_7by7, kernalRadius, imgColor)
# Setting imageColor to false so I can filter the grayscale images
imgColor = False
Filtered_Image5 = convolveFunction(inputImage5, boxBlur_7by7, kernalRadius, imgColor)

# Displaying all of the filtered images by the box blur filter
cv2.imshow('BoxBlur_Before:', inputImage4)
cv2.imshow('BoxBlur_lena5:', Filtered_Image)
cv2.imshow('BoxBlur_lena10:', Filtered_Image2)
cv2.imshow('BoxBlur_lena15:', Filtered_Image3)
cv2.imshow('BoxBlur_lena30:', Filtered_Image4)
cv2.imshow('BoxBlur_lena_gray:', Filtered_Image5)
cv2.imshow('BoxBlur_lena_gray(gif)', Filtered_Image6)
cv2.waitKey(0)

# Witing all of the Box Blur filtered functions and saving them into seperate files

cv2.imwrite('BoxBlur_lena5.jpg', Filtered_Image)
cv2.imwrite('BoxBlur_lena10.jpg', Filtered_Image2)
cv2.imwrite('BoxBlur_lena15.jpg', Filtered_Image3)
cv2.imwrite('BoxBlur_lena30.jpg', Filtered_Image4)
cv2.imwrite('BoxBlur_lena_gray.bmp', Filtered_Image5)
cv2.imwrite('BoxBlur_lena_gray(gif).jpg', Filtered_Image6)

# Plugging all of the images into the gaussian filter and reseting the imgcolor value
# also updating radius to be accurate

imgColor = True
# Designating kernel radius for this filter
kernalRadius = 7

Filtered_Image = convolveFunction(inputImage, gaussianBlur_15x15, kernalRadius, imgColor)
Filtered_Image2 = convolveFunction(inputImage2, gaussianBlur_15x15, kernalRadius, imgColor)
Filtered_Image3 = convolveFunction(inputImage3, gaussianBlur_15x15, kernalRadius, imgColor)
Filtered_Image4 = convolveFunction(inputImage4, gaussianBlur_15x15, kernalRadius, imgColor)
Filtered_Image6 = convolveFunction(inputImage6, gaussianBlur_15x15, kernalRadius, imgColor)
# Setting imageColor to false so I can filter the grayscale images
imgColor = False
Filtered_Image5 = convolveFunction(inputImage5, gaussianBlur_15x15, kernalRadius, imgColor)


# Displaying all of the filtered images by the Gaussian Blur filter
cv2.imshow('GaussianBlur_Before:', inputImage4)
cv2.imshow('GaussianBlur_lena5:', Filtered_Image)
cv2.imshow('GaussianBlur_lena10:', Filtered_Image2)
cv2.imshow('GaussianBlur_lena15:', Filtered_Image3)
cv2.imshow('GaussianBlur_lena30:', Filtered_Image4)
cv2.imshow('GaussianBlur_lena_gray:', Filtered_Image5)
cv2.imshow('GaussianBlur_lena_gray(gif)', Filtered_Image6)
cv2.waitKey(0)

# Witing all of the Gaussian Blur filtered functions and saving them into seperate files

cv2.imwrite('GaussianBlur_lena5.jpg', Filtered_Image)
cv2.imwrite('GaussianBlur_lena10.jpg', Filtered_Image2)
cv2.imwrite('GaussianBlur_lena15.jpg', Filtered_Image3)
cv2.imwrite('GaussianBlur_lena30.jpg', Filtered_Image4)
cv2.imwrite('GaussianBlur_lena_gray.bmp', Filtered_Image5)
cv2.imwrite('GaussianBlur_lena_gray(gif).jpg', Filtered_Image6)

imgColor = True
# Designating kernel radius for this filter
kernalRadius = 7

Filtered_Image = convolveFunction(inputImage, MotionBlur_15x15, kernalRadius, imgColor)
Filtered_Image2 = convolveFunction(inputImage2, MotionBlur_15x15, kernalRadius, imgColor)
Filtered_Image3 = convolveFunction(inputImage3, MotionBlur_15x15, kernalRadius, imgColor)
Filtered_Image4 = convolveFunction(inputImage4, MotionBlur_15x15, kernalRadius, imgColor)
Filtered_Image6 = convolveFunction(inputImage6, MotionBlur_15x15, kernalRadius, imgColor)
# Setting imageColor to false so I can filter the grayscale images
imgColor = False
Filtered_Image5 = convolveFunction(inputImage5, MotionBlur_15x15, kernalRadius, imgColor)


# Displaying all of the filtered images by the motion blur filter

cv2.imshow('MotionBlur_Before:', inputImage4)
cv2.imshow('MotionBlur_lena5:', Filtered_Image)
cv2.imshow('MotionBlur_lena10:', Filtered_Image2)
cv2.imshow('MotionBlur_lena15:', Filtered_Image3)
cv2.imshow('MotionBlur_lena30:', Filtered_Image4)
cv2.imshow('MotionBlur_lena_gray:', Filtered_Image5)
cv2.imshow('MotionBlur_lena_gray(gif)', Filtered_Image6)
cv2.waitKey(0)

# Writing all of the Motion Blur filtered functions and saving them into seperate files

cv2.imwrite('MotionBlur_lena5.jpg', Filtered_Image)
cv2.imwrite('MotionBlur_lena10.jpg', Filtered_Image2)
cv2.imwrite('MotionBlur_lena15.jpg', Filtered_Image3)
cv2.imwrite('MotionBlur_lena30.jpg', Filtered_Image4)
cv2.imwrite('MotionBlur_lena_gray.bmp', Filtered_Image5)
cv2.imwrite('MotionBlur_lena_gray(gif).jpg', Filtered_Image6)
cv2.waitKey(0)

