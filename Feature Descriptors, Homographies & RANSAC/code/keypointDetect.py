import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]
    #print("Gaussian Pyramid Shape is:",gaussian_pyramid.shape)
    #print("Gaussian Pyramid Shape is:",gaussian_pyramid.shape)
    for i in DoG_levels:
        DoG_pyramid.append(gaussian_pyramid[:,:,i+1]-gaussian_pyramid[:,:,i]) # Make a list of images
    
    DoG_pyramid=np.stack(DoG_pyramid,axis=-1)  # break down the list and make a horizontal concetanation

    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here
    principal_curvature = np.zeros(DoG_pyramid.shape)
    #print("Shape of the principal_curvature",principal_curvature.shape)
    for i in range(DoG_pyramid.shape[2]):
        img     =DoG_pyramid[:,:,i]
        dx      =cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
        dy      =cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
        dx2     =cv2.Sobel(dx,cv2.CV_64F,1,0, ksize=5)
        dy2     =cv2.Sobel(dy,cv2.CV_64F,0,1, ksize=5)
        dxy     =cv2.Sobel(dx,cv2.CV_64F,0,1, ksize=5)
        dyx     =cv2.Sobel(dy,cv2.CV_64F,1,0, ksize=5)

        #print("Shape of the dx",dx.shape)
    # H=[ dx2 dxy 
    #    dyx dy2]
        TraceH=dx2+dy2
        DetH=(dx2*dy2)-(dxy*dxy)
        #print(i)
        #print("Det is",DetH)
        R=(TraceH**2)/DetH
        #print("R is",R.shape)
        principal_curvature[:,:,i]=R

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    locsDoG = np.empty((0, 3))
    for i in range(1, DoG_pyramid.shape[0]-1):
        for j in range(1, DoG_pyramid.shape[1]-1):
            for k in range(DoG_pyramid.shape[2]):
                if abs(DoG_pyramid[i,j,k]) < th_contrast:
                    continue
                if abs(principal_curvature[i,j,k]) > th_r:
                    continue

                a = [-1, 0, 1]
                b = [-1, 1]
                if k == 0:
                    b = [1]
                if k == DoG_pyramid.shape[2] - 1:
                    b = [-1]

                neighbours = []
                for offset_x in a:
                    for offset_y in a:
                        neighbours.append(DoG_pyramid[i+offset_x, j+offset_y, k])
                for offset_z in b:
                    neighbours.append(DoG_pyramid[i, j, k+offset_z])

                neighbours = np.array(neighbours)
                if np.argmin(neighbours) == 4 or np.argmax(neighbours) == 4: # original pixel
                    #if abs(DoG_pyramid[i,j,k]) > th_contrast:
                        #print(principal_curvature[i,j,k])
                        #if abs(principal_curvature[i,j,k]) < th_r:
                    index = np.array([j,i,k]).reshape(1,3)
                    locsDoG = np.append(locsDoG, index, axis=0)

    return locsDoG.astype(int)


    #return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gaussian_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gaussian_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)


    return locsDoG, gaussian_pyramid





if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)   uncomment before submission
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)  #uncomment before submission
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    #print("WE ARE HERE")
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    for i in range(locsDoG.shape[0]):
        cv2.circle(im, (locsDoG[i,0], locsDoG[i,1]), 1, color=(0,255,0), lineType=cv2.LINE_8)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    im = cv2.resize(im, (im.shape[1]*5, im.shape[0]*5))
    cv2.imwrite('../results/keypoints.png', im)
    cv2.imshow('image', im)
    cv2.waitKey(0) # press any key to exit


