import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...


    N = p1.shape[1] # number of pixels correspondances
    A = np.zeros((2*N, 9)) 

    for i in range(N):
    
        A[2*i] = [0, 0, 0, -p2[0, i], -p2[1, i], -1, p1[1, i]*p2[0, i], p1[1, i]*p2[1, i], p1[1, i]]
        A[2*i+1] = [p2[0, i], p2[1, i], 1, 0, 0, 0, -p1[0, i]*p2[0, i], -p1[0, i]*p2[1, i], -p1[0, i]]


    (U, S, V) = np.linalg.svd(A)

    R = V[-1,:] / V[-1,-1]

    H2to1 = R.reshape(3,3)

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...

    n = matches.shape[0] #number of matches
    maxNumInliers = -1 
    bestH = np.zeros((3,3)) #initialization of H matrix

    X = np.zeros((n, 2)) 
    U = np.zeros((n, 2))
    X[:, :] = locs1[matches[:, 0], 0:2]
    U[:, :] = locs2[matches[:, 1], 0:2]

    p1 = np.zeros((2, 4))
    p2 = np.zeros((2, 4))

    for iter in range(num_iter):
        indexs = np.random.choice(n, 4, replace=False)
        for i in range(indexs.shape[0]):
            x = locs1[matches[indexs[i], 0], 0:2]
            u = locs2[matches[indexs[i], 1], 0:2]
            p1[:, i] = x
            p2[:, i] = u

        H = computeH(p1, p2)
        X_h = np.append(np.transpose(X), np.ones((1, n)), axis=0) 
        U_h = np.append(np.transpose(U), np.ones((1, n)), axis=0) 
        reproj = np.matmul(H, U_h)
        reproj_norm = np.divide(reproj, reproj[2, :])

        error = X_h - reproj_norm
        num_inliers = 0
        for i in range(n):
            squared_dist = error[0, i]**2 + error[1, i]**2
            if squared_dist <= tol**2:
                num_inliers += 1

        if num_inliers > maxNumInliers:
            bestH = H
            maxNumInliers = num_inliers
#    print("most inliers found by RANSAC: ", maxNumInliers)

    return bestH
        
    

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

