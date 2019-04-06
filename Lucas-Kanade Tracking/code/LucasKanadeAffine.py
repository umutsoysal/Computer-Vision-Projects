import numpy as np
import cv2
import matplotlib.pyplot as plt

def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
	# put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	p = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]])

	w = It.shape[1]
	h = It.shape[0]

	xv, yv = np.meshgrid(np.arange(h), np.arange(w))
	xv = np.reshape(xv, (1, xv.shape[0]*xv.shape[1]))
	yv = np.reshape(yv, (1, yv.shape[0]*yv.shape[1]))

	Iy, Ix = np.gradient(It)

	mask = np.ones((h, w)).astype(np.float32)
	A = np.zeros((w*h, 6)).astype(np.float32)
	b = np.zeros((w*h)).astype(np.float32)

	threshold = 0.01
	dp_norm = threshold

	while dp_norm >= threshold:

		warped_img   = cv2.warpAffine(It, M, (w, h))
		warped_gradX = cv2.warpAffine(Ix, M, (w, h)).reshape((1, w*h))
		warped_gradY = cv2.warpAffine(Iy, M, (w, h)).reshape((1, w*h))
		warped_mask  = cv2.warpAffine(mask, M, (w, h))
		masked_img = warped_mask * It1

		# Construct A and b matrices
		
		A[:, 0] = warped_gradX[:] * xv[:]
		A[:, 1] = warped_gradX[:] * yv[:]
		A[:, 2] = warped_gradX[:]
		A[:, 3] = warped_gradY[:] * xv[:]
		A[:, 4] = warped_gradY[:] * yv[:]
		A[:, 5] = warped_gradY[:]

		b = (warped_img - masked_img).reshape((w*h, 1))

		deltap= np.linalg.lstsq(A, b,rcond=-1)[0]
		dp_norm = np.linalg.norm(deltap)
		p = p+ deltap
		M[0][0] = 1.0 + p[0]
		M[0][1] = p[1]
		M[0][2] = p[2]
		M[1][0] = p[3]
		M[1][1] = 1.0 + p[4]
		M[1][2] = p[5]
	return M
