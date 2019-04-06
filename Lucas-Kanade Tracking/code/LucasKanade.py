import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
    
    dx=1
    dy=1
      
    #print(It_p.shape)
    xx=np.arange(0,It.shape[1]) # array for pixel numbers in x direction
    yy=np.arange(0,It.shape[0]) # array for pixel numbers in y direction
    exInterp1=RectBivariateSpline(yy,xx,It) #interpolation of It
    
    #Same thing for the second template
    xx=np.arange(0,It1.shape[1])
    yy=np.arange(0,It1.shape[0])
    exInterp2=RectBivariateSpline(yy,xx,It1) 


    x_1=rect[0]  #left top corner
    y_1=rect[1]  
    x_2=rect[2]  #bottom right corner
    y_2=rect[3]   


    # Calculate the gradient across whole image
    Itx,Ity=np.gradient(It1)


    #Interpolate the gradient in the window 
    exInterpdx=RectBivariateSpline(yy,xx,Itx)
    exInterpdy=RectBivariateSpline(yy,xx,Ity)
    

    # Define the window
    xx1=np.arange(min(x_1,x_2), max(x_1,x_2))
    yy1=np.arange(min(y_1,y_2), max(y_1,y_2))
    #exInterp2=RectBivariateSpline(y,x,It)
    #yv, xv = np.meshgrid(yy1, xx1, sparse=False, indexing='ij')
    
    # Interpolate the template for the window
    T=exInterp1(yy1,xx1)
    
    tol = 0.01;
    deltaP=np.ones(2) # initialization of deltap
    counter=0 
    while (np.linalg.norm(deltaP)>tol):
        #1 Warp image at t+1
        xx2=np.arange(min(x_1,x_2), max(x_1,x_2)) 
        yy2=np.arange(min(y_1,y_2), max(y_1,y_2))
        
        # initial guess for p
        xx2=xx2+p0[1]
        yy2=yy2+p0[0]


        yv, xv = np.meshgrid(yy2, xx2, sparse=False, indexing='ij')
        I_Warped=exInterp2(yy2,xx2)  #warp second image to calculate new position of the window
        #It1_p=It1[int((y_1)):int((y_2))+1,int((x_1)):int((x_2))+1]
        #I_Warped=exInterp2.ev(yy+p0[0],xx+p0[1])
        #print("THIS IS WARPED")
        #print(len(x))
        #print(len(y))
        #print(It1_p.shape)
        #x=np.linspace(min(x_1,x_2), max(x_1,x_2), It1_p.shape[1])
        #y=np.linspace(min(y_1,y_2), max(y_1,y_2), It1_p.shape[0])
        #exInterp2=RectBivariateSpline(y,x,It1_p)
        #plt.imshow(I_Warped)
        #plt.figure(2)
        #plt.title("I_Warped")
        
        #2 Compute error image
        b = (T - I_Warped);
        #3. Compute the gradient
        #print("Ity shape is",Ity.shape)
        #print("Itx shape is",Itx.shape)
        #Itx1d=exInterp2.ev(yv,xv,dx=1).flatten()
        #Ity1d=exInterp2.ev(yv,xv,dy=1).flatten()
        Itx1d = exInterpdx(yy2,xx2).flatten()
        Ity1d = exInterpdy(yy2,xx2).flatten()
        #4. Evaluate Jacobian
        deltaI=np.column_stack((Itx1d,Ity1d))
        #5. Compute Hessian
        A=deltaI@np.eye(2)
        
        H= A.transpose()@A;

        deltaP=np.linalg.lstsq(A, np.ravel(b),rcond=-1)[0]
        #deltaP = np.linalg.inv(H) @ A.transpose() @ np.ravel(b)
        #print("deltaP")
        #print("deltaP",deltaP)
        #print("deltaP")
        #print("p0",p0)
        
        #9. Update parameters
        p0=p0+deltaP 
        counter+=1
        #print(deltaP[0]**2+deltaP[1]**2)
        if counter>10000:
            print("counter is done",counter)
            break
    
    u=p0[0]
    v=p0[1]
    p=p0
    return p
