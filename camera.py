import numpy as np
import cv2

# Define a Camera class with methods for calibration
# and undistorting images
class Camera():
    
    Matrix = None
    Distortion = None
    ImageSize = None
    
    def __init__(self, mtx = None, dist = None):
        self.Matrix = mtx
        self.Distortion = dist
    
    # Define corner detection
    def find_corners(self, img, dimensions):
        # Define translation and distortion parameters
        objpoints = []
        imgpoints = []

        # Initialise chessboard point space parameter
        objp = np.zeros((dimensions[1] * dimensions[0], 3), np.float32)
        objp[:,:2] = np.mgrid[0:dimensions[0], 0:dimensions[1]].T.reshape(-1, 2) # x,y coordinates

        # convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # find corners usng grayscale image
        [ret, corners] = cv2.findChessboardCorners(gray, dimensions, None)

        return [ret, objp, corners]
    
    # Camera calibration function
    def calibrate(self, images, dimensions):
        # Define real-world and image space coordinate arrays
        objpoints = []
        imgpoints = []

        self.ImageSize = (images[0].shape[1], images[0].shape[0])

        for img in images:
            # detect corners in the chessboard calibration image
            [ret, points, corners] = self.find_corners(img, dimensions)

            if (ret == True):
                # add the real-world points along with image space points to the arrays
                objpoints.append(points)
                imgpoints.append(corners)
        
        # calibrate camera using point spaces
        [ret, mtx, dist, rvecs, tvecs] = cv2.calibrateCamera(objpoints, imgpoints, self.ImageSize, None, None)    
        
        # store calibration
        self.Matrix = mtx
        self.Distortion = dist
        
        return [self.Matrix, self.Distortion]
    
    ### Define the warping functions ###
    def transform(self, src, dst):
        if (src is not None and dst is not None):
            self.M = cv2.getPerspectiveTransform(src, dst)
        if (src is not None and dst is not None):
            self.Minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp an image using the specified source and destination coordinate mappings.
    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.ImageSize, flags=cv2.INTER_LINEAR)

    # Unwarps an image using the specified source and destination coordinate mappings.
    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.ImageSize, flags=cv2.INTER_LINEAR)
    
    # Image undistortion using translation coefficients
    def undistort(self, img):
        return cv2.undistort(img, self.Matrix, self.Distortion, None, self.Matrix)