import numpy as np
import cv2

### Lane Finding functions ###

# Masks using colour thresholding
# - thresholds: 2D array of threshold values for each colour channel
def mask_thresholds(img, thresholds = [(220,255), (0,255), (0,90)], mask = 1):
    # init empty binary image map
    binary_result = np.zeros((img.shape[0], img.shape[1]))
    # apply thresholds for each channel
    for t in range(len(thresholds)):
        channel = img[:,:,t]
        binary_result[((binary_result == mask) \
                       | ((channel >= thresholds[t][0]) & (channel <= thresholds[t][1])))] = mask
    
    return binary_result

# Computes the raw sobel values for each direction (x and y) for a given image
# - img: Input image in RGB format
# - kernel: Scalar value >= 3, for the kernel size
def compute_sobel(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # compute sobel derivatives
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel)
    
    return [sobelx, sobely]

# Computes the sobel threshold binary map for a given image
# - img: Input image in RGB format
# - kernel: Scalar value >= 3, for the kernel size
# - thresh_x: 2D Tuple of threshold values for the x coordinate
# - thresh_y: 2D Tuple of threshold values for the y coordinate
def sobel_threshold(img, kernel = 3, thresh_x = (20, 100), thresh_y = (0, 255), mask = 1):
    # compute sobel
    [sobelx, sobely] = compute_sobel(img, kernel)
    
    # take the max values
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # scale to 0-255 range
    scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    
    # apply threshold values
    binary_output = np.zeros_like(abs_sobelx)
    if (thresh_x is not None):
        binary_output[((scaled_sobelx >= thresh_x[0]) & (scaled_sobelx <= thresh_x[1]))] = mask
    
    if (thresh_y is not None and thresh_x is not None):
        binary_output[(((scaled_sobely >= thresh_y[0]) & (scaled_sobely <= thresh_y[1]))\
                      & (binary_output == mask))] = mask
    elif (thresh_y is not None):
        binary_output[((scaled_sobely >= thresh_y[0]) & (scaled_sobely <= thresh_y[1]))] = mask
    
    return binary_output

# Computes the sobel magnitude binary map for a given image
# - img: Input image in RGB format
# - kernel: Scalar value >= 3, for the kernel size
# - thresh: 2D Tuple of threshold values
def sobel_magnitude(img, kernel = 3., thresh = (20, 100), mask = 1):
    # compute sobel
    [sobelx, sobely] = compute_sobel(img, kernel)
    
    # take the max values
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # compute the magnitude
    mag_sobel = np.sqrt(abs_sobelx**2. + abs_sobely**2.)
    # scale within 0-255 range
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # apply threshold values
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = mask
    
    return binary_output

# Computes the sobel direction gradient for a given image
# - img: Input image in RGB format
# - kernel: Scalar value >= 3, for the kernel size
# - thresh: 2D Tuple of threshold values
def sobel_direction(img, kernel=3, thresh = (0, np.pi/2), mask = 1):
    # compute sobel
    [sobelx, sobely] = compute_sobel(img, kernel)
    # take the max values
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # compute directional gradients
    grad = np.arctan2(abs_sobely, abs_sobelx)
    # create mask
    binary_output = np.zeros_like(grad)
    binary_output[(grad >= thresh[0]) & (grad <= thresh[1])] = mask
    
    return binary_output

# Applies a combination of CV techniques to an input image for masking lane lines 
# in an image.
# - img: Input image in RGB format
# - kernels: 3D tuple of kernel values for computing absolute, magnitude and directional sobel coordinates
# - abs_thresh: 2D tuple of min and max threshold values for the absolute Sobel value mask
# - grad_thresh: 2D tuple of min and max threshold values for the magnitude Sobel value mask
# - dir_thresh: 2D tuple of min and max threshold values for the directional Sobel value mask
# Returns: Binary mask where the mask value is a likely candidate lane line.
def lane_line_mask(img, kernels, abs_thresh, mag_thresh, dir_thresh, mask = 1):
    # compute sobel threshold, magnitude and directional gradients
    sobel_abs = sobel_threshold(img, kernel=kernels[0], thresh_x=abs_thresh[0], thresh_y=abs_thresh[1], mask=mask)
    sobel_mag = sobel_magnitude(img, kernel=kernels[1], thresh=mag_thresh, mask=mask)
    sobel_dir = sobel_direction(img, kernel=kernels[2], thresh=dir_thresh, mask=mask)
    # combine the sobel masks
    combined = np.zeros_like(sobel_dir)
    combined[((sobel_abs == mask) | ((sobel_mag == mask) & (sobel_dir == mask)))] = mask
    # convert to YUV colour space
    yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # threshold the Y,U,V colour channels (ignore U channel)
    yuv_mask = mask_thresholds(yuv_img, [(220,255), (0,0), (0,90)], mask)
    # combine the detected lines from sobel and colour thresholding
    result = np.zeros_like(combined)
    result[((combined == mask) | (yuv_mask == mask))] = mask
    
    return result

# Normalises the input to the value of 0-1.
# - Returns: array of size X
def normalise(X):
    return X / np.sum(X, axis=0)

# Finds the nearest element to the specified value
# - Returns: index of nearest element
def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

# Returns the max value of a given signal
# - Returns: Tuple of max index and corresponding value
def compute_signal(signal, min_range, max_range, offset, threshold):
    # normalise ranges
    min_range_i = max(min(min_range, max_range), 0)
    max_range_i = min(max(max_range, min_range), len(signal))
    
    if (max_range_i - min_range_i <= 0):
        max_range_i = min(max_range_i + offset, len(signal))
    
    centerv = np.argmax(signal[min_range_i : max_range_i]) + min_range_i - offset
    maxv = signal[centerv + offset]
    
    if (maxv <= threshold):
        centerv = int(np.median([min_range_i, max_range_i])) - offset
        maxv = signal[centerv]
        
    return [centerv, maxv]

# Returns a weight array based on the similarity of the supplied value arrays
# - weights: weight array of same length as vals
# - vals: value array of same length as prev_vals
# - prev_vals: previous value array
def similarity_weights(weights, vals, prev_vals, eps = 1e-08):
    dx = np.sqrt(((np.array(vals) + eps) - np.array(prev_vals)) ** 2.)
    dxv = 1. - (dx / np.sum(dx))
    return weights * ((dxv ** 2) / np.sum(dxv ** 2))

# Fits a polynomial function to the inputs and returns the predicted value
# - X: Training points
# - y: Labels
# - theta: Polynomial term
# - x : Sample to predict
def predict_sample(X, y, theta, x):
    params = np.polyfit(X, y, theta)
    return np.polyval(params, x)

# Fits valid lane pixel windows which correlate to prospective lane lines in an image.
# - image: Input image (single channel)
# - window_width: Width of the sliding window
# - window_height: Height of the sliding window
# - margin: Sliding window margin
# - threshold: Value threshold for max convolution
# - zoom: Zoom height margin for computing the greater window signal
# - gain: memory weight for lane heading
# - scan: memory length for ranging lane distance values
# - memory: number of timesteps for computing lane heading
# - theta: degrees of freedom of the lane heading
# - gamma: thresholding value to keep discount factors within limits
# - eps: epsilon parameter to avoid divide by zero errors
def find_window_centroids(img, window_width, window_height, margin, threshold = 0.0001, zoom = 2, \
                          gain = 0.8, scan = 3, memory = 7, theta = 2, gamma = 1.5, start_height = 0.8, \
                          predictive_search = True, eps = 1e-08):
    # Create our window template that we will use for convolutions
    window = np.ones(window_width) 
    
    height = img.shape[0]
    width = img.shape[1]
    
    # Use window_width/2 as offset because convolution signal reference is 
    # at right side of window, not center of window
    offset = int(window_width * 0.5)
    
    # Convolved signal of the lower vertical region
    hist = np.sum(img[int(height * start_height) :, :], axis=0)
    window_signal = normalise(np.convolve(window, hist))
    
    # Find Left lane signal
    [l_center, l_max] = compute_signal(window_signal, 0, int(width/2), offset, threshold)
    l_dist = l_pred_center = 0
    l_center_prev = lz_center_prev = l_center
    
    # Find Right lane signal
    [r_center, r_max] = compute_signal(window_signal, int(width/2), len(window_signal), offset, threshold)
    r_dist = r_pred_center = 0
    r_center_prev = rz_center_prev = r_center
    
    # Initialise moving average memory and add what we found for the first layer
    window_memory = np.empty((0,8)) # levels x [l_center, l_dist, l_max, l_disc, r_center, r_dist, r_disc, r_max]
    window_memory = np.vstack((window_memory, [l_center, l_dist, l_center, l_max, \
                                               r_center, r_dist, r_center, r_max]))
    # Initialise value / weights
    l_max_prev = lz_max = l_max
    r_max_prev = rz_max = r_max
    l_weights = np.array([0.5, 0.5])
    r_weights = np.array([0.5, 0.5])
    
    pred_dist = (r_center - l_center)
    
    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(height / window_height)):
        # convolve the window into the vertical slice of the image
        min_range = int(height - (level + 1) * window_height)
        max_range = int(height - (level) * window_height)
        # Compute the convolved signal from the current window
        image_layer = np.sum(img[min_range : max_range, :], axis=0)
        window_signal = normalise(np.convolve(window, image_layer))
        
        # Find the zoom range window
        min_zoom_range = int(max(height - (level + zoom) * window_height, 0))
        max_zoom_range = int(min(height - max((level - zoom), 0) * window_height, height))
        # Compute the convolved signal from the current zoom level
        region_layer = np.sum(img[min_zoom_range : max_zoom_range, :], axis=0)
        zoom_signal = normalise(np.convolve(window, region_layer))
        
        if (level > 1 and predictive_search):
        # Find the best left centroid by using past left center as a reference
            y_vals = np.arange(level - memory, level)[-len(window_memory[-memory:]):]
            # find left heading
            l_pred_center = predict_sample(y_vals, window_memory[-memory:,0], theta, level)
            # find right heading
            r_pred_center = predict_sample(y_vals, window_memory[-memory:,4], theta, level)
        else:
            l_pred_center = l_center_prev + min(max(l_dist, -margin), margin)
            r_pred_center = r_center_prev + min(max(r_dist, -margin), margin)
        
        # Upper and lower bounded discount factors
        l_pred_center = int(max(l_pred_center, 0))
        r_pred_center = int(min(r_pred_center, width))
        
        ### LEFT ###
        # Compute window and zoom signals for left
        l_min_index = int(max(l_pred_center + offset - margin, 0))
        l_max_index = int(min(l_pred_center + offset + margin, r_center_prev - offset))
        # Compute the left window signal
        [l_center, l_max] = compute_signal(window_signal, l_min_index, l_max_index, offset, threshold)
        # Compute the left zoom region signal
        [lz_center, lz_max] = compute_signal(zoom_signal, l_min_index, l_max_index, offset, threshold)
        # Smooth left detectors
        l_weights = similarity_weights(l_weights + eps, [l_center, lz_center], [l_center_prev, lz_center_prev])
        l_center_mu = np.average([l_center, lz_center], weights = l_weights)
        l_max_new = l_weights.mean()
        
        ### RIGHT ###
        # Compute window and zoom signals for right
        r_min_index = int(max(r_pred_center + offset - margin, l_center_prev + offset))
        r_max_index = int(min(r_pred_center + offset + margin, width))
        # Compute the right window signal
        [r_center, r_max] = compute_signal(window_signal, r_min_index, r_max_index, offset, threshold)
        # Compute the right zoom region signal
        [rz_center, rz_max] = compute_signal(zoom_signal, r_min_index, r_max_index, offset, threshold)
        # Smooth right detectors
        r_weights = similarity_weights(r_weights + eps, [r_center, rz_center], [r_center_prev, rz_center_prev])
        r_center_mu = np.average([r_center, rz_center], weights = r_weights)
        r_max_new = r_weights.mean()
        
        # Averaged bilateral shift to avoid lane divergence / convergence errors
        if (level >= scan):
            y_vals_dist = np.arange(level - scan, level)[-len(window_memory[-scan:]):]
            mean_dist = predict_sample(y_vals_dist, (window_memory[-scan:,4] - window_memory[-scan:,0]), 1, level)
            if (l_max > r_max):
                r_center = np.average([l_pred_center + mean_dist, r_center_mu], weights = [gain, 1.0-gain])
            else:
                l_center = np.average([r_pred_center - mean_dist, l_center_mu], weights = [gain, 1.0-gain])
        
        # Remember lane headings
        l_dist = l_center - l_center_prev
        r_dist = r_center - r_center_prev
        
        # Update max t-1
        l_max_prev = l_max
        r_max_prev = r_max
        l_center_prev = int(l_center)
        r_center_prev = int(r_center)
        lz_center_prev = int(lz_center)
        rz_center_prev = int(rz_center)
        # Add what we found for that layer
        window_memory = np.vstack((window_memory, [l_center, l_dist, l_pred_center, l_max_new, \
                                                   r_center, r_dist, r_pred_center, r_max_new]))
    
    return window_memory

# Smoothes a path by gradient descent
# path: a 2D array of path points
def smooth_path(path, beta = 0.5, alpha = 0.1, tolerance = 1e-08, max_iter = 1000):
    # Make a deep copy of path
    output = np.copy(path)
    # set diff to tolerance
    diff = tolerance
    count = 0
    while (diff >= tolerance and count <= max_iter):
        # reset difference
        diff = 0
        count += 1
        for i in range(1, len(path)-1):
            for j in range(len(path[i])):
                aux = output[i][j]
                # compute gradients
                path_diff = (path[i][j] - output[i][j])
                path_err = (output[i-1][j] + output[i+1][j] - 2.0 * output[i][j])
                output[i][j] += (beta * path_diff) + (alpha * path_err)
                # accumulate loss
                diff += abs(aux - output[i][j])
    # return smoothed path
    return output

### Window Drawing functions ###

# Applies a window mask on the current image, centroid pair at the specified level
# Returns: mask of the current sliding window
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    if not np.isnan(center):
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),\
               max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

# Draws the centroids returned by find_window_centroids
# Returns: a new image with each window overlayed on the original image
def draw_window_centroids(img, centroids, window_width, window_height, color='red'):
    # If we found any window centers
    if len(centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows 	
        for level in range(0,len(centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        if (color=='red'):
            template = np.array(cv2.merge((template, zero_channel, zero_channel)), np.uint8)
        else:
            template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)
        warpage = np.array(cv2.merge((img, img, img)), np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((img, img, img)),np.uint8)
    
    return output