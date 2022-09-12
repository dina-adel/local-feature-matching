import time
import cv2
import numpy as np
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_interest_points(image, feature_width):
    """
    Returns interest points for the input image
    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_features() and match_features())
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:
    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image
    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    """
    s = time.time()

    sigma = 0.2
    step = 2
    # threshold = 0.005  # for notre_dame
    # threshold = 0.001  # for mt_rushmore
    threshold = 0.0005  # for e_gaudi

    xs = []
    ys = []

    img_h = image.shape[0]
    img_w = image.shape[1]

    # First, compute the x and y derivatives using a Sobel filter

    dx = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    dy = cv2.Sobel(image, cv2.CV_8U, 0, 1, ksize=5)

    # blurring
    Ix = gaussian(dx, sigma=sigma)
    Iy = gaussian(dy, sigma=sigma)

    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy

    # Third,feature window
    window = feature_width // 2

    # looping windows to find the corners in the image
    for y in range(window, img_h - window, step):
        for x in range(window, img_w - window, step):

            Sxx = np.sum(Ixx[y - window:y + window + 1, x - window:x + window + 1])
            Syy = np.sum(Iyy[y - window:y + window + 1, x - window:x + window + 1])
            Sxy = np.sum(Ixy[y - window:y + window + 1, x - window:x + window + 1])

            # Fourth, compute the Harris Response
            # R = detM - k*traceM**2 or R = detM/traceM
            detM = (Sxx * Syy) - (Sxy ** 2)
            traceM = Sxx + Syy

            R = detM - (0.06 * traceM ** 2)  # 0.06 based on search and recommendations

            if R > threshold:
                xs.append(x-1)
                ys.append(y-1)

    xs = np.array(xs)
    ys = np.array(ys)
    print('getting interest points took => ', time.time()-s)
    return xs, ys


def get_features(image, x, y, feature_width, sigma=0.8):
    """"
    Returns feature descriptors for a given set of interest points.
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length
    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.
    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.
    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.
    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions
        - skimage.filters (library)
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.
    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    """

    s = time.time()

    n_interest_points = x.shape[0]  # number of interest points
    features = np.zeros((len(x), 4, 4, 8))  # features array: len_features x win_4 x win_4 x num_bins_per_window

    filtered_img = gaussian(image, sigma=sigma)  # blur image

    # getting gradient magnitude and orientation for each pixel
    gx, gy = np.gradient(filtered_img)
    magnitudes = np.sqrt(gx ** 2 + gy ** 2)

    angles = np.arctan2(gy, gx)
    angles[angles < 0] += 2 * np.pi

    # loop over interest points window (16x16)
    for n in range(n_interest_points):
        angles_window = angles[y[n] - feature_width//2: y[n] + feature_width//2, x[n] - feature_width//2:x[n] + feature_width//2]
        grad_window = magnitudes[y[n] - feature_width//2: y[n] + feature_width//2, x[n] - feature_width//2:x[n] + feature_width//2]

        # loop over sub-windows (4x4)
        for i in range(feature_width//4):
            for j in range(feature_width//4):
                curr_gradient = grad_window[i * (feature_width//4):(i + 1) *(feature_width//4),
                                            j * (feature_width//4):(j + 1) * (feature_width//4)].flatten()
                curr_angles = angles_window[i * (feature_width//4):(i + 1) *(feature_width//4),
                                            j * (feature_width//4):(j + 1) * (feature_width//4)].flatten()

                # histogram with 8 bins and extract SIFT-like features
                features[n, i, j] = np.histogram(curr_angles, bins=8,
                                                 range=(0, 2 * np.pi), weights=curr_gradient)[0]

    features = features.reshape((len(x), -1,))  # num_features x 128

    # normalize features
    features_norm = np.linalg.norm(features)
    if features_norm != 0:
        features = features / features_norm

    features = features**0.8

    print('getting features took => ', time.time()-s)

    return features


def match_features(im1_features, im2_features):
    """
     Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.
    For extra credit you can implement spatial verification of matches.
    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.
    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).
    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2
    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    """

    s = time.time()

    # initialize variables
    matches = []
    confidences = []
    threshold = 0.9

    """ Trying PCA: Performance is faster, but worse
    scaler = StandardScaler()
    scaler.fit(im1_features)
    im1_features = scaler.transform(im1_features)
    im2_features = scaler.transform(im2_features)

    pca = PCA(n_components=64)
    pca.fit(im1_features)
    im1_features = pca.transform(im1_features)
    im2_features = pca.transform(im2_features)
    """

    # Calculate euclidean distance between each feature in 1st image and all other features in 2nd image
    dist = cdist(im1_features, im2_features, metric='euclidean')

    # loop over im1 features
    for i in range(im1_features.shape[0]):
        # sort the distances in ascending order, and get the sorted indices
        sorted_index = np.argsort(dist[i, :]).astype(int)

        if dist[i, sorted_index[0]] < (threshold*dist[i, sorted_index[1]]):
            # if the ratio between closest two features is < 0.9 -> append as a good match
            matches.append([i, sorted_index[0]])
            confidences.append(1.0 - dist[i, sorted_index[0]] / dist[i, sorted_index[1]])

    confidences = np.asarray(confidences)
    matches = np.asarray(matches)

    print('getting matches took => ', time.time()-s)

    return matches, confidences
