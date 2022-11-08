# Import everything from the function file
from solution import *

# Creating the dataframe to store results
DATA_FRAME = pandas.DataFrame(columns=["Feature", "Unifrom", "Metric", "Result",
                              "Skimage", "Image Size", "R", "L", "Step", "Histogram", "Histogram Tile Size"])

# Loop through image sizes and run the entire processing loop
for size in IMAGE_SIZE:
    IMAGES = []  # Define empty image array
    # Loop through main folder
    for subfolder in os.listdir(PATH):
        # Make and check subfolder path
        sf = os.path.join(PATH, subfolder)
        if os.path.isdir(sf):
            # Loop through subfolder
            for image in os.listdir(sf):
                # Make file path and check if its an image
                image_f = os.path.join(sf, image)
                if os.path.isfile(image_f) and "json" not in image_f:
                    # Read image in greyscale
                    img = cv2.imread(
                        image_f, cv2.IMREAD_GRAYSCALE)
                    # Resize the image and cast it to float to avoid overflow errors
                    img = cv2.resize(size, size).astype(np.float64)
                    IMAGES.append(img)  # Add the image to the list

    # Compute basic pixel by pixel feature vector
    pixel_by_pixel_features = [image.flatten()
                               for image in copy.deepcopy(IMAGES)]

    # Save results of above define feature vector rank-1 classification accuracy for all three metrics
    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "euclidean", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]
    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "cityblock", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]
    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "cosine", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]

    # Loop through radius and word length pairs
    for r, l in zip(R, L):
        # Loop through local region overlap steps
        for step in STEP:
            # Compute the baseline feature vectors using skimage, and remove pixels based on step value
            skimage_lbp = [feature.local_binary_pattern(
                image, l, r, method="default").flatten()[::step] for image in copy.deepcopy(IMAGES)]
            uniform_skimage_lbp = [feature.local_binary_pattern(
                image, l, r, method="uniform").flatten()[::step] for image in copy.deepcopy(IMAGES)]

            # Create the relevant LBP window for computing feature vectors
            lbp_footprint = make_lbp_window(r, l)

            # Using the generic_filter function pass through the images, computing their LBP values and
            #   removing pixels based on step value
            lbp_images = [generic_filter(
                image, simple_lbp, footprint=lbp_footprint, mode="nearest").flatten()[::step] for image in copy.deepcopy(IMAGES)]
            uniform_lbp_images = [generic_filter(
                image, uniform_lbp, footprint=lbp_footprint, mode="nearest").flatten()[::step] for image in copy.deepcopy(IMAGES)]

            # Compute the rank-1 accuracy for both the implemented and skimage created feature vectors for all three metrics
            # First normal LBP
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "euclidean", accuracy_score(
                lbp_images), accuracy_score(skimage_lbp), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cityblock", accuracy_score(
                lbp_images, "cityblock"), accuracy_score(skimage_lbp, "cityblock"), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cosine", accuracy_score(
                lbp_images, "cosine"), accuracy_score(skimage_lbp, "cosine"), size, r, l, step, False, None]

            # Then uniform LBP
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "euclidean", accuracy_score(
                uniform_lbp_images), accuracy_score(uniform_skimage_lbp), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cityblock", accuracy_score(
                uniform_lbp_images, "cityblock"), accuracy_score(uniform_skimage_lbp, "cityblock"), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cosine", accuracy_score(
                uniform_lbp_images, "cosine"), accuracy_score(uniform_skimage_lbp, "cosine"), size, r, l, step, False, None]

            # Loop through the histogram tile size values
            for hist_area in HIST_AREA:
                # Compute histograms
                lbp_hist = make_histograms(
                    copy.deepcopy(lbp_images), hist_area, size)
                skimage_lbp_hist = make_histograms(
                    copy.deepcopy(skimage_lbp), hist_area, size)

                # And save their rank-1 classification accuracy
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "euclidean", accuracy_score(
                    lbp_hist), accuracy_score(skimage_lbp_hist), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cityblock", accuracy_score(
                    lbp_hist, "cityblock"), accuracy_score(skimage_lbp_hist, "cityblock"), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cosine", accuracy_score(
                    lbp_hist, "cosine"), accuracy_score(skimage_lbp_hist, "cosine"), size, r, l, step, True, hist_area]

                # Repeat for uniform LBP
                uniform_lbp_hist = make_histograms(copy.deepcopy(
                    uniform_lbp_images), hist_area, size)
                uniform_skimage_lbp_hist = make_histograms(
                    copy.deepcopy(uniform_skimage_lbp), hist_area, size)

                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "euclidean", accuracy_score(
                    uniform_lbp_hist), accuracy_score(uniform_skimage_lbp_hist), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cityblock", accuracy_score(
                    uniform_lbp_hist, "cityblock"), accuracy_score(uniform_skimage_lbp_hist, "cityblock"), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cosine", accuracy_score(
                    uniform_lbp_hist, "cosine"), accuracy_score(uniform_skimage_lbp_hist, "cosine"), size, r, l, step, True, hist_area]
