from solution import *


DATA_FRAME = pandas.DataFrame(columns=["Feature", "Unifrom", "Metric", "Result",
                              "Skimage", "Image Size", "R", "L", "Step", "Histogram", "Histogram Tile Size"])

for size in IMAGE_SIZE:
    IMAGES = []
    for subfolder in os.listdir(PATH):
        sf = os.path.join(PATH, subfolder)
        if os.path.isdir(sf):
            for image in os.listdir(sf):
                image_f = os.path.join(sf, image)
                if os.path.isfile(image_f) and "json" not in image_f:
                    img = cv2.imread(
                        image_f, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(size, size).astype(np.float64)
                    IMAGES.append(img)

    pixel_by_pixel_features = [image.flatten()
                               for image in copy.deepcopy(IMAGES)]

    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "euclidean", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]
    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "cityblock", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]
    DATA_FRAME.loc[len(DATA_FRAME.index)] = ["Pixel By Pixel", False, "cosine", accuracy_score(
        pixel_by_pixel_features), None, size, None, None, None, None, None]

    for r, l in zip(R, L):
        skimage_lbp = [feature.local_binary_pattern(
            image, l, r, method="default").flatten() for image in copy.deepcopy(IMAGES)]
        uniform_skimage_lbp = [feature.local_binary_pattern(
            image, l, r, method="uniform").flatten() for image in copy.deepcopy(IMAGES)]

        lbp_footprint = make_lbp_window(r, l)

        lbp_images = [generic_filter(
            image, simple_lbp, footprint=lbp_footprint, mode="nearest").flatten() for image in copy.deepcopy(IMAGES)]
        uniform_lbp_images = [generic_filter(
            image, uniform_lbp, footprint=lbp_footprint, mode="nearest").flatten() for image in copy.deepcopy(IMAGES)]

        for step in STEP:
            lbp_images = lbp_images[::step]
            uniform_lbp_images = uniform_lbp_images[::step]

            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "euclidean", accuracy_score(
                lbp_images), accuracy_score(skimage_lbp), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cityblock", accuracy_score(
                lbp_images, "cityblock"), accuracy_score(skimage_lbp, "cityblock"), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cosine", accuracy_score(
                lbp_images, "cosine"), accuracy_score(skimage_lbp, "cosine"), size, r, l, step, False, None]

            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "euclidean", accuracy_score(
                uniform_lbp_images), accuracy_score(uniform_skimage_lbp), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cityblock", accuracy_score(
                uniform_lbp_images, "cityblock"), accuracy_score(uniform_skimage_lbp, "cityblock"), size, r, l, step, False, None]
            DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", True, "cosine", accuracy_score(
                uniform_lbp_images, "cosine"), accuracy_score(uniform_skimage_lbp, "cosine"), size, r, l, step, False, None]

            for hist_area in HIST_AREA:
                lbp_hist = make_histograms(
                    copy.deepcopy(lbp_images), hist_area, size)
                skimage_lbp_hist = make_histograms(
                    copy.deepcopy(skimage_lbp), hist_area, size)

                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "euclidean", accuracy_score(
                    lbp_hist), accuracy_score(skimage_lbp_hist), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cityblock", accuracy_score(
                    lbp_hist, "cityblock"), accuracy_score(skimage_lbp_hist, "cityblock"), size, r, l, step, True, hist_area]
                DATA_FRAME.loc[len(DATA_FRAME.index)] = ["LBP", False, "cosine", accuracy_score(
                    lbp_hist, "cosine"), accuracy_score(skimage_lbp_hist, "cosine"), size, r, l, step, True, hist_area]

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
