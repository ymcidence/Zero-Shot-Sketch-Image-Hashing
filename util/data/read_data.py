from skimage.io import imread


# from skimage.transform import resize


def read_image(file_name):
    this_image = imread(file_name)
    # if this_image.ndim == 2:
    #     this_image = np.tile(this_image[..., np.newaxis], (1, 1, 3))
    # out_img = resize(this_image, (227, 227), mode='reflect') - mean
    # out_img[:, :, 0], out_img[:, :, 2] = out_img[:, :, 2], out_img[:, :, 0]
    return this_image
