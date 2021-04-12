import pydicom as dicom
import skimage.exposure as exp
import skimage.filters as filt
import skimage.transform as trans
import PIL.Image
import numpy as np
import os


def png_loader(path) -> np.ndarray:
    '''
    Loads a png image and returns a numpy array of the pixel data.
    :param path: path of the image
    :return: numpy array of the pixel data
    '''
    img = PIL.Image.open(path)

    x = np.array(img)

    return x

def pil_loader(path) -> PIL.Image:
    '''
    Loads an image as a pil image - required for torch transforms .
    :param path: path of the image
    :return: pil image
    '''
    img = PIL.Image.open(path)
    
    return img


def dicom_loader(path, auto_load_png=True, raw=True) -> np.ndarray:
    '''
    Loads the dicom and does some simple pre-processing, returning a numpy array. Pre-processing is inverting MONOCHROME1
    images so the bones are all white, and clipping any boarders of block colours. If the dicom file is actually a png
    file, just return the png data.
    :param path: path of dicom file.
    :param auto_load_png: if True, if a png file is provided, it will load it without prodcing an error.
    :param raw: if true returns the pixels as stored in the dicom, if false, checks
    :return:numpy array
    '''

    if auto_load_png:
        extension = os.path.splitext(path)[1]
        # if its a png image then just return the image
        if extension.lower() == '.png':
            return png_loader(path)

    dc = dicom.dcmread(path)
    pixeldata = None

    if hasattr(dc, 'PhotometricInterpretation'):            # could the image be inverted???
        if dc.PhotometricInterpretation == 'MONOCHROME1':
            # need to sort out re inverting the image...
            if dc.PixelRepresentation == 0:
                # unsigned integers
                max = 2 ** dc.BitsStored - 1
                pixeldata = max-dc.pixel_array
            else:
                # 2s complement
                pixeldata = dc.pixel_array * -1
        else:
            pixeldata = dc.pixel_array
    else:
        pixeldata = dc.pixel_array

    # pixeldata now contains the image, in classical xray form - white bones

    # now to clip any black frames or txt that is around the image
    min, max = np.min(pixeldata), np.max(pixeldata)

    mask = ((pixeldata > min) & (pixeldata < max))
    mask = filt.gaussian(mask, sigma=10) > 0.9  # blurs out any antialiasing around labels.

    mask0, mask1 = mask.any(0), mask.any(1)
    m, n = pixeldata.shape

    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax() - 1
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax() - 1

    if raw:
        return pixeldata[row_start:row_end, col_start:col_end].astype('int16')
    else:
        # correct window width and centre
        vals = pixeldata[row_start:row_end, col_start:col_end] #.astype('int16')

        # try:

        WC = dc['00281050'].value
        WW = dc['00281051'].value
        Intercept = dc['00281052'].value
        Slope = dc['00281053'].value

        vals = vals * Slope + Intercept

        lowestVisibleValue = (WC - 0.5 - ((WW - 1) / 2))
        highestVisibleValue = (WC - 0.5 + ((WW - 1) / 2))

        lessthan = vals<=lowestVisibleValue
        morethan = vals>=highestVisibleValue

        vals = ((vals - (WC - 0.5)) / ((WW - 1) + 0.5)*255)+127.5


        vals[lessthan] = 0
        vals[morethan] = 255
        
        return vals.astype('uint8')

        # except:
        #     print('No WW,WC,Intercept or Slope and the dicom loader function was requesting windowed pixel data.')
def get_alpha(np_image) -> np.ndarray:
    '''
    returns the alpha channel or greysclae channel of an image
    :param np_image: an array of an image

    :return: image in the format (x,y)
    '''
    if len(np_image.shape) == 2:
        return np_image
    else:
        return np_image[::, ::, -1]    # return the last channel which is alpha no matter if its a greyscale or RGB
def segmentation_image_loader(filename) -> np.ndarray:
    '''
    loads the sementation images created in GIMP (normally saved as [X,Y,2] = 2 channels = grey and alpha and returns [X,Y] with only
    the pixel data for the alpha channel - ie. it does not matter what colour

    :param filename:
    :return:
    '''

    img = PIL.Image.open(filename)

    x = np.array(img)
    return get_alpha(x)

def resize_image(input, new_size):
    out = trans.resize(input, new_size, preserve_range=True, anti_aliasing=True)  # for adapt hist - size must be divisible by 2
    return out

def to_basic(input, destination_file_name=None, dicom_loader_function=None, image_size=1024):
    '''
    applies a basic resize, uses dicoms original window centre and width.
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array.
    '''

    if type(input) is np.ndarray:
        dc_pixels = input
    else:
        dc_pixels = dicom_loader_function(input, raw=False)

    out = trans.resize(dc_pixels, (image_size, image_size), preserve_range=True, anti_aliasing=True)  # for adapt hist - size must be divisible by 2


    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'L').save(destination_file_name)
        return True


def to_adaptive_hist(input, destination_file_name=None, dicom_loader_function=None, image_size=512):
    '''
    applies an adaptive_histogram function to an input image.
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array.
    '''

    if type(input) is np.ndarray:
        dc_pixels = input
    else:
        dc_pixels = dicom_loader_function(input)

    resized = trans.resize(dc_pixels, (image_size, image_size), anti_aliasing=True)  # for adapt hist - size must be divisible by 2

    out = exp.equalize_adapthist(resized)
    out = (out * 255).astype(np.uint8)

    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'L').save(destination_file_name)
        return True


def to_sato_1(input, destination_file_name=None, dicom_loader_function=None, image_size=512):
    '''
    applies an sato function to an input image. (LineSafe preprocessing)
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array.
    '''

    if type(input) is np.ndarray:
        dc_pixels = input
    else:
        dc_pixels = dicom_loader_function(input)

    resized = trans.resize(dc_pixels, (image_size, image_size),
                           anti_aliasing=True)  # for adapt hist - size must be divisible by 2

    equalised = filt.gaussian(exp.equalize_adapthist(resized, clip_limit=0.75), sigma=0.5)
    sato_white = filt.sato(equalised, sigmas=(1.2, 1), black_ridges=False)
    equalised_sato_white = exp.equalize_adapthist(sato_white)
    out = ((equalised_sato_white + equalised) / 2 * 255).astype(np.uint8)

    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'L').save(destination_file_name)
        return True
def to_basic_png(input, destination_file_name=None, dicom_loader_function=None, image_size=512):
    '''
    applies an resize and converts to basic png (0-255) function to an input image.
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array.
    '''

    if type(input) is np.ndarray:
        dc_pixels = input
    else:
        dc_pixels = dicom_loader_function(input)

    out = trans.resize(dc_pixels, (image_size, image_size),
                       anti_aliasing=True)  # for adapt hist - size must be divisible by 2

    out = (out * 255).astype(np.uint8)

    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'L').save(destination_file_name)
        return True

def to_exposure_equalised(input, destination_file_name=None, dicom_loader_function=None, image_size=512):
    '''
    applies an resize and converts to basic png (0-255) function to an input image.
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array.
    '''
    from skimage.morphology import disk
    from skimage.filters import rank

    if type(input) is np.ndarray:
        dc_pixels = input
    else:
        dc_pixels = dicom_loader_function(input)

    out = trans.resize(dc_pixels, (image_size, image_size),
                       anti_aliasing=True)  # for adapt hist - size must be divisible by 2
    out = exp.equalize_hist(out)

    out = (out * 255).astype(np.uint8)

    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'L').save(destination_file_name)
        return True

def to_stacked_sato_and_basic(input, destination_file_name=None, dicom_loader_function=None, image_size=512):
    '''
    applies an resize and converts to basic png (0-255) function to an input image.
    :param input: either dicom pixel data as a numpy array or dicom file name
    :param destination_file_name: the filename of the destination file. This should usually be a png file name. if None, the return will be the converted pixel data
    :param dicom_loader_function: a loader function that loads the dicom file. Can be None. If so, input has to be the already loaded dicom pixel data.
    :param image_size: size of the image to process to. Images should be squared
    :return: if a destination filmname is provided, true if successful. If no file name, the np array. FORMAT = R,G,B where B is 0's
    '''

    sato = to_sato_1(input, dicom_loader_function=dicom_loader_function, image_size=image_size)
    basic = to_exposure_equalised(input, dicom_loader_function=dicom_loader_function, image_size=image_size)


    out = np.stack([sato,basic, np.zeros_like(sato)])
    out = np.moveaxis(out, 0, -1)

    if destination_file_name is None:
        return out
    else:
        PIL.Image.fromarray(out, 'RGB').save(destination_file_name)
        return True

def auto_loader(filename, default_dicom_processor=to_sato_1):
    '''
    a function that returns a numpy array, filename can be '.png' or dicom file. if A dicom file, the default dicom processor is
    applied to the image before it is returned.
    '''

    img = dicom_loader(filename, auto_load_png=True, raw=True)
    extension = os.path.splitext(filename)[1]   
        # if its a png image then just return the image
    if extension.lower() == '.png' or default_dicom_processor is None:
        return img
    else:
        return default_dicom_processor(img)


if __name__ == '__main__':
    pass

