import argparse
import time
import numpy as np
import png, os, pydicom # Note that the package png requires module 'pypng'

class PreprocessInput:
    '''
    Apply preprocessing steps before sending file to algorithm
    '''
    def __init__(self):
        pass

    def apply_series_filter(self):
        pass

    def zip_to_tmp_folder(self):
        '''
        Utility function to convert outputs in csv to CARPL findings format
        :return:
        '''
        pass

    def tar_to_tmp_folder(self):
        '''
        Utility function to convert outputs in csv to CARPL findings format
        :return:
        '''
        pass

    def dcm2jpg(self, image_path, output_folder):
        '''
        Utility function to convert DICOM to JPG
        :input: DICOM file path
        :return: JPG file path
        '''
        try:
            ds = pydicom.dcmread(image_path)
            shape = ds.pixel_array.shape
            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)
            if ds.PhotometricInterpretation == "MONOCHROME1":
                image_2d = np.amax(image_2d) - image_2d

            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)

            # Write the file
            with open(os.path.join(output_folder, image_path.replace(".dcm", "")) + '.jpg', 'wb') as jpg_file:
                w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(jpg_file, image_2d_scaled)
        except Exception as e:
            print(e)
            print('Could not convert: ', image_path)

