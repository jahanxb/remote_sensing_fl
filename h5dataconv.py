import h5py
from PIL import Image
import os
import numpy as np

import os


def create_h5_dataset(image_dir, output_filename,val, img_shape=(256, 256, 3)):
    # Get a list of all the image files in the dataset directory
    if val:
        # if size>0 that means its validation
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        size = 1
        image_files  = image_files[:size]
        # Create a new HDF5 file
        with h5py.File(output_filename, 'w') as f:

            # Create a dataset in the file to store the images
            img_data = f.create_dataset('images', (len(image_files), *img_shape), dtype='uint8')
        
            # Loop over the image files and add them to the HDF5 dataset
            for i, image_file in enumerate(image_files):
                # Open the image file
                with Image.open(image_file) as img:
                    # Resize the image (if necessary)
                    img_resized = img.resize(img_shape[:2])
                
                    # Convert the image data to a NumPy array
                    img_array = np.array(img_resized)
                
                    # Add the image data to the HDF5 dataset
                    img_data[i] = img_array

        # Print a message indicating that the HDF5 file has been created
        print(f'Created HDF5 file "{output_filename}" with {len(image_files)} images.')    
    else:
        image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        size = 5
        image_files  = image_files[:size]
        # Create a new HDF5 file
        with h5py.File(output_filename, 'w') as f:

            # Create a dataset in the file to store the images
            img_data = f.create_dataset('images', (len(image_files), *img_shape), dtype='uint8')
        
            # Loop over the image files and add them to the HDF5 dataset
            for i, image_file in enumerate(image_files):
                # Open the image file
                with Image.open(image_file) as img:
                    # Resize the image (if necessary)
                    img_resized = img.resize(img_shape[:2])
                
                    # Convert the image data to a NumPy array
                    img_array = np.array(img_resized)
                
                    # Add the image data to the HDF5 dataset
                    img_data[i] = img_array

        # Print a message indicating that the HDF5 file has been created
        print(f'Created HDF5 file "{output_filename}" with {len(image_files)} images.')    
    





if __name__ == '__main__':
    
    # Specify the path to your dataset of images
    train_image_dir = 'NWPU-RESISC45/airplane'
    val_image_dir = 'NWPU-RESISC45/airplane'

    # Create the HDF5 files
    create_h5_dataset(train_image_dir, 'data/train_small.h5',False)
    create_h5_dataset(val_image_dir, 'data/val.h5',True)
 
    # # Specify the path to your dataset of images
    # image_dir = 'NWPU-RESISC45/airplane'

    # # Get a list of all the image files in the dataset directory
    # image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    # # Create a new HDF5 file
    # with h5py.File('train_small.h5', 'w') as f:

    #     # Create a dataset in the file to store the images
    #     img_data = f.create_dataset('images', (len(image_files), 256, 256, 3), dtype='uint8')
    
    #     # Loop over the image files and add them to the HDF5 dataset
    #     for i, image_file in enumerate(image_files):
    #         # Open the image file
    #         with Image.open(image_file) as img:
    #             # Resize the image (if necessary)
    #             img_resized = img.resize((256, 256))
            
    #             # Convert the image data to a NumPy array
    #             img_array = np.array(img_resized)
            
    #             # Add the image data to the HDF5 dataset
    #             img_data[i] = img_array

    # # Print a message indicating that the HDF5 file has been created
    # print(f'Created HDF5 file with {len(image_files)} images.')
