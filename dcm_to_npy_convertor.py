import os
import numpy as np
import pydicom
from colorama import Fore, Style



def is_dicom_pixel_data_compressed(dicom_file_path):
    """ In some cases, dicom file might get compressed due to sizing. To get actual accuracy of model,
    uncompressed data should be given to model.Hence it's mandatory to check whether file is compressed or not

    Parameters
    ----------
    dicom_file_path : str
        File path that represents dicom file

    Returns
    ------
    True: if dicom file is compressed or else return False

    """

    try:
        ds = pydicom.dcmread(dicom_file_path)

        # Check if Transfer Syntax UID indicates compression
        transfer_syntax_uid = ds.file_meta.TransferSyntaxUID
        compressed_transfer_syntax_uids = [
            '1.2.840.10008.1.2.4.',  # JPEG Compression
            '1.2.840.10008.1.2.4.50', '1.2.840.10008.1.2.4.51',  # JPEG Baseline and Extended
            '1.2.840.10008.1.2.4.90', '1.2.840.10008.1.2.4.91',  # JPEG 2000 Lossless and Part 2 Lossless
            '1.2.840.10008.1.2.4.70', '1.2.840.10008.1.2.4.80'  # JPEG Lossless and RLE Lossless
        ]

        return any(uid in transfer_syntax_uid for uid in compressed_transfer_syntax_uids)
    except pydicom.errors.InvalidDicomError as e:
        print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}Invalid DICOM file: {e}")
    except AttributeError as e:
        print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}Attribute error: {e}")
    except Exception as e:
        print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}An error occurred: {e}")

#
# https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data
def convert_dicom_to_npy(dicom_folder, npy_folder):
    """
    This method used to convert dicom file to npy format. For deep learning models using medical images in the
    DICOM format, it's common to focus on extracting and use the pixel data (image data) rather than converting all
    metadata to NumPy arrays.
    The pixel data contains the actual image information and is the primary input for training deep learning models.

    Sample data source: https://www.kaggle.com/competitions/vinbigdata-chest-xray-abnormalities-detection/data
    Data source is stored in testdata folder

    :param dicom_folder: Folder where dicom files are stored
    :param npy_folder: Folder where npy files are stored
    :return: None
    """
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm') or f.endswith('.dicom')]

    for dicom_file in dicom_files:
        try:
            dicom_path = os.path.join(dicom_folder, dicom_file)
            print("\n")
            print(f"{Fore.GREEN}[INFO] {Style.RESET_ALL} ********************CURRENT IMAGE ********************")
            print("DICOM FILE PATH:", dicom_path)

            # Check whether pixel data is compressed or not
            compressed = is_dicom_pixel_data_compressed(dicom_path)
            print(f"The pixel data is {'compressed' if compressed else 'not compressed'}.")

            # Prepare npy file path
            npy_path = os.path.join(npy_folder, dicom_file.replace('.dcm', '.npy'))

            # Read DICOM file
            dicom_data = pydicom.dcmread(dicom_path)

            # # Open a text file for writing
            # with open('output.txt', 'w') as output_file:
            #     # Iterate over attributes and write key-value pairs to the file
            #     for key, value in dicom_data.items():
            #         output_file.write(f"{key}: {value}\n")

            # Print DICOM metadata
            print("DICOM Metadata:")
            print(dicom_data)

            # Print pixel data
            print("Pixel Data:")
            pixel_data = dicom_data.pixel_array
            print(pixel_data)

            # Save as NPY file with pixel data
            np.save(npy_path, pixel_data.astype(np.float32))

        except pydicom.errors.InvalidDicomError as e:
            print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}Invalid DICOM file: {e}")
        except Exception as e:
            print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}An error occurred: {e}")


def verify_npy(npy_folder):
    """
    Verify converted npy file format from dicom file
    :param npy_folder: Folder where npy files are stored
    :return:
    """
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith('.npy')]

    for npy_file in npy_files:
        try:
            # Load the NPY file
            loaded_array = np.load(npy_folder + '/' + npy_file)
            print("\n")
            print(f"{Fore.GREEN}[INFO] {Style.RESET_ALL} ********************CURRENT IMAGE ********************")
            print("\n")

            # Print information about the loaded array
            print("File path:", npy_file)
            print("Loaded array shape:", loaded_array.shape)
            print("Loaded array data type:", loaded_array.dtype)
            print("Loaded array content:")
            print(loaded_array)

        except Exception as e:
            print(f"{Fore.RED}[ERROR] {Style.RESET_ALL}Error loading NPY file: {e}")


# Set DICOM folder path and NPY folder path
dicom_folder = "/Users/mathurigathavarajah/Personal/Cardiff-Met/Research/dicom_npy_convertor/test_data"
npy_folder = "/Users/mathurigathavarajah/Personal/Cardiff-Met/Research/dicom_npy_convertor/npy_images"

# Create the NPY folder if it doesn't exist
os.makedirs(npy_folder, exist_ok=True)

# Convert DICOM files to NPY format
print("\n")
print(" -------------- CONVERT DICOM TO NPY -------------- ")
print("\n")
convert_dicom_to_npy(dicom_folder, npy_folder)
print("\n")
print(" -------------- VERIFY CONVERTED NPY -------------- ")
print("\n")
verify_npy(npy_folder)
