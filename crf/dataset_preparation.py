import os
import pickle
import re
import shutil
import cv2
from threading import Thread

import numpy as np
from validate_or_test import evaluate_model_on_given_dataset
from models.zones_segmentation_model import ZonesUNet
from data_processing.glacier_zones_data import GlacierZonesDataModule
from data_processing.data_postprocessing import reconstruct_from_grayscale_patches_with_origin

def create_dataset(checkpoint_file_path, hparams_file_path, patch_directory, dst_directory):
    """
    This method creates output from a neural network model and saves the raw prediction output as a pickles numpy array.
    This dataset is necessary for later steps, to find good parameters for our CRF.
    :param checkpoint_file_path: path to the model checkpoint file
    :param hparams_file_path: path to the model hyperparameter file
    :param patch_directory: path to the directory where the patches will temporarily be saved
    :param dst_directory: path to the destination directory
    """
    assert os.path.exists(checkpoint_file_path), "Checkpoint path does not exist"
    assert os.path.exists(hparams_file_path), "Hyperparameter path does not exist"

    model = ZonesUNet.load_from_checkpoint(checkpoint_path=checkpoint_file_path, hparams_file=hparams_file_path, map_location=None)
    datamodule = GlacierZonesDataModule(batch_size=model.hparams.batch_size, augmentation=False, parent_dir="..",
                                        bright=0, wrap=0, noise=0, rotate=0, flip=0)

    mode = "test"
    evaluate_model_on_given_dataset(mode, model, datamodule, patch_directory)  # generate patches from neural network
    reconstruct_from_patches_and_pickle(src_directory=patch_directory, dst_directory=dst_directory)  # combine patches and save them

def reconstruct_from_patches_and_pickle(src_directory, dst_directory):
    """
    This function is identical to the "reconstruct_from_patches_and_binarize" function in data_processing.data_postprocessing,
    with the only change that it doesn't create PNGs but pickles the reconstructed prediction in its "raw" form (all channels preserved).
    Only works with "zones" modality, since nothing else is needed for CRFs
    """
    assert os.path.exists(src_directory), "Source path does not exist"
    assert os.path.exists(dst_directory), "Destination path does not exist"

    patches = os.listdir(src_directory)
    list_of_names = []
    for patch_name in patches:
        list_of_names.append(os.path.split(patch_name)[1].split("__")[0])
    image_names = set(list_of_names)
    for name in image_names:
        print(f"File: {name}")
        # #####################################################################################################
        # Search all patches that belong to the image with the name "name"
        # #####################################################################################################
        pattern = re.compile(name)
        patches_for_image_names = [a for a in patches if pattern.match(a)]
        assert len(patches_for_image_names) > 0, "No patches found for image " + name
        patches_for_image = []  # Will be Number_Of_Patches x Number_Of_Classes x Height x Width
        irow = []
        icol = []
        padded_bottom = int(patches_for_image_names[0][:-4].split("_")[-5])
        padded_right = int(patches_for_image_names[0][:-4].split("_")[-4])

        for file_name in patches_for_image_names:
            # #####################################################################################################
            # Get the origin of the patches out of their names
            # #####################################################################################################
            # naming convention: nameOfTheOriginalImage__PaddedBottom_PaddedRight_NumberOfPatch_irow_icol.png

            # Mask patches are 3D arrays with class probabilities
            with open(os.path.join(src_directory, file_name), "rb") as fp:
                class_probabilities_array = pickle.load(fp)
                assert class_probabilities_array.ndim == 3, "Patch " + file_name + " has not enough dimensions (3 needed). Found: " + str(
                    class_probabilities_array.ndim)
                assert len(
                    class_probabilities_array) <= 4, "Patch " + file_name + " has too many classes (<4 needed). Found: " + str(
                    len(class_probabilities_array))
                patches_for_image.append(class_probabilities_array)
            irow.append(int(os.path.split(file_name)[1][:-4].split("_")[-2]))
            icol.append(int(os.path.split(file_name)[1][:-4].split("_")[-1]))

        # Images are masks and store the probabilities for each class (patch = number_class x height x width)
        class_patches_for_image = []
        patches_for_image = [np.array(x) for x in patches_for_image]
        patches_for_image = np.array(patches_for_image)
        for class_layer in range(len(patches_for_image[0])):
            class_patches_for_image.append(patches_for_image[:, class_layer, :, :])

        class_probabilities_complete_image = []

        # #####################################################################################################
        # Reconstruct image (with number of channels = classes) from patches
        # #####################################################################################################
        for class_number in range(len(class_patches_for_image)):
            class_probability_complete_image, _ = reconstruct_from_grayscale_patches_with_origin(
                class_patches_for_image[class_number],
                origin=(irow, icol), use_gaussian=True)
            class_probabilities_complete_image.append(class_probability_complete_image)

        ######################################################################################################
        # Cut Padding
        ######################################################################################################
        class_probabilities_complete_image = np.array(class_probabilities_complete_image)
        class_probabilities_complete_image = class_probabilities_complete_image[:, :-padded_bottom, :-padded_right]
        with open(os.path.join(dst_directory, name), "wb") as fp:
            pickle.dump(class_probabilities_complete_image, fp)


def combine_and_cut_datasets(dataset_directory, bounding_boxes_directory, output_directory, input_directory, satellite="",
                             orbit=-1, training_dataset=False, groundturth_directory="", delete_patches=False):
    """
    This method takes multiple network results and combines them into one 3d dataset (time being the new dimension).
    Dimensions of the final datasets are [time x height x width x class], all scaled to the same height/width.
    The resulting dataset then can be given to the 2d CRF

    :param dataset_directory: path to the datasets directory
    :param bounding_boxes_directory: path to the bounding boxes directory
    :param output_directory: the directory where the final datasets get saved
    :param satellite: which satellite images should be used, put empty string to use all satellites.
                      might result in downscaling of some images
    :param orbit: only images from this orbit are taken into the dataset. important since different orbits create different images
    :param training_dataset: whether to create a training dataset which includes input and groundtruth data
    :param input_directory: if training_dataset=True, where the input images are located
    :param groundturth_directory: if training_dataset=True, where the groundtruth images are located
    :param delete_patches: whether to delete the patches after combining them
    """
    files = os.listdir(dataset_directory)
    key_words = []  # keywords are glacier names in this case
    for file in files:
        if file.split("_")[0] not in key_words:
            key_words.append(file.split("_")[0])  # add all unique keywords to the list
    for key in key_words:
        # get all files from one key_word, must all be from one satelite (since different satelites make different image sizes)
        matching_files = [a for a in files if key in a and satellite in a and (orbit == -1 or orbit == int(a[-3:]))]
        # if no data was found -> inform user and skip to next glacier
        if len(matching_files) == 0:
            print("No files from satellite " + satellite + " and orbit " + str(orbit) + " (-1 is default value and not an error) were found for " + str(key))
            continue
        # get all bounding boxes and sum them up to one bounding box containing all the other ones
        smallest_patch_size = []  # [width, height]
        final_prediction_dataset = []
        final_input_dataset = []
        if training_dataset:
            final_groundtruth_dataset = []

        for file in sorted(matching_files):
            with open(os.path.join(dataset_directory, file), "rb") as fp:
                dataset_layer = pickle.load(fp)

            with open(os.path.join(bounding_boxes_directory, file + "_front_extent_coord.txt")) as f:
                coord_file_lines = f.readlines()
            left_upper_corner = [max(round(float(coord)), 0) for coord in coord_file_lines[1].split(",")]
            right_lower_corner = [max(round(float(coord)), 0) for coord in coord_file_lines[3].split(",")]

            # slice dataset_layer according to final_bounding_box
            dataset_layer = dataset_layer[:, right_lower_corner[1]:left_upper_corner[1]+1, left_upper_corner[0]:right_lower_corner[0]+1]
            final_prediction_dataset.append(dataset_layer)

            input_layer = cv2.imread(os.path.join(input_directory, file + ".png"))
            input_layer = cv2.cvtColor(input_layer, cv2.COLOR_BGR2GRAY)
            input_layer = input_layer[right_lower_corner[1]:left_upper_corner[1] + 1, left_upper_corner[0]:right_lower_corner[0] + 1]
            final_input_dataset.append(input_layer)

            if training_dataset:
                groundtruth_layer = cv2.imread(os.path.join(groundturth_directory, file + "_zones.png"))
                groundtruth_layer = cv2.cvtColor(groundtruth_layer, cv2.COLOR_BGR2GRAY)
                # cut layer
                groundtruth_layer = groundtruth_layer[right_lower_corner[1]:left_upper_corner[1] + 1, left_upper_corner[0]:right_lower_corner[0] + 1]
                final_groundtruth_dataset.append(groundtruth_layer)

            # remember the smallest patch, so we can resize later
            if len(smallest_patch_size) == 0:
                # initialize
                smallest_patch_size.append(np.shape(dataset_layer)[2])
                smallest_patch_size.append(np.shape(dataset_layer)[1])
            else:
                # replace the smallest size where needed
                if smallest_patch_size[0] > np.shape(dataset_layer)[2]:
                    smallest_patch_size[0] = np.shape(dataset_layer)[2]
                if smallest_patch_size[1] > np.shape(dataset_layer)[1]:
                    smallest_patch_size[1] = np.shape(dataset_layer)[1]

        # scale down all layers to the smallest layer using an image editing library (cv2)
        for index, layer in enumerate(final_prediction_dataset):
            # only resize if necessary
            if smallest_patch_size[0] != np.shape(layer)[2] or smallest_patch_size[1] != np.shape(layer)[1]:
                layer_as_list = list(layer)  # list with every entry being the predictions for a different class
                for i in range(len(layer_as_list)):
                    layer_as_list[i] = cv2.resize(layer_as_list[i], dsize=(smallest_patch_size[0], smallest_patch_size[1]), interpolation=cv2.INTER_CUBIC)
                final_prediction_dataset[index] = np.asarray(layer_as_list)
                final_input_dataset[index] = cv2.resize(final_input_dataset[index], dsize=(smallest_patch_size[0], smallest_patch_size[1]), interpolation=cv2.INTER_CUBIC)
                final_input_dataset[index] = cv2.normalize(final_input_dataset[index], None, 0, 255, cv2.NORM_MINMAX)  # normalize input images
                if training_dataset:
                    final_groundtruth_dataset[index] = cv2.resize(final_groundtruth_dataset[index], dsize=(smallest_patch_size[0], smallest_patch_size[1]), interpolation=cv2.INTER_CUBIC)

        # rearrange so class-axes is the 4th axes, as required by the CRF docs
        final_prediction_dataset = np.asarray(final_prediction_dataset, dtype=np.float32)
        final_prediction_dataset = np.transpose(final_prediction_dataset, axes=(0, 2, 3, 1))
        final_input_dataset = np.asarray(final_input_dataset, dtype=np.uint8)
        final_input_dataset = np.expand_dims(final_input_dataset, axis=3)  # add axis since CRF needs a channel axis (which we don't have in a grayscale image)
        if training_dataset:
            final_groundtruth_dataset = np.asarray(final_groundtruth_dataset, dtype=np.float32)
            final_dataset = {
                "input": final_input_dataset,
                "prediction": final_prediction_dataset,
                "groundtruth": final_groundtruth_dataset
            }
        else:  # inference dataset
            final_dataset = {
                "input": final_input_dataset,
                "prediction": final_prediction_dataset
            }

        orbit_as_string = "" if orbit == -1 else "_" + str(orbit)
        with open(os.path.join(output_directory, key + "_" + satellite + orbit_as_string + "_" + str(smallest_patch_size[0]) + "_" + str(smallest_patch_size[1])), "wb") as fp:
            pickle.dump(final_dataset, fp)

        if delete_patches:
            shutil.rmtree(dataset_directory)  # clean up


def average_histogram_matching(image_path, output_path, amount_threads=4, filenames_list=[]):
    """
    This method histogram matches all images to the average histogram of all images in the image_path directory.
    This will give all images from the same galcier a similar grayvalue distribution.

    :param amount_threads: how many threads should be used for mutlithreading
    :param image_path: path to the satellite images
    :param output_path: path where the images should be saved
    :param filenames_list: (optional) only images files in this list will be processed
    """
    if len(filenames_list) == 0:
        filenames = os.listdir(image_path)
    else:
        filenames = filenames_list
    histograms = np.zeros((len(filenames), 256))

    # calculate cdf histogram for all images, multithreaded (about twice as fast)
    def cumulative_histogram_thread(index, result_object, filenames_chunk):
        for file_index, current_filename in enumerate(filenames_chunk):
            result_object[index + file_index] = np.cumsum(np.histogram(cv2.imread(os.path.join(image_path, current_filename), cv2.IMREAD_GRAYSCALE), 256, density=True)[0])  # normalized cumulative histogram

    threads = [None] * amount_threads
    current_result_index = 0
    filename_chunks = np.array_split(filenames, amount_threads)
    for i in range(amount_threads):
        threads[i] = Thread(target=cumulative_histogram_thread, args=(current_result_index, histograms, filename_chunks[i]))
        threads[i].start()
        current_result_index += len(filename_chunks[i])

    # wait for all threads to finish
    for i in range(amount_threads):
        threads[i].join()

    average_cumulative_histogram = np.mean(histograms, axis=0)

    # histogram match all images and write them to output_path (not multithreaded, no speed gain [python multithreading is weird])
    for current_filename in filenames:
        current_img = cv2.imread(os.path.join(image_path, current_filename), cv2.IMREAD_GRAYSCALE)
        current_cdf = np.cumsum(np.histogram(current_img, 256, density=True)[0])
        for y in range(current_img.shape[0]):
            for x in range(current_img.shape[1]):
                if current_img[y, x] == 0:  # leave 0 zero
                    continue
                current_img[y, x] = np.abs(average_cumulative_histogram - current_cdf[current_img[y, x]]).argmin()

        cv2.imwrite(os.path.join(output_path, current_filename), current_img)





if __name__ == "__main__":
    src = os.getcwd()

    checkpoint_path = os.path.join(src, "..", "checkpoints", "zones_segmentation", "run_4", "-epoch=147-avg_metric_validation=0.90.ckpt")
    hparams_path = os.path.join(src, "..", "tb_logs", "zones_segmentation", "run_4", "log", "version_0", "hparams.yaml")
    bounding_boxes_path = os.path.join(src, "..", "data_raw", "bounding_boxes")

    patch_path = os.path.join(src, "patches")
    if not os.path.exists(patch_path):
        os.makedirs(patch_path)

    layer_destination_path = os.path.join(src, "dataset")
    if not os.path.exists(layer_destination_path):
        os.makedirs(layer_destination_path)

    final_destination_path = os.path.join(src, "final_training_dataset_orbit_val")
    if not os.path.exists(final_destination_path):
        os.makedirs(final_destination_path)


    ## example for further preprocessing using histogram matching, not used in the
    # testpath = os.path.join(src, "hist_matched")
    # if not os.path.exists(testpath):
    #     os.makedirs(testpath)
    #
    # all_filenames = os.listdir(os.path.join(src, "..", "data_raw", "sar_images", "train"))
    # filtered_filenames = [a for a in all_filenames if "JAC" in a]
    #
    # average_histogram_matching(os.path.join(src, "..", "data_raw", "sar_images", "train"),
    #                            testpath, 4, filtered_filenames)


    # THE 2 LINES BELOW ONLY NEED TO BE DONE ONCE PER NETWORK CHECKPOINT, COMMENT OUT AFTERWARDS TO SAVE COMPUTING TIME UNLESS "delete_patches=True" IN "combine_and_cut_datasets"
    create_dataset(checkpoint_path, hparams_path, patch_path, layer_destination_path)
    shutil.rmtree(patch_path)  # clean up

    #TODO: the current example creates one dataset for every glacier, but only from 'TDX' satellite and orbit '24'
    combine_and_cut_datasets(layer_destination_path, bounding_boxes_path, final_destination_path,
                             input_directory=os.path.join(src, "..", "data_raw", "sar_images", "val"),
                             orbit=24,
                             satellite="TDX",
                             training_dataset=True,
                             groundturth_directory=os.path.join(src, "..", "data_raw", "zones", "val"))
