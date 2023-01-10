import denseCRF
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pickle
import os
import torch
import os
import random
import optuna
import numpy as np
import sys
import visualization

from data_processing.data_postprocessing import postprocess_zone_segmenation, extract_front_from_zones
from validate_or_test import front_delineation_metric, visualizations, mask_prediction_with_bounding_box

class CRF2D:
    """
    This class is used to calculate an output-probability map of a given input
    """
    def __init__(self):
        # the hyperparameters used can be set manually here
        self.hyperparameter = \
            (
                13.8,  # weight of bilateral term
                58.3,  # spatial std
                3.0,  # rgb  std
                89.5,  # weight of spatial term
                3.0,  # spatial std
                5.0  # iteration
            )

    def inference(self, input_dataset, prediction_dataset, visualize=False, groundtruth=None):
        """
        :param input_dataset: inputs used to calculate the prediction, must be of shape [H, W, C], where C is the amount of channels and must be 3
        :param prediction_dataset: prediction to use, must be of shape [H, W, C], where C is the amount of classes
        :param visualize: whether to display the result after calculating
        :param groundtruth: if given, will be displayed when 'visualize' is True
        :return: the (hopefully) improved prediction
        """
        result = denseCRF.densecrf(input_dataset, prediction_dataset, self.hyperparameter)

        if visualize:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
            ax1.set_title("Input")
            ax1.imshow(input_dataset, interpolation='nearest')
            ax2.set_title("Prediction")
            ax2.imshow(prediction_dataset[:, :, 3], interpolation='nearest')
            ax4.set_title("Output")
            ax4.imshow(result, interpolation='nearest')
            if groundtruth is not None:
                ax3.set_title("Groundtruth")
                ax3.imshow(groundtruth[:, :, 0], interpolation='nearest')

            plt.show()

        return result.astype("uint8")

    def inference_path(self, input_directory, prediction_directory, output_directory):
        """
        Inference on data given by paths to all relevant data for the inference
        :param input_directory: Path to the raw images
        :param prediction_directory: Path to the prediction numpy array pickles
        :param output_directory: Path to the output directory
        """
        assert os.path.exists(input_directory), "Input path does not exist"
        assert os.path.exists(prediction_directory), "Prediciton path does not exist"
        assert os.path.exists(output_directory), "Output path does not exist"

        # loop over predicitons
        for file_name in os.listdir(prediction_directory):
            # load pickle prediction
            with open(os.path.join(prediction_directory, file_name), "rb") as fp:
                prediction = pickle.load(fp)
                prediction = np.transpose(prediction, axes=(1, 2, 0))
                prediction = prediction.astype("float32")

            # load input image as rgb
            input_layer = cv2.imread(os.path.join(input_directory, file_name + ".png"))
            # skip if input cannot be found
            if input_layer is None:
                continue

            # perform inference and save output as png
            result = self.inference(input_layer, prediction)
            result[result == 1] = 64
            result[result == 2] = 127
            result[result == 3] = 254
            result_image = Image.fromarray(result, mode="L")
            result_image.save(os.path.join(output_directory, file_name + ".png"), "PNG")

    def train(self, training_dataset_batch, batchsize=12, n_iterations=0, duration=0, checkpoint_file=""):
        """
        Use optuna to find the optimal hyperparameters for the CRF
        :param training_dataset_batch: list of paths to training datasets created using 'dataset_preparation.py'
        :param batchsize: size per batch during learning, higher batchsize -> longer calculation, more generalized results
        :param n_iterations: amount of iterations the training will run for
        :param duration: how long the training will run for in seconds, if n_iterations is given, duration will not be used
        :param checkpoint_file: optional, path to a checkpoint file where training progress will be saved and loaded,
                                this way training can be interrupted and continued
        """
        assert n_iterations != 0 or duration != 0, "Either n_iterations or duration must be given"

        if checkpoint_file != "" and os.path.isfile(checkpoint_file):
            # load checkpoint file
            with open(checkpoint_file, "rb") as fp:
                study = pickle.load(fp)
        else:
            study = optuna.create_study()

        if n_iterations != 0:
            study.optimize(lambda trial: self.__train_step(trial, training_dataset_batch, batchsize), n_trials=n_iterations)
        else:
            study.optimize(lambda trial: self.__train_step(trial, training_dataset_batch, batchsize), timeout=duration)

        if checkpoint_file != "":
            with open(checkpoint_file, "wb") as fp:
                pickle.dump(study, fp)

    #TODO: replace/change this '__build_batch()' function when using your own data, otherwise it might not work
    def __build_batch(self, dataset_paths, batchsize, visualize=False):
        """
        Helper Method to create batches of shuffled 2d data from multiple 3d datasets
        :param dataset_paths: list of paths to the datasets generated by 'dataset_generation.py'
        :param batchsize: amount of pictures in the batch
        :return: dict with ndarray of n-batchsize 2d images for groundtruth, input and prediction
        """
        final_batch = {
            "input": [],
            "prediction": [],
            "groundtruth": []
        }
        # create a semi-random distribution for how many elements from every dataset should be taken into the batch
        # this is not perfectly random, but close enough
        dataset_distribution = [1] * batchsize
        dataset_distribution.extend([0] * max(0, len(dataset_paths) - batchsize))
        random.shuffle(dataset_distribution)
        # combine 2 random elemtens until distribution size fits
        while len(dataset_distribution) > len(dataset_paths):
            dataset_distribution.append(dataset_distribution.pop(random.randint(0, len(dataset_distribution)-1)) +
                                        dataset_distribution.pop(random.randint(0, len(dataset_distribution)-1)))
            random.shuffle(dataset_distribution)

        # load images in batch
        for index in range(len(dataset_paths)):
            with open(dataset_paths[index], "rb") as fp:
                dataset = pickle.load(fp)

            for item_to_copy in random.sample(range(dataset["input"].shape[0]), dataset_distribution[index]):
                final_batch["input"].append(dataset["input"][item_to_copy])
                final_batch["prediction"].append(dataset["prediction"][item_to_copy])
                final_batch["groundtruth"].append(dataset["groundtruth"][item_to_copy])

        # visualize the current batch
        if visualize:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
            ax1.set_title("Input")
            plot_image1 = ax1.imshow(final_batch["input"][0], interpolation='nearest')
            ax2.set_title("prediction")
            plot_image2 = ax2.imshow(final_batch["prediction"][0][:, :, 3], interpolation='nearest')
            ax3.set_title("groundtruth")
            plot_image3 = ax3.imshow(final_batch["groundtruth"][0], interpolation='nearest')

            ax4.axis("off")
            slider_object = plt.axes([0.55, 0.20, 0.4, 0.05])
            slidx = plt.Slider(slider_object, 'idx', 0, np.shape(final_batch["input"])[0] - 1, valinit=0, valfmt='%d')

            def update(val):
                plot_image1.set_data(final_batch["input"][int(slidx.val)])
                plot_image2.set_data(final_batch["prediction"][int(slidx.val)][:, :, 3])
                plot_image3.set_data(final_batch["groundtruth"][int(slidx.val)])
                fig.canvas.draw_idle()

            slidx.on_changed(update)
            plt.show()

        return final_batch

    def __train_step(self, trial, training_dataset_batch, batchsize):
        """
        Helper method to feed to optuna, currently only works with grayvalue images (see TODO below)
        """
        self.hyperparameter = \
            (
                trial.suggest_float('bilateral_weight', 1.0, 100.0),  # weight of bilateral term
                trial.suggest_float('bilateral_spatial', 1.0, 100.0),  # spatial std
                trial.suggest_float('bilateral_rgb', 1.0, 100.0),  # rgb std
                trial.suggest_float('spatial_weight', 1.0, 100.0),  # weight of spatial term
                trial.suggest_float('spatial_spatial', 1.0, 100.0),  # spatial std
                trial.suggest_float('iteration', 1.0, 8.0)  # iteration
            )

        # load batch and calculate the CRF result
        batch = self.__build_batch(training_dataset_batch, batchsize)

        # convert groundtruths of batch from grayscale to labelmap
        converted_groundtruth = np.asarray(batch["groundtruth"], dtype=object)
        converted_groundtruth /= 254.0  # max value of an uint8 image
        converted_groundtruth *= (np.shape(batch["prediction"][0])[2] - 1)  # multiply with amount of classes -1
        for index in range(len(converted_groundtruth)):
            converted_groundtruth[index] = np.rint(converted_groundtruth[index])  # round up/down since the values sometimes are slightly below/above the wanted value
            converted_groundtruth[index] = converted_groundtruth[index].astype(np.uint8)

        current_batch_losses = []
        for index in random.sample(list(range(batchsize)), batchsize):
            # calculate crf result, np.repeat because rgb images are needed, but we only have grayscale
            # TODO: better conversion from grayscale to rgb (not just expanding dinemsions, because that breaks everything in case actual rgb images are given)
            crf_output = self.inference(np.repeat(batch["input"][index], 3, axis=2), batch["prediction"][index],
                                        False, groundtruth=np.repeat(np.expand_dims(batch["groundtruth"][index], axis=2), 3, axis=2))
            # convert output to labelmap
            temp_result = np.zeros((np.shape(batch["prediction"][0])[2], crf_output.shape[0], crf_output.shape[1]), dtype=np.uint8)
            for i in range(crf_output.shape[0]):
                for j in range(crf_output.shape[1]):
                    temp_result[crf_output[i][j]][i][j] = 1

            # calculate loss for each data individually because they don't have the same size
            loss = torch.nn.CrossEntropyLoss()
            current_crf_output_tensor = torch.from_numpy(np.expand_dims(temp_result, axis=0))  # add axis so we emulate a batch of size 1
            current_groundtruth_tensor = torch.from_numpy(np.expand_dims(converted_groundtruth[index], axis=0))  # add axis so we emulate a batch of size 1

            loss_num = loss(current_crf_output_tensor.type(torch.float), current_groundtruth_tensor.type(torch.long))
            current_batch_losses.append(loss_num.item())

        return np.exp(sum(current_batch_losses) / len(current_batch_losses))  # return exp of average loss

    # load parameters again, ex: loading trained parameters for inference
    def load_hyperparameter(self, filepath):
        with open(filepath, "rb") as fp:
            temp_hyperparameter = pickle.load(fp)
        self.hyperparameter = temp_hyperparameter

    # save current parameters to a file
    def save_hyperparameter(self, filepath):
        with open(filepath, "wb") as fp:
            pickle.dump(self.hyperparameter, fp)


if __name__ == '__main__':
    crf2d = CRF2D()

    is_training = True
    is_inference = True
    create_visualizations = False

    if is_training:
        crf2d.train(
            [
                os.path.join("final_training_dataset", "JAC__2040_2438"),
                os.path.join("final_training_dataset", "SI__952_843"),
                os.path.join("final_training_dataset", "Jorum__313_234"),
                os.path.join("final_training_dataset", "DBE__517_453"),
                os.path.join("final_training_dataset", "Crane__562_715"),
             ], 16, duration=60*60, checkpoint_file="crf2d_all_datasets.ckpt")

        parameter_save_path = "crf_save"
        if not os.path.exists(parameter_save_path):
            os.makedirs(parameter_save_path)
        crf2d.save_hyperparameter(os.path.join(parameter_save_path, "parameters"))  # save the parameters

    elif is_inference:  # run inference, save output and print front delination metric
        run_number = "run3"  # select dataset from specific neural network version
        layer_destination_path = "dataset/" + run_number
        inference_output_path = "inference_output/" + run_number
        if not os.path.exists(inference_output_path):
            os.makedirs(inference_output_path)

        print("Starting inference")
        crf2d.load_hyperparameter(os.path.join("crf_save", "parameters"))  # load parameter
        crf2d.inference_path(
           os.path.join("..", "data_raw", "sar_images", "test"),
           layer_destination_path,
           inference_output_path
        )

        print("CRF inference finished, starting regular post processing")
        inference__post_process_output_path = "inference_post_output/" + run_number
        if not os.path.exists(inference__post_process_output_path):
            os.makedirs(inference__post_process_output_path)

        # post process inference output (copied from "validate_or_test.py" line 255 from baseline project (see README.md)
        for file_name in os.listdir(inference_output_path):
            complete_predicted_mask = cv2.imread(os.path.join(inference_output_path, file_name).__str__(), cv2.IMREAD_GRAYSCALE)

            resolution = int(os.path.split(file_name)[1][:-4].split('_')[-3])
            meter_threshold = 750  # in meter
            pixel_threshold = meter_threshold / resolution

            post_complete_predicted_mask = postprocess_zone_segmenation(complete_predicted_mask)
            post_complete_predicted_mask = extract_front_from_zones(post_complete_predicted_mask, pixel_threshold)
            post_complete_predicted_mask = mask_prediction_with_bounding_box(post_complete_predicted_mask, file_name, os.path.join("..", "data_raw", "bounding_boxes"))

            cv2.imwrite(os.path.join(inference__post_process_output_path, file_name), post_complete_predicted_mask)

        # calculate metrics for comparing performance. will only work with CaFFe dataset
        front_delineation_metric(inference__post_process_output_path, os.path.join("..", "data_raw", "fronts", "test"))

    elif create_visualizations:
        visualizations("inference_post_output",
                       os.path.join("..", "data_raw", "fronts", "test"),
                       os.path.join("..", "data_raw", "sar_images", 'test'),
                       os.path.join("..", "data_raw", "bounding_boxes"),
                       "visualization_folder")


