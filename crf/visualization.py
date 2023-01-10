from matplotlib import pyplot as plt
import os
import pickle
import math
import numpy as np


def study_visualization(path_to_study, hyperparameter_to_visualize):
    """
    This is a helper method to visualize hyperparameter choices and their resulting loss values.
    It might help to understand how different hyperparameter effect the quality of the output.
    This method displays the relation between hyperparameter(s) and the resulting loss.

    :param path_to_study: Path to the pickled study file
    :param hyperparameter_to_visualize: iterable of string of the hyperparameter name(s) to be displayed
    """

    with open(path_to_study, "rb") as fp:
        study = pickle.load(fp)

    fig = plt.figure()
    if len(hyperparameter_to_visualize) == 2:
        ax = fig.add_subplot(projection='3d')
        point_cloud = ([], [], [])  # tuple for (hyperparameter 1, hyperparameter 2, loss value)
        for trial in study.trials:
            point_cloud[0].append(trial.params[hyperparameter_to_visualize[0]])
            point_cloud[1].append(trial.params[hyperparameter_to_visualize[1]])
            point_cloud[2].append(trial.values[0])

        ax.scatter(point_cloud[0], point_cloud[1], point_cloud[2], marker="o")
        ax.set_xlabel(hyperparameter_to_visualize[0])
        ax.set_ylabel(hyperparameter_to_visualize[1])
        ax.set_zlabel('loss')
    elif len(hyperparameter_to_visualize) == 1:
        ax = fig.add_subplot()
        point_cloud = ([], [])  # tuple for (hyperparameter, loss value)
        for trial in study.trials:
            point_cloud[0].append(trial.params[hyperparameter_to_visualize[0]])
            point_cloud[1].append(trial.values[0])

        ax.scatter(point_cloud[0], point_cloud[1], marker="o")
        ax.set_xlabel(hyperparameter_to_visualize[0])
        ax.set_ylabel('loss')
    else:  # more hyperparameters given -> show multiple 2d graphs
        n_figure_cols = math.ceil(math.sqrt(len(hyperparameter_to_visualize)))
        n_figure_rows = math.ceil(len(hyperparameter_to_visualize) / n_figure_cols)
        ax_list = fig.subplots(nrows=n_figure_rows, ncols=n_figure_cols)
        for ax_counter, ax in enumerate(np.ndarray.flatten(ax_list)):
            if ax_counter >= len(hyperparameter_to_visualize):
                continue
            point_cloud = ([], [])  # tuple for (hyperparameter, loss value)
            for trial in study.trials:
                point_cloud[0].append(trial.params[hyperparameter_to_visualize[ax_counter]])
                point_cloud[1].append(trial.values[0])
            ax.scatter(point_cloud[0], point_cloud[1], marker="o")
            ax.set_xlabel(hyperparameter_to_visualize[ax_counter])
            ax.set_ylabel('loss')

    # try maximizing the window, does not work on every OS and backend, so catch any exeptions
    try:
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
        # manager.frame.Maximize(True)  # this one might work if the above one doesn't
    except AttributeError:
        pass
    plt.show()


def dataset_visualization(path_to_dataset, prediction_class_num=3):
    """
    Visualizes a dataset created by dataset_preparation.py, so you can visually confirm/debug the data.
    :param path_to_dataset: path to pickled dataset file
    :param prediction_class_num: which prediction class to display
    """
    # load dataset
    with open(path_to_dataset, "rb") as fp:
        dataset = pickle.load(fp)
        assert isinstance(dataset, dict), "Pickled file is not a dictionary"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    if "groundtruth" in dataset:  # if training dataset
        ax1.set_title("Input")
        plot_image1 = ax1.imshow(dataset["input"][0], interpolation='nearest')
        ax2.set_title("Prediction")
        plot_image2 = ax2.imshow(dataset["prediction"][0, :, :, prediction_class_num], interpolation='nearest')
        ax3.set_title("Groundtruth")
        plot_image3 = ax3.imshow(dataset["groundtruth"][0], interpolation='nearest')
    else:  # inference dataset
        ax1.set_title("Input")
        plot_image1 = ax1.imshow(dataset["input"][0], interpolation='nearest')
        ax2.set_title("Prediction")
        plot_image2 = ax2.imshow(dataset["prediction"][0, :, :, prediction_class_num], interpolation='nearest')

    ax4.axis("off")
    sliderObject = plt.axes([0.55, 0.20, 0.4, 0.05])
    slidx = plt.Slider(sliderObject, 'idx', 0, np.shape(dataset["prediction"])[0] - 1, valinit=0, valfmt='%d')

    def update(val):
        if "groundtruth" in dataset:
            plot_image1.set_data(dataset["input"][int(slidx.val)])
            plot_image2.set_data(dataset["prediction"][int(slidx.val), :, :, prediction_class_num])
            plot_image3.set_data(dataset["groundtruth"][int(slidx.val)])
        else:
            plot_image1.set_data(dataset["input"][int(slidx.val)])
            plot_image2.set_data(dataset["prediction"][int(slidx.val), :, :, prediction_class_num])

        fig.canvas.draw_idle()

    slidx.on_changed(update)

    plt.show()
    pass


if __name__ == "__main__":
    # visualization of 2d crf .ckpt file
    study_visualization(
        "crf2d.ckpt",
        ('bilateral_weight', 'bilateral_spatial', 'bilateral_rgb', 'spatial_weight', 'spatial_spatial', 'iteration')
    )

    # visualize a dataset created by dataset_preparation.py
    dataset_visualization(
     os.path.join("final_training_dataset", "Crane__562_715")
    )



