# GlacierCRF
Using Conditional Random Fields to post process neural network predictions using the predictions and the original inputs

The original paper can be found here: (as soon as it is published)

---
### How to use this according to the paper?
This github can be used as a standalone project (if you have the [pretrained network checkpoints](https://zenodo.org/record/6469519)). The baseline by Gourmelon this was built upon can be found [here](https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them). The files and folder `validate_or_test.py`, `models` and `data_processing` were taken for some functionality (mainly related to the U-Net and dataset building and preprocessing). In case you want to rebuild/retrain the complete project (including the baseline) from scratch, you should start with Gourmelon's baseline and then only add the `crf` directory to their project.

Every file has a `if __name__ == "__main__"` function on the bottom to show how to use the functions/classes. You might still need to adjust some parameter/paths/etc

This project is designed to work only with the [CaFFe dataset](https://doi.pangaea.de/10.1594/PANGAEA.940950) which should be placed in an additional directory (ex.: `data_raw`) and then preprocessed as described in Gourmelon's baseline.

---
### How can I use my own dataset?

If you have your own dataset consisting of
- input files (only grayscale images work, aka only one color channel)
- predictions (ex.: from a neural network, one channel per class and channels being the last dimension; see code comments for further explanaition)
- groundtruths (final image, again only one channel layer, aka grayscale image)

you can follow these steps to use the CRF with your own data.

- Ignore the `dataset_preparation.py` file
- Change `__build_batch()` in `CRF2D.py` so it loads batches with your own data into the `final_batch` dictionary
- Remove any post-processing in the `main` function (unless you want the same post-processing), the output from the CRF is probably all you will want

This should be all that is needed and you should be able to use the CRF with your own data.
---
