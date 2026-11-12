# ResNet50 Tuberculosis Classifier: Thesis Prototype for efficientNet B3

## Overview

This notebook (`resnet50_v6.ipynb`) is my group's undergraduate thesis prototype. It classifies chest X-rays as Normal or Tuberculosis, and the best run it shows reached 82.86% test accuracy (AUC 0.87).

- **Model:** ResNet50 pretrained on ImageNet, with its weights frozen. A small dense head sits on top: 128 → 256 → 512 units, dropout 0.3, and a sigmoid output.
- **Explainability:** Grad-CAM heatmaps on the last convolutional block (`conv5_block3_out`) show which lung regions drove each prediction.
- **Environment:** Google Colab, Python 3.10, TensorFlow/Keras 2.15. A comparison against an EfficientNetB3 model is included at the end.

## Status: prototype only, not the notebook for efficientNet Model

This is a research and learning prototype from a thesis. It has not been clinically validated and must not be used to diagnose or screen patients.

The notebook is kept as a record of the work and its results, not as maintained or reproducible software. The faults listed below are left in on purpose, so the outputs stay exactly as they were produced.

## Data

The notebook uses two public datasets. Neither one is included in this repo.

| Dataset | Images used | Split (train / val / test) | Used for |
| --- | --- | --- | --- |
| Kaggle Tuberculosis Chest X-ray Database | 1,400 (700 Normal sampled, 700 TB) | 979 / 211 / 210 (70 / 15 / 15%) | Earlier run, shown in the evaluation outputs |
| Segmented Shenzhen + Montgomery CXR sets | 704 (labels taken from `_0` / `_1` filename suffix) | 598 / 36 / 70 (85 / 5 / 10%) | Current code and the v6.3 training run |

Dataset References: 

Segmentation Model: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

Classification Model:
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
## Pipeline

1. **Configure.** Set the dataset and model save paths, the model name, and the hyperparameters: batch size 8, 512×512 input, Adam learning rate 2e-4.
2. **Split.** Build DataFrames of file paths and labels, then split them with `train_test_split`.
3. **Load images.** `ImageDataGenerator` applies ResNet50 preprocessing. Training images are also augmented: zoom 0.1, rotation 10°, brightness 0.9–1.1, horizontal flip.
4. **Build.** A frozen ResNet50 base (max pooling) plus the dense head, trained with binary cross-entropy and tracking accuracy and AUC.
5. **Train.** Up to 20 epochs, with early stopping (patience 5), ReduceLROnPlateau (factor 0.5), a best-model checkpoint, and a CSV log.
6. **Evaluate.** On the test set: a classification report, a confusion matrix, and a ROC curve.
7. **Explain.** Grad-CAM heatmaps are overlaid on single X-rays and on five random test images.

## Results shown in the notebook

All three evaluations ran on the same 210-image Kaggle test set (97 Normal, 113 TB). The two ResNet50 models tie at about 82–83% accuracy. The EfficientNetB3 result is not valid (see Known faults).

| Model (notebook cell) | Accuracy | AUC | TB recall | TB precision | Normal recall |
| --- | --- | --- | --- | --- | --- |
| ResNet50, earlier run (cell 22) | 82.4% | 0.897 | 0.80 | 0.87 | 0.86 |
| ResNet50 v5, loaded from file (cell 31) | 82.9% | 0.874 | 0.73 | 0.93 | 0.94 |
| EfficientNetB3 v2, loaded from file (cell 35) | 67.1% | 0.649 | — | — | 0.31 |

The v6.3 training run on the segmented set (cell 17) has no test evaluation; it stopped with an error after training. Its first epoch reached 66.7% validation accuracy.

## Known faults and errors

The biggest problem is that the outputs and the code don't match: the code trains on the segmented set, but the evaluation outputs come from older runs on the Kaggle set.

**Errors in the code**

| Issue | Cell | Effect |
| --- | --- | --- |
| `preprocess_input` is reassigned to MobileNetV2's version | 26 | Grad-CAM and single-image predictions afterwards use the wrong preprocessing; the 0.51 score in cell 30 can't be trusted |
| `NameError: name 'csv' is not defined` after training | 17 | The import cell wasn't re-run in that session; the ~3-hour (11,147 s) run finished, but the training-duration row was never logged |
| EfficientNetB3 evaluated with ResNet50 preprocessing | 35 | Loss 29.0 and 67.1% accuracy reflect the wrong preprocessing, not the model |
| 224×224 images fed to a model trained at 512×512 | 26, 27 | Single-image predictions and Grad-CAM on the v6 model don't match its input size |
| The "model already exists" guard catches its own `sys.exit()` | 3 | Execution carries on anyway and the safety flag is never checked, so an existing model can be overwritten |
| `EPOCHS = 10` is defined, but `fit` uses `epochs=20` | 7, 17 | The setting has no effect |
| Converts an unrelated EfficientNetB3 model to `.h5` | 18 | Out of place in this notebook |

**Reproducibility**

- Cells were run out of order, across sessions from February to June 2024, and most have no execution count.
- Paths are hard-coded to a personal Google Drive. The notebook also loads saved model files that aren't in the repo (ResNet50 v5, EfficientNetB3 v2).
- The random test-image picks (cell 32) use no seed.

**Methodology**

- Splits aren't stratified. The segmented validation set (36 images: 23 Normal, 13 TB) and test set (70 images) are too small for reliable estimates.
- The decision threshold is fixed at 0.5 rather than tuned on validation data.
- There's no cross-validation, no external test set, and no check for overlapping images between datasets.
- The ResNet50 base stays frozen, so the model is never fine-tuned.

**Code quality**

- Dead code: an unused `augment` layer pipeline, large commented-out blocks in cells 26, 29 and 32, and an unused `generate_random_numbers`.
- `Image` from PIL is shadowed by `IPython.display.Image`. Some comments are in Tagalog.

## What I'd do differently

- Use stratified train/val/test splits with one fixed seed, and a validation set large enough to tune on.
- Use one preprocessing function and one input size, from training through Grad-CAM.
- Unfreeze the top ResNet50 blocks for fine-tuning at a low learning rate.
- Choose the decision threshold on validation data (for example with Youden's J), and report sensitivity and specificity alongside accuracy.
- Use k-fold cross-validation, test on a held-out external dataset, and check for duplicate images across datasets.
- Run the notebook top to bottom in one session, so every output matches the code above it.

## Reproducing

1. Use Python 3.10 with TensorFlow/Keras 2.15, plus pandas, scikit-learn, OpenCV, seaborn, matplotlib, openpyxl and pydot. A GPU runtime (for example Colab) is recommended.
2. Download the datasets listed under Data.
3. Edit the path constants in the configuration cell (cell 3): `DATASET_PATH`, `SEGMENTED_DATASET` and `MODEL_SAVEPATH`.
4. Run the cells in order from the top. Skip the cells that load model files not included here (cells 18, 29 and 34).

## Sample results and Verifications

Sample output 

<img width="450" height="485" alt="image" src="https://github.com/user-attachments/assets/6008aa07-aa34-41c5-8abc-d2ea13a1ff39" />

Confusion Matrix

<img width="646" height="645" alt="image" src="https://github.com/user-attachments/assets/e15d0178-3bd1-43ce-837a-c1a5f1e8ac2e" />


