# Breast Cancer Image Classification Using Deep Learning

## 1. Project Overview

This project uses deep learning and transfer learning to classify breast ultrasound images into three dataset categories: **Benign, Malignant, and Normal**. It uses the Breast Ultrasound Images Dataset (BUSI) and a MobileNetV2-based convolutional neural network.

The project includes image preprocessing, dataset visualization, model training, performance evaluation, and prediction on a user-selected ultrasound image through a Python interface.

**Note:** This is an educational machine learning project, not a medical diagnostic system.

## 2. Objectives

- Classify breast ultrasound images into benign, malignant, and normal categories.
- Apply image preprocessing and data augmentation to prepare images for training.
- Use transfer learning with the MobileNetV2 architecture.
- Evaluate model performance using multiple classification metrics.
- Visualize class distribution, training performance, confusion matrix, and ROC curves.
- Allow users to select an image from their computer and view the model's prediction.

## 3. Dataset

**Dataset:** Breast Ultrasound Images Dataset (BUSI)

**Source:** [Download the BUSI Dataset from Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

The dataset contains breast ultrasound images organized into three categories.

| Class | Description |
|---|---|
| Benign | Images associated with benign findings in the dataset |
| Malignant | Images associated with malignant findings in the dataset |
| Normal | Images categorized as normal in the dataset |

The notebook scans the dataset folders, checks image files, excludes mask images, and creates a dataset table for further processing.

### Dataset Split

The dataset is divided into training, validation, and testing subsets using stratified sampling.

| Subset | Percentage | Purpose |
|---|---:|---|
| Training | 70% | Train the model |
| Validation | 15% | Monitor training and select the best model |
| Testing | 15% | Evaluate performance on held-out images |
| **Total** | **100%** | |

The split uses a random seed of `42` to support reproducibility. The exact number of images in each subset depends on the dataset files successfully detected by the notebook.

## 4. Technologies Used

- **Programming language:** Python
- **Deep learning framework:** TensorFlow and Keras
- **Model architecture:** MobileNetV2 with transfer learning
- **Image processing:** Pillow and TensorFlow
- **Data handling:** NumPy and Pandas
- **Visualization:** Matplotlib and Seaborn
- **Model evaluation:** Scikit-learn
- **Development environment:** Jupyter Notebook in Visual Studio Code
- **Image selection interface:** Tkinter

## 5. Model Architecture and Configuration

The project uses MobileNetV2 with pretrained ImageNet weights as the feature extraction backbone. Its original classification head is replaced with layers suitable for the three target classes.

| Parameter | Configuration |
|---|---|
| Input image size | 160 × 160 pixels |
| Input channels | 3 (RGB) |
| Base model | MobileNetV2 |
| Pretrained weights | ImageNet |
| Base model during initial training | Frozen |
| Pooling layer | Global Average Pooling |
| Dropout rate | 0.30 |
| Output layer | Dense, 3 neurons |
| Output activation | Softmax |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Sparse Categorical Cross-Entropy |
| Batch size | 16 |
| Maximum training epochs | 12 |
| Random seed | 42 |

### Image Preprocessing

The notebook performs the following preprocessing steps:

1. Reads and validates image files.
2. Converts images to RGB format.
3. Resizes images to 160 × 160 pixels.
4. Applies MobileNetV2-specific input normalization.
5. Applies data augmentation during training, including horizontal flipping, rotation, and zoom.

Early stopping, model checkpointing, and learning-rate reduction are used to help manage training and retain the best validation model.

## 6. Project Workflow

1. **Dataset loading:** Locate and validate ultrasound images.
2. **Exploratory data analysis:** Visualize class distribution and sample images.
3. **Data preparation:** Create stratified training, validation, and test sets.
4. **Preprocessing:** Resize, normalize, and batch images.
5. **Model building:** Configure MobileNetV2 with a three-class classification head.
6. **Model training:** Train using the training set and monitor validation performance.
7. **Model evaluation:** Calculate classification metrics on the held-out test set.
8. **Visualization:** Generate training curves, a confusion matrix, and ROC curves.
9. **Baseline comparison:** Compare the trained model against a majority-class baseline.
10. **Image prediction:** Select a new image and display the predicted dataset class and model scores.

## 7. Performance Evaluation

The trained model is evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified test images |
| Precision | Measures the correctness of positive predictions |
| Recall | Measures how many actual examples of a class are identified |
| Macro F1-score | Calculates the F1-score for each class and averages them equally |
| Confusion matrix | Shows correct and incorrect predictions for each class |
| ROC curve | Illustrates class-wise discrimination across thresholds |
| ROC-AUC | Summarizes the area under the ROC curve |

The notebook also generates a classification report and compares the trained model with a baseline that predicts the most common training class.

### Actual Model Results

**Update this table after running all notebook cells.** The figures must be taken from the test evaluation output, not from the training accuracy.

| Evaluation metric | Result |
|---|---:|
| Test accuracy | Update after execution |
| Macro precision | Update after execution |
| Macro recall | Update after execution |
| Macro F1-score | Update after execution |
| Macro ROC-AUC (one-vs-rest) | Update after execution |
| Baseline test accuracy | Update after execution |
| Baseline macro F1-score | Update after execution |

The notebook saves the detailed per-class metrics and comparison results as CSV files.

## 8. Interactive Image Prediction

After training, the notebook allows the user to select an ultrasound image from their computer using a Tkinter file-selection dialog.

The selected image is:

1. Displayed in the notebook.
2. Converted to RGB and resized to 160 × 160 pixels.
3. Preprocessed using MobileNetV2's preprocessing function.
4. Passed to the trained model for prediction.
5. Classified into one of the three dataset categories.

The output displays the predicted dataset class, the model's confidence score, and the scores for all three classes in descending order.

**Important:** A model confidence score is not a measure of medical certainty. The output is for educational demonstration only and must not be used to diagnose or guide treatment.

## 9. Project Structure

```text
Task-3 Breast Cancer Image Classification/
│
├── Breast_Cancer_Classification.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   └── README.md
│
├── models/
│   ├── best_model.keras
│   ├── breast_ultrasound_classifier.keras
│   └── class_names.json
│
└── visualizations/
    ├── class_distribution.png
    ├── sample_images.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    └── baseline_comparison.png
```

The model and visualization files are generated when the relevant notebook cells run successfully. Training history and classification metric CSV files are also saved by the notebook. The dataset and generated model files may not be present in a fresh GitHub clone until they have been downloaded or generated locally.

## 10. Installation and Execution

### Prerequisites

- Python 3.10 or a compatible Python version supported by the installed TensorFlow release
- Visual Studio Code
- Jupyter and Python extensions for VS Code
- Git, for version control
- Internet access for downloading the dataset and pretrained weights

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Breast-Cancer-Image-Classification.git
cd Breast-Cancer-Image-Classification
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install ipywidgets
```

The `ipywidgets` package is optional for the Tkinter-based image-selection cell but is needed if you also use a notebook upload widget.

### Step 4: Download the Dataset

Download the BUSI dataset from Kaggle, extract it, and place the class folders inside the project's `dataset/` directory.

Check that the notebook can find image files under the `benign`, `malignant`, and `normal` folders. If the extracted ZIP creates an extra parent directory, keep the original class folders nested under `dataset/`; the notebook searches recursively.

### Step 5: Open and Run the Notebook

1. Open the project folder in Visual Studio Code.
2. Open `Breast_Cancer_Classification.ipynb`.
3. Select the Python interpreter associated with `.venv`.
4. Run the notebook cells from top to bottom.
5. Wait for dataset analysis, model training, and evaluation to finish.
6. Run the final image-prediction cell and select an ultrasound image.

The first run may take longer because TensorFlow may download pretrained MobileNetV2 weights and the model must be trained. Training time depends on your hardware.

## 11. Generated Outputs

After successful execution, the project produces:

- Class distribution chart
- Sample ultrasound image visualization
- Training and validation accuracy/loss curves
- Training history CSV file
- Test classification report CSV file
- Confusion matrix visualization
- Class-wise ROC curves and AUC values
- Baseline comparison results and chart
- Saved Keras model files
- Class-name mapping JSON file
- Prediction output for a selected image

## 12. Limitations

- The model is trained on the BUSI dataset and may not generalize to images from other hospitals, scanners, or populations.
- An image-level random split may produce optimistic results if related images or images from the same patient appear across different subsets.
- Dataset quality, class imbalance, image artifacts, and the number of available samples can affect performance.
- ImageNet pretrained weights were not specifically trained for medical ultrasound interpretation.
- A high confidence score does not guarantee a correct prediction.
- The model has not been clinically validated and is not suitable for real-world diagnosis.

## 13. Future Improvements

- Use patient-level splitting where patient identifiers are available.
- Investigate duplicate and near-duplicate images across data subsets.
- Compare MobileNetV2 with other architectures.
- Tune hyperparameters and apply controlled fine-tuning.
- Add explainability visualizations such as Grad-CAM.
- Evaluate performance on an independent external dataset.
- Improve reproducibility by recording dataset versions, package versions, and training configuration.

## 14. References

1. Al-Dhabyani, W., Gomaa, M., Khaled, H., and Fahmy, A. (2020). *Dataset of breast ultrasound images.* Data in Brief, 28, 104863. https://doi.org/10.1016/j.dib.2019.104863
2. Kaggle. *Breast Ultrasound Images Dataset (BUSI).* https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset
3. TensorFlow. *MobileNetV2 documentation.* https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
4. Scikit-learn. *Classification metrics documentation.* https://scikit-learn.org/stable/modules/model_evaluation.html

## 15. Disclaimer

This project was developed for educational and internship purposes to demonstrate image classification using deep learning. It is not a medical device and must not be used as a substitute for professional medical advice, diagnosis, or treatment.
