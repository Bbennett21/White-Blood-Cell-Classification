# White Blood Cell Classification

## Introduction/Background
White blood cells (leukocytes) are critical components of the immune system, defending the body against infections and diseases. Medical professionals use blood smears to examine the morphology of white blood cells and obtain differential counts. This information is essential for diagnosing a wide range of conditions including infections, inflammatory disorders, leukemia, and immune system dysfunctions. Pathologists traditionally examine blood cells through visual inspection of blood smears using microscopes, manually identifying and classifying different types of white blood cells based on their morphological characteristics.

This traditional manual approach, while effective, is labor-intensive, time-consuming, and subject to variability from observer to observer, which can affect diagnostic accuracy and consistency. The increasing demand for hematological analyses in both routine screenings and specialized diagnostics has created a push for more efficient and standardized methodologies. Our goal is to automate this labor-intensive task by developing a system capable of accurately predicting different types of normal peripheral blood cells, thereby improving diagnostic efficiency and reducing the workload for healthcare professionals.

## Literature Review
Studies have explored machine learning techniques for medical image classification:
- Kandel, Castelli, and Popovič developed a convolutional neural network (CNN) to classify histopathology images, comparing six different optimizers and finding that adaptive ones worked best [1].
- Khan et al. used a CNN with a dual-attention network to detect white blood cells in blood smears, improving accuracy with a generative model [2].
- Bain established foundational principles for diagnosis from blood smears, emphasizing the importance of systematic morphological assessment and its clinical implications [3].
- Nazlibilek et al. investigated automatic segmentation and classification of white blood cells using image processing techniques, demonstrating early successes in computational approaches to cell identification [4].
- Hegde et al. developed algorithms specifically for white blood cell classification with a focus on acute lymphoblastic leukemia detection, showing the potential clinical applications of these technologies [5].
- Aksoy developed an innovative hybrid model for automatic detection of white blood cells in clinical laboratories, which our project draws inspiration from [6].

## Dataset
Our dataset consists of normal peripheral blood cell images taken from healthy individuals at the Hospital Clinic of Barcelona. The dataset contains a total of 17,092 images of individual normal cells, which were acquired using the analyzer CellaVision DM96 in the Core Laboratory at the Hospital Clinic of Barcelona. The dataset is available at: [Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset/data).

The dataset is organized in the following eight groups: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts and platelets (thrombocytes). The size of the images is 360 x 363 pixels, in format JPG, and they were annotated by expert clinical pathologists. The images were captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection.

![White Blood Cell Types](/supplementary_images/blood-cell-chart.png)
*Figure 1: Examples of the eight different white blood cell types in our dataset.*

This high-quality labelled dataset may be used to train and test machine learning and deep learning models to recognize different types of normal peripheral blood cells. To our knowledge, this is the first publicly available set with large numbers of normal peripheral blood cells, so that it is expected to be a canonical dataset for model benchmarking.

## Problem Statement
Manual inspection of White Blood Cells (WBCs) in blood smears is labor-intensive, subjective, and prone to human error. This traditional approach requires pathologists to visually examine hundreds of cells under a microscope, a process that typically takes 15-30 minutes per sample and demands sustained concentration to ensure accuracy. The subjective nature of visual assessment leads to inter-observer variability, with studies reporting disagreement rates of up to 30% among pathologists when classifying certain blood cell types. Furthermore, the growing volume of hematological tests in clinical settings has created a bottleneck in laboratory workflows, delaying diagnosis and treatment decisions. These challenges are particularly pronounced in resource-limited settings where expert pathologists may be scarce, potentially compromising patient care due to delayed or inaccurate cell differential counts.

## Motivation
An automated system can improve accuracy, efficiency, and consistency in white blood cell classification, aiding in early cancer detection and reducing pathologists' workload. By leveraging machine learning techniques, a system could analyze thousands of cells in minutes rather than hours, enabling faster turnaround times for critical diagnostic tests. Standardized classification algorithms would eliminate observer variability, ensuring consistent results regardless of operator expertise or fatigue. This consistency is particularly valuable for monitoring disease progression and treatment response over time. Furthermore, automation allows pathologists to focus their expertise on challenging cases that require human judgment, optimizing the allocation of specialized human resources. Early and accurate identification of abnormal white blood cell patterns can facilitate prompt intervention in conditions like leukemia, where early detection significantly improves prognosis. Additionally, the digital nature of automated systems creates opportunities for telemedicine applications, extending high-quality hematological diagnostics to underserved regions through remote analysis capabilities.

## Methods
The data is divided into training and testing sets with an 80-20 split to avoid model overfitting. Each set is further split into features and labels for supervised learning.

### Method I - Supervised Learning (CNN) [7] - COMPLETED
- Data is resized using bilinear interpolation to match the shape needed for the neural network
- Normalize data to ensure that all the images are within [0,1] range to help with consistency when comparing data in the model
- Split dataset into testing(80%), validation(10%), and training(10%) data 
- No need to find one-hot vectors because tensorflow converts labels into integer indices because we used image_dataset_from_directory()
- Pre-processed training, validation, and testing sets fed into a Convolutional Neural Network(CNN)
     - Designed to process images by extracting relevant features
     - Automatic feature extraction, good at learning hierarchical features from image data, allowing for efficient processing
     - Good at capturing high level patterns like shapes and structures which is an important feature when distinguishing between white blood cells


### Method II - Supervised Learning (SVM) [8] - COMPLETED
- Resize images and flatten them into a 1-D array.
- Normalize image pixels.
- Extract shape features using a histogram of oriented gradients (HOG) and color features using L*a*b color space.
- Features are concatenated into a single feature vector and processed through Principal Component Analysis (PCA) before being input into a Support Vector Machine (SVM).
- SVM is chosen due to its effectiveness in image classification tasks.
- Achieved 43.14% accuracy on the test set, indicating moderate performance.
- Performance analysis through precision-recall and ROC curves shows variability in classification quality across different cell types.

### Method III - Supervised Learning (Random Forest) [9] - COMPLETED
- Uses the same preprocessing and feature extraction steps as Method II.
- Features are fed into a Random Forest Classifier.
- Random Forest builds multiple decision trees and outputs the majority class.
- Selected for its robustness, flexibility, and interpretability.
- Achieved 86.32% accuracy on the test set, demonstrating strong performance.
- Feature importance analysis reveals which visual characteristics are most effective for classification.

## Results and Discussion

### Model 1 (CNN) Results

![CNN Accuracy and Loss](/output_figures/cnn/accuracy_loss_vs_epochs.png)
Figure 1: Training and validation accuracy/loss over epochs

#### Evaluating the model on all datasets

**Train Loss:** 0.0255  
**Train Accuracy:** 0.9935 (99.35%)

**Validation Loss:** 0.1291  
**Validation Accuracy:** 0.9611 (96.11%)

**Test Loss:** 0.1102  
**Test Accuracy:** 0.9676 (96.76%)

To measure the model's success, we wanted to achieve and accuracy of at least 85%. Based on the results, it seems that our model is pretty accurate. The model seems to generalize well when working with unseen validation data and unseen real-world testing data, espeically because the test accuracy is slightly higher than validation accuracy. However, the gap between trianing accuracy (99.35%) and validation(96.1%) and test (96.8%) accuracy is small which suggests that slight overfitting of the training data might be happening. If this model were to be remade, it might be helpful to implement regularization. 

![CNN Confusion Matrix](/output_figures/cnn/cnn_confusion_matrix.png)
Figure 2: Confusion matrix for the CNN model showing classification performance across cell types.

Sometimes accuracy cannot fully define the model's effectivness. So, the F1 score can be used to determine **precision**(how many predicted positives were correct) and **recall**(how many positives are correctly predicted) of the model. The confusion matrix can be helpful to find the F1 score. 

Precision = TP / (TP + FP)  
Recall = TP / (TP + FN)  
F1 = 2 * (Precision * Recall) / (Precision + Recall)

**F1 scores (CNN):**  
Basophil - 0.942  
Eosinophil - 0.992  
Erythroblast - 0.956  
Immature Granulocyte - 0.917  
Lymphocyte - 0.986  
Monocyte - 0.956  
Neutrophil - 0.963  
Platelet - 0.991

Weighted F1 score: sum(wi * F1i) = 0.963  
wi = total samples in class / total samples in dataset

Weight F1 score was high which indicates high model performance without major bias.

### Model 2 (SVM) Results

The Support Vector Machine classifier achieved moderate results with an overall accuracy of 43.14% on the test set. This demonstrates the challenges in distinguishing between certain white blood cell types using the extracted image features with SVM.

![PCA Variance](/output_figures/svm/pca_variance.png)
Figure 3: Principal Component Analysis variance showing cumulative explained variance by components.

![SVM Confusion Matrix](/output_figures/svm/svm_confusion_matrix.png)
Figure 4: Confusion matrix for the SVM model showing classification performance across cell types.

![ROC Curve](/output_figures/svm/roc_curve.png)
Figure 5: ROC curve for SVM classification showing model performance across different thresholds.

![Precision-Recall Curve](/output_figures/svm/precision_recall_curve.png)
Figure 6: Precision-Recall curve for SVM model showing the trade-off between precision and recall.

#### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| basophil     | 0.47      | 0.03   | 0.05     | 267     |
| eosinophil   | 0.32      | 0.36   | 0.34     | 623     |
| erythroblast | 0.54      | 0.51   | 0.52     | 286     |
| ig           | 0.30      | 0.39   | 0.34     | 545     |
| lymphocyte   | 0.59      | 0.37   | 0.45     | 251     |
| monocyte     | 0.46      | 0.21   | 0.29     | 313     |
| neutrophil   | 0.34      | 0.47   | 0.39     | 683     |
| platelet     | 0.83      | 0.91   | 0.87     | 451     |
| **accuracy** |           |        | 0.43     | 3419    |
| macro avg    | 0.48      | 0.40   | 0.41     | 3419    |
| weighted avg | 0.45      | 0.43   | 0.42     | 3419    |

Key findings from the SVM model:

1. **Moderate performance** with overall accuracy of 43.14%, which falls below our target threshold of 85%.
2. **Platelets** have the highest F1-score (0.87), making them the most reliably identified cell type.
3. **Basophils** have the lowest recall (0.03), indicating they are frequently misclassified.
4. **Class imbalance** is present, with neutrophils having the highest support (683 samples).
5. **Dimensionality reduction** through PCA shows that a subset of features captures most of the variance.

The model shows significantly lower performance compared to both CNN and Random Forest approaches, suggesting that SVM may not be the optimal classifier for this particular image classification task without more sophisticated feature engineering.

### Model 3 (Random Forest) Results

The Random Forest classifier achieved strong results across all cell types with an overall accuracy of 86.32% on the test set. This demonstrates the model's effectiveness at distinguishing between different white blood cell types using extracted image features.

![Cumulative Feature Importance](/output_figures/random_forest/cumulative_importance.png)
Figure 7: Cumulative feature importance showing how many features are needed to capture most of the signal.

![Feature Importance](/output_figures/random_forest/feature_importance.png)
Figure 8: Feature importance bar plot showing the most important features for classification.

![Sample Tree](/output_figures/random_forest/tree_simplified.png)
Figure 9: Sample tree of depth 2 from the random forest.

![Random Forest Confusion Matrix](/output_figures/random_forest/rf_confusion_matrix.png)
Figure 10: Confusion matrix for the Random Forest model showing classification performance across cell types.

Key findings from the Random Forest model:

To measure the Random Forest model's success, we wanted to achieve an accuracy of at least 85%. With an overall test accuracy of 86.32%, the model successfully meets our target threshold. The model demonstrates good generalization between validation (85.55%) and test sets (86.32%), indicating good performance on unseen data. Feature importance analysis revealed that a relatively small subset of features contributes most of the predictive power, with cumulative importance curves showing that approximately 1,500 features capture 95% of the signal. While the model performs exceptionally well on platelets (F1-score of 0.97), it struggles more with monocytes (recall of 0.63), suggesting potential areas for improvement in future iterations.

**F1 scores (Random Forest):**  
Basophil - 0.79  
Eosinophil - 0.88  
Erythroblast - 0.92  
Immature Granulocyte - 0.78  
Lymphocyte - 0.88  
Monocyte - 0.73  
Neutrophil - 0.90  
Platelet - 0.97

The model shows consistent performance between validation and test sets, suggesting good generalization to new data. The balanced performance across most cell types indicates the effectiveness of the extracted features in capturing the distinctive visual characteristics of each cell type.

## Comparison of Models

All three models (CNN, SVM, and Random Forest) were successfully implemented, with the CNN and Random Forest models exceeding our target accuracy of 85%. The SVM model performed significantly worse at 43.14% accuracy.

Key comparisons:
- CNN achieved the highest overall accuracy (96.76%), followed by Random Forest (86.32%), and SVM (43.14%)
- CNN excels at automatically learning hierarchical features from raw images
- Random Forest provides greater interpretability through feature importance analysis
- SVM struggled with several cell types, particularly basophils (recall of 0.03)
- Both CNN and Random Forest models identified platelets most reliably (F1 > 0.97)
- The significantly lower performance of SVM suggests that the extracted features or the classifier itself may not be well-suited for this particular classification task without further optimization

## Project Goals
- **Efficiency** – Automating blood cell classification reduces manual workload for pathologists.
- **Ethicality** – We'll check for bias in our model to ensure fair classification across all cell types.

## Gantt Chart
[GanttChart.xlsx](https://docs.google.com/spreadsheets/d/1t6ufq1hn6nNWufRAuDkOpr4TDCa34HMI/edit?usp=sharing)

## Contribution Table
| Name           | Contribution                                  |
|----------------|----------------------------------------------|
| Bryson Bennett | Model I (CNN), Model III (Random Forest) |
| Raj Bhat       | Model I (CNN)            |
| Daniel Houston | Model II (SVM)                               |
| Priyam Kadakia | Model III (Random Forest) |
| Savan Shah     | Model II (SVM)                               |

## GitHub Repository Structure
[Project Repository](https://github.gatech.edu/bbennett62/CS4641)

`/bloodcells_dataset/`  
Contains all images in our dataset, organized into subdirectories by blood cell classification:
- `/basophil/`
- `/eosinophil/`
- `/erythroblast/`
- `/ig/` (immature granulocyte)
- `/lymphocyte/`
- `/monocyte/`
- `/neutrophil/`
- `/platelet/`

`/output_figures/`  
Contains visualization outputs from model training and evaluation.

- `/cnn/`  
  - `accuracy_loss_vs_epochs.png` – Training and validation accuracy/loss over epochs  
  - `cnn_confusion_matrix.png` – Confusion matrix for CNN model performance

- `/random_forest/`  
  - `cumulative_importance.png` – Cumulative feature importance  
  - `feature_importance.png` – Feature importance bar plot  
  - `rf_confusion_matrix.png` – Confusion matrix for Random Forest model  
  - `tree_enhanced_depth_3.png` – Visualization of an enhanced decision tree  
  - `tree_simplified.png` – Visualization of a simplified decision tree  

- `/svm/`  
  - `pca_variance.png` – Principal Component Analysis variance explained  
  - `precision_recall_curve.png` – Precision-Recall curve for SVM model  
  - `roc_curve.png` – ROC curve for SVM classification  
  - `svm_confusion_matrix.png` – Confusion matrix for SVM model performance  

`/results/`  
Contains Markdown summaries of final model results:
- `cnn_results.md` – Detailed results for CNN model
- `random_forest_results.md` – Detailed results for Random Forest model
- `svm_results.md` – Detailed results for SVM model

`/src/`  
Contains all source code for implemented models:
- `cnn.py` – Convolutional Neural Network implementation  
- `random_forest.py` – Random Forest implementation  
- `svm.py` – Support Vector Machine implementation  

`/supplementary_images/`  
Contains additional images for documentation:
- `blood-cell-chart.png` – Visual representation of the different blood cell types in our dataset

`README.md`  
- Project overview and documentation

`_config.yml`  
- Configuration for the GitHub Pages website


## Citations
[1] Kandel, M. Castelli, and A. Popovič, "Comparative study of first order optimizers for image classification using convolutional neural networks on histopathology images," *J. Imaging*, vol. 6, no. 9, p. 92, Sep. 2020, doi: 10.3390/jimaging6090092.

[2] S. Khan, M. Sajjad, N. Abbas, J. Escorcia-Gutierrez, M. Gamarra, and K. Muhammad, "Efficient leukocytes detection and classification in microscopic blood images using convolutional neural network coupled with a dual attention network," *Comput. Biol. Med.*, vol. 174, p. 108146, 2024, doi: 10.1016/j.compbiomed.2024.108146.

[3] B. J. Bain, "Diagnosis from the blood smear," *New England Journal of Medicine*, vol. 353, no. 5, pp. 498-507, 2005.

[4] S. Nazlibilek, D. Karacor, T. Ercan, M. H. Sazli, O. Kalender, and Y. Ege, "Automatic segmentation, counting, size determination and classification of white blood cells," *Measurement*, vol. 55, pp. 58-65, 2014.

[5] R. B. Hegde, K. Prasad, H. Hebbar, and B. M. K. Singh, "Development of a robust algorithm for classification of white blood cells and acute lymphoblastic leukemia detection," *International Journal of Medical Engineering and Informatics*, vol. 10, no. 3, pp. 191-204, 2018.

[6] A. Aksoy, "An Innovative Hybrid Model for Automatic Detection of White Blood Cells in Clinical Laboratories," *Diagnostics* 2024, 14, 2093. https://doi.org/10.3390/diagnostics14182093

[7] "How to preprocess and train a CNN (step-by-step)," Kaggle. January 18, 2021. [Kaggle Link](https://www.kaggle.com/code/vesuvius13/how-to-preprocess-and-train-a-cnn-step-by-step).

[8] Basthikodi M, Chaithrashree M, Ahamed Shafeeq BM, Gurpur AP. "Enhancing multiclass brain tumor diagnosis using SVM and innovative feature extraction techniques." *Scientific Reports*, 2024;14(1). doi:10.1038/s41598-024-77243-7.

[9] "RandomForestClassifier." Scikit-learn. Accessed February 19, 2025. [Scikit-learn Link](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).