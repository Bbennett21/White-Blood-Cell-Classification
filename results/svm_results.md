# Blood Cell Classification with SVM

## Model Performance

### Results
- **Accuracy**: 0.4314

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

## Key Findings

1. **Moderate performance** with overall accuracy of 43.14%
2. **Platelets** have the highest F1-score (0.87), making them the most reliably identified cell type
3. **Basophils** have the lowest recall (0.03), indicating they are frequently misclassified
4. **Class imbalance** is present, with neutrophils having the highest support (683 samples)