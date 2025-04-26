# Blood Cell Classification with Random Forest

## Model Performance

### Validation Set Results
- **Accuracy**: 0.8555

#### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| basophil     | 0.84      | 0.71   | 0.77     | 122     |
| eosinophil   | 0.80      | 0.90   | 0.84     | 311     |
| erythroblast | 0.93      | 0.79   | 0.85     | 155     |
| ig           | 0.74      | 0.86   | 0.79     | 289     |
| lymphocyte   | 0.92      | 0.82   | 0.87     | 122     |
| monocyte     | 0.92      | 0.63   | 0.74     | 142     |
| neutrophil   | 0.89      | 0.92   | 0.90     | 333     |
| platelet     | 0.97      | 0.97   | 0.97     | 235     |
| **accuracy** |           |        | 0.86     | 1709    |
| macro avg    | 0.88      | 0.83   | 0.84     | 1709    |
| weighted avg | 0.86      | 0.86   | 0.85     | 1709    |

### Test Set Results
- **Accuracy**: 0.8632

#### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| basophil     | 0.95      | 0.67   | 0.79     | 122     |
| eosinophil   | 0.84      | 0.91   | 0.88     | 312     |
| erythroblast | 0.96      | 0.87   | 0.92     | 155     |
| ig           | 0.72      | 0.84   | 0.78     | 290     |
| lymphocyte   | 0.88      | 0.88   | 0.88     | 121     |
| monocyte     | 0.85      | 0.63   | 0.73     | 142     |
| neutrophil   | 0.88      | 0.92   | 0.90     | 333     |
| platelet     | 0.97      | 0.97   | 0.97     | 235     |
| **accuracy** |           |        | 0.86     | 1710    |
| macro avg    | 0.88      | 0.84   | 0.85     | 1710    |
| weighted avg | 0.87      | 0.86   | 0.86     | 1710    |

## Key Findings

1. **Strong performance** across all cell types with overall accuracy of 86.3% on the test set
2. **Platelets** have the highest F1-score (0.97), making them the most reliably identified cell type
3. **Monocytes** have the lowest recall (0.63), indicating they are often misclassified
4. **Consistent performance** between validation and test sets suggests good generalization