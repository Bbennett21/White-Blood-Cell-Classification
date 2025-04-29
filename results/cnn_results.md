# Blood Cell Classification with CNN

## Model Performance

### Training Results
- **Loss**: 0.0255
- **Accuracy**: 0.9935 (99.35%)

### Validation Results
- **Loss**: 0.1291
- **Accuracy**: 0.9611 (96.11%)

### Test Results
- **Loss**: 0.1102
- **Accuracy**: 0.9676 (96.76%)

#### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| basophil     | 0.94      | 0.96   | 0.95     | 122     |
| eosinophil   | 0.99      | 0.99   | 0.99     | 296     |
| erythroblast | 0.98      | 0.90   | 0.94     | 143     |
| ig           | 0.89      | 0.94   | 0.91     | 284     |
| lymphocyte   | 0.93      | 0.95   | 0.94     | 132     |
| monocyte     | 0.93      | 0.95   | 0.94     | 153     |
| neutrophil   | 0.98      | 0.95   | 0.96     | 347     |
| platelet     | 1.00      | 1.00   | 1.00     | 223     |
| **accuracy** |           |        | 0.96     | 1700    |
| macro avg    | 0.96      | 0.95   | 0.95     | 1700    |
| weighted avg | 0.96      | 0.96   | 0.96     | 1700    |

## Key Findings

1. **Excellent performance** with overall test accuracy of 96.76%
2. **Strong generalization** demonstrated by similar validation and test accuracies
3. **Very low training loss** (0.0255) indicating good fit to training data
4. **Consistent performance** across validation and test sets suggests robust model