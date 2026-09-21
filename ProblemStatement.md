## Problem Statement

Electrocardiogram signals are used to monitor heart activity. Today detecting anomalies in ECG is hard because of the below reasons.

1. ECG signals vary between patients.
2. Abnormal samples are less common than normal samples.
3. ECG recordings may contain noise and missing values.
4. Different models capture different signal properties.

Different model detect anomalies under different conditions. User may need to implement and test each model separately. 
This makes it difficult to understand which model is most suitable for a particular ECG anomaly-detection task

## Solution

A Unified Dashboard will be built and shown the model specific anomaly scores, classify the signals and visualize suspicious regions and provide evaluation metrics for different models.
