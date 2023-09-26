import numpy as np
'''
accuracy = [40.17, 39.70, 40.85, 39.10, 39.52]
precision_mac = [0.03,0.03,0.03,0.03,0.03]
precision_we = [0.16, 0.16, 0.17, 0.15, 0.16]
recall_mac= [0.07, 0.08, 0.07, 0.07, 0.07]
recall_we = [0.40, 0.40, 0.41, 0.39, 0.40]
f1_mac = [0.04,0.04,0.04,0.04,0.04]
f1_we = [0.23,0.23,0.24,0.22,0.22]
'''
'''
accuracy = [81.68,81.31,82.85,80.42,80.61]
precision_mac = [0.68,0.69,0.71,0.72,0.72]
precision_we = [0.90,0.89,0.90,0.89,0.89]
recall_mac= [0.74,0.75,0.80,0.81,0.78]
recall_we = [0.82,0.81,0.83,0.80,0.81]
f1_mac = [0.62,0.62,0.67,0.67,0.67]
f1_we = [0.84,0.83,0.85,0.83,0.83]
'''
'''
accuracy = [97.7,97.75,97.65,97.54,97.88]
precision_mac = [0.87,0.93,0.91,0.92,0.91]
precision_we = [0.98,0.98,0.98,0.98,0.98]
recall_mac= [0.91,0.90,0.92,0.95,0.89]
recall_we = [0.98,0.98,0.98,0.98,0.98]
f1_mac = [0.87,0.91,0.91,0.93,0.89]
f1_we = [0.98,0.98,0.98,0.98,0.98]
'''

accuracy = []
precision_mac = []
precision_we = []
recall_mac= []
recall_we = []
f1_mac = []
f1_we = []

'''
accuracy = []
precision_mac = []
precision_we = []
recall_mac= []
recall_we = []
f1_mac = []
f1_we = []
'''

# Accuracy
print("Accuracy")
print(np.average(np.array(accuracy)))
print(np.std(np.array(accuracy)))

# Precision macro
print("Precision macro")
print(np.average(np.array(precision_mac)))
print(np.std(np.array(precision_mac)))

# Precision weighted
print("Precision weighted")
print(np.average(np.array(precision_we)))
print(np.std(np.array(precision_we)))

# Recall macro
print("Recall macro")
print(np.average(np.array(recall_mac)))
print(np.std(np.array(recall_mac)))

# Recall weighted
print("Recall weighted")
print(np.average(np.array(recall_we)))
print(np.std(np.array(recall_we)))

# F1 macro
print("F1 macro")
print(np.average(np.array(f1_mac)))
print(np.std(np.array(f1_mac)))

# F1 weighted
print("F1 weighted")
print(np.average(np.array(f1_we)))
print(np.std(np.array(f1_we)))