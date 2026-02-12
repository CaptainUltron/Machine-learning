#Take any feature from your dataset. Observe the density pattern for that feature by plotting the 
#histogram. Use buckets (data in ranges) for histogram generation and study. Calculate the mean and 
#variance from the available data. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    data = pd.read_csv("LAB03/train.csv")
    feature_data = data.iloc[:, 0].to_numpy(dtype=float)  
    plt.hist(feature_data, bins=20, edgecolor='black')
    plt.title('Histogram of Feature 1')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.show()
    
    mean_value = np.mean(feature_data)
    variance_value = np.var(feature_data)
    
    print(f"Mean: {mean_value}")
    print(f"Variance: {variance_value}")

if __name__ == "__main__":
    main()