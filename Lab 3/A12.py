import pandas as pd
import numpy as np
from A9 import predict
from A6 import split_data
from sklearn.metrics import confusion_matrix, classification_report, f1_score

def main():
    data = pd.read_csv("LAB03/train.csv")
    X = data.iloc[:, :-1].to_numpy(dtype=float)
    y = data.iloc[:, -1].to_numpy()

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)

    # for training data
    train_predictions = predict(X_train, y_train, X_train, 0, len(X_train), k=3)

    cm_train = confusion_matrix(y_train, train_predictions)
    print("Confusion Matrix (Training):\n", cm_train)

    print("\nClassification Report (Training):")
    print(classification_report(y_train, train_predictions))

    train_f1 = f1_score(y_train, train_predictions, average='weighted')

    # for testing data
    test_predictions = predict(X_train, y_train, X_test, 0, len(X_test), k=3)

    cm_test = confusion_matrix(y_test, test_predictions)
    print("Confusion Matrix (Testing):\n", cm_test)

    print("\nClassification Report (Testing):")
    print(classification_report(y_test, test_predictions))

    test_f1 = f1_score(y_test, test_predictions, average='weighted')

    # inference
    print("\nModel Learning Outcome:")

    if train_f1 < 0.7 and test_f1 < 0.7:
        print("Model is UNDERFITTING (low performance on both training and testing data).")

    elif train_f1 - test_f1 > 0.15:
        print("Model is OVERFITTING (high training performance but poor test performance).")

    else:
        print("Model is WELL-FITTED (good generalization with similar train and test performance).")


if __name__ == "__main__":
    main()
