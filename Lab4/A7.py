import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def main():
    data = pd.read_csv("train.csv")
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier()

    param_grid = {'n_neighbors': list(range(1, 31))}
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    grid_best_k = grid.best_params_['n_neighbors']
    grid_pred = grid.best_estimator_.predict(X_test)
    grid_acc = accuracy_score(y_test, grid_pred)

    param_dist = {'n_neighbors': list(range(1, 31))}
    random = RandomizedSearchCV(knn,param_distributions=param_dist,n_iter=10,cv=5,random_state=42,scoring='accuracy')
    random.fit(X_train, y_train)

    random_best_k = random.best_params_['n_neighbors']
    random_pred = random.best_estimator_.predict(X_test)
    random_acc = accuracy_score(y_test, random_pred)

    print("GridSearchCV:")
    print("Best k =", grid_best_k)
    print("Accuracy =", grid_acc)

    print("\nRandomizedSearchCV:")
    print("Best k =", random_best_k)
    print("Accuracy =", random_acc)


if __name__ == "__main__":
    main()
