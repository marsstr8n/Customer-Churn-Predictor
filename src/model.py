# model application - best one is Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def train_logistic_model(X_train, y_train, scoring='f1', cv=5, verbose=2, random_state=0):
    base_clf = LogisticRegression(solver='liblinear', max_iter=1000)

    param_grid = [
    {
        'solver': ['liblinear',],
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1],
    },
    # lbfgs and sag support l2 only
     {
        'solver': ['lbfgs',],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1],
    },
     
    {
        'solver': ['sag',],
        'penalty': ['l2'],
        'C': [0.01, 0.1, 1],
    },
    # saga with l1, l2
    {
        'solver': ['saga',],
        'penalty': ['l1','l2'],
        'C': [0.01, 0.1, 1],
    },
    # saga with elasticnet, required l1_ratio
    {
        'solver': ['saga',],
        'penalty': ['elasticnet'],
        'C': [0.01, 0.1, 1],
        'l1_ratio': [0.3, 0.5, 0.7]
    }
]

    grid_search = GridSearchCV(
        estimator=base_clf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        verbose=verbose,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_result = grid_search.best_params_, grid_search.best_score_
    best_params = grid_search.best_params_
    best_params["random_state"] = random_state

    best_clf = LogisticRegression(**best_params)

    best_clf.fit(X_train, y_train)
    
    return best_clf