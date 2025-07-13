# create a class for model train and evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt



class ModelEvaluator:
    def __init__(self, X_test, y_test, target_names):
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names
        
    def evaluate(self, model):
        y_pred = model.predict(self.X_test)
        print(classification_report(
            self.y_test,
            y_pred,
            target_names=self.target_names
        ))

        # make confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_names)
        disp.plot(cmap='Blues')

        # plotting
        RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)

        # get the AUC score
        y_probs = model.predict_proba(self.X_test)[:, 1]  
        roc_auc = roc_auc_score(self.y_test, y_probs)
        print(f"ROC AUC Score: {roc_auc:.2f}") 
        plt.show()
        
