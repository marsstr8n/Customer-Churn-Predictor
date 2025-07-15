# create a class for model train and evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt



class ModelEvaluator:
    def __init__(self, X_test, y_test, target_names, mode="console"):
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names
        self.mode = mode
        
    def evaluate(self, model):
        import matplotlib.pyplot as plt
        import streamlit as st

        y_pred = model.predict(self.X_test)
        report = classification_report(
            self.y_test,
            y_pred,
            target_names=self.target_names
        )
        
        if self.mode == "streamlit":
            
            st.subheader("Classification Report")
            st.code(report)
            
        else:
            print("Classification Report")
            print(report)

        # make confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # display
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.target_names)
        fig_cm, ax_cm = plt.subplots(figsize=(5,5))
        disp.plot(cmap='Blues', ax=ax_cm)
        
        if self.mode == "streamlit":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(fig_cm, use_container_width=False)
            
        else:
            print("Confusion Matrix")
            plt.show()


        # plotting ROC curve
        fig_roc, ax_roc = plt.subplots(figsize=(5,5))
        RocCurveDisplay.from_estimator(model, self.X_test, self.y_test, ax=ax_roc)
        
        if self.mode == "streamlit":
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.subheader("ROC Curve")
                st.pyplot(fig_roc)
            
        else:
            print("ROC Curve")
            plt.show()


        # get the AUC score
        y_probs = model.predict_proba(self.X_test)[:, 1]  
        roc_auc = roc_auc_score(self.y_test, y_probs)
        
        if self.mode=="streamlit":
            st.markdown(f"**ROC AUC Score:** `{roc_auc:.2f}`")
        else:
            print(f"ROC AUC Score: {roc_auc:.2f}") 
      
        
