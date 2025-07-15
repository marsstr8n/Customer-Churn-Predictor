import shap
import matplotlib.pyplot as plt
import pandas as pd

def shap_explainer(model, X_train, X_test):
    """Compute and plot SHAP plots - using LinearExplainer
    
    Red vs Blue (SHAP Color Meaning):
    - RED: Feature pushed prediction toward positive class (churn = 1)
    - BLUE: Feature pushed prediction toward negative class (churn = 0)
    - SHAP values near zero mean low impact.

    Args:
        model (_type_): Trained model
        X_train (df): training set
        X_test (df): test set
    """
    # instantiate LinearExplainer
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    
    # global feature importance - bar plot
    bar = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    save_bar = "plots/bar_summary.png"
    bar.savefig(save_bar, bbox_inches='tight')
    plt.close(bar)
    

    # force plot (local) for individual prediction explanation
    shap.initjs()
    fp = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0]) # analyses the first row in X_test, see how each feature contributes to the predicted outcome
    shap.save_html("plots/force_plot_1.html", fp)

    # dynamic
    fp_multi = shap.force_plot(explainer.expected_value, shap_values[:100], X_test.iloc[:100])   # this one is for 100 values in test set
    shap.save_html("plots/force_plot_100.html", fp_multi)


    # decision plot - static plot
    decision = plt.figure()
    shap.decision_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    save_decision = "plots/decision_plot.png"
    decision.savefig(save_decision, bbox_inches='tight')
    plt.close(decision)


    # shap waterfall plot - static plot
    waterfall = plt.figure()
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], X_test.iloc[0])
    save_waterfall = "plots/waterfall_plot.png"
    waterfall.savefig(save_waterfall, bbox_inches='tight')
    plt.close(waterfall)
    
    # bee swarm plot - static plot
    bee = plt.figure()
    shap.summary_plot(shap_values, X_test)
    save_bee = "plots/bee_swarm.png"
    bee.savefig(save_bee, bbox_inches='tight') # expand the border
    plt.close(bee)
    
