from yellowbrick.features import FeatureImportances
from yellowbrick.classifier import ConfusionMatrix, ClassificationReport, ROCAUC, DiscriminationThreshold
from matplotlib import plt

# Explain an ML model performance using multiple plots
def Explainable_ML(model, X_train, y_train):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    model = model
    model.importance_type = 'total_gain'

    visualgrid = [
        FeatureImportances(model,  ax=axes[0][0], colormap= 'winter'),
        #ConfusionMatrix(model, ax=axes[0][1], cmap= 'GnBu'),
        #ClassificationReport(model, ax=axes[1][0], cmap= 'GnBu'),
    ]

    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.finalize()

    plt.show()

# XGB_Model_fitted = XGB_Model.fit(train.drop(["sii"],axis=1), train["sii"])
# Explainable_ML(XGB_Model_fitted, train.drop(["sii"],axis=1), train["sii"])

# Create a plot of permutation importance score for a model
def plot_permutation_importance(model, X, y, title):
    n_top_features=15
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0)
    sorted_idx = perm_importance.importances_mean.argsort()[-n_top_features:]
    
    # Plotting permutation importance
    sns.barplot(x=perm_importance.importances_mean[sorted_idx], 
                y=X.columns[sorted_idx], palette="Blues_r")
    sns.set_title(f"{title} - Permutation Importance")
    sns.set_xlabel("Permutation Score")
  
    plt.tight_layout()
    plt.show()

#plot_permutation_importance(vote_model, X, y, "Voting Regressor", axs[0, 0])