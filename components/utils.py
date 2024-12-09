import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Find Kbest categorical features based on the chi2 formula 
def Kbest(cat_dummies):
    # Ten features with highest chi-squared statistics are selected
    chi2_features = SelectKBest(chi2, k=10)
    X_kbest_features = chi2_features.fit_transform(cat_dummies.drop("sii",axis=1),cat_dummies["sii"])
    
    feature_names = cat_dummies.drop("sii", axis=1).columns
    
    scores = chi2_features.scores_
    
    # Create a DataFrame to visualize the results
    feature_scores = pd.DataFrame({'Feature': feature_names, 'Score': scores})
    
    # Sort the DataFrame by scores in descending order
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)
    
    # Print the top 10 features
    print(feature_scores.head(10))
    # Reduced features
    print('Original feature number:', cat_dummies.shape[1])
    print('Reduced feature number:', X_kbest_features.shape[1])

    # Return the dataframe with only the columns that have the highest chi2 square correlation
    return cat_dummies[list(feature_scores[0:10].Feature)].reset_index().drop('index',axis=1)

# check the correlation of every feature to the target column
def corr_to_target(train, target_column, threshold=0.1):
    corr_matrix = train.corr()
    target_correlations = corr_matrix[target_column].abs()
    return target_correlations
