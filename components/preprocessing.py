### Combines dataframe with the target series
def concat_df_a_target(df,target):
    return pd.concat([df,target], axis=1)

### Drops all the nan features in a particular subset (which means particular column or a batch of columns)
def drop_Nans(train,subset):
    dropped_df = train.dropna(subset=subset).reset_index().drop('index',axis=1)
    return dropped_df

### Returns a list containing the columns that appear in the train set but not in the test set (of course the roles can also be reveres)
def feature_difference(train, test):
    # Get the set of column names from each DataFrame
    train_set = set(train.columns)
    test_set = set(test.columns)

    # find the difference in cols
    feature_difference_cols = train_set - test_set

    return feature_difference_cols

### Caps the outliers 
def cap_outliers(train, columns, method='iqr', threshold=1.5):
    '''
    Caps the outliers for a subset of columns because If I am not cartefull the function will alter the target column 
    input:
    train - the dataframe
    columns - subset of columns, in order to not cap unwanted columns
    returns: the capped dataframe
    '''

    train_copy = train.copy()
    
    for col in columns:
        if col != 'sii':
            Q1 = train[col].quantile(0.25)
            Q3 = train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            train_copy[col] = np.clip(train[col], lower_bound, upper_bound)
        
    return train_copy


# This function is not neccasry
def correct_outliers_dk(df):
    '''
    This function correct outliers based on domain knowledge and sense,
    for example heart_rate can not be zero or a very small number 
    '''
    train = df.copy()
    # Define thresholds
    bmi_threshold = 7
    weight_threshold = 35
    diastolic_bp_threshold = 35
    systolic_bp_threshold = 65
    heart_rate_threshold = 45

    # Correct the outliers
    train.loc[train['Physical-BMI'] <= bmi_threshold, 'Physical-BMI'] = bmi_threshold
    train.loc[train['Physical-Weight'] <= weight_threshold, 'Physical-Weight'] = weight_threshold
    train.loc[train['Physical-Diastolic_BP'] < diastolic_bp_threshold, 'Physical-Diastolic_BP'] = diastolic_bp_threshold
    train.loc[train['Physical-Systolic_BP'] < systolic_bp_threshold, 'Physical-Systolic_BP'] = systolic_bp_threshold
    train.loc[train['Physical-HeartRate'] < heart_rate_threshold, 'Physical-HeartRate'] = heart_rate_threshold
    swap_condition = train['Physical-Diastolic_BP'] > train['Physical-Systolic_BP']
    train.loc[swap_condition, ['Physical-Diastolic_BP', 'Physical-Systolic_BP']] = train.loc[swap_condition, ['Physical-Systolic_BP', 'Physical-Diastolic_BP']].values
    
    return train

### combines all the outlier methods in order to handle them 
def handle_outliers(train): 
    '''
    The following function handle the outliers, both the statistical domain knowledge. 
    The function receives from the user 
    train - the train dataframe set
    '''
    
    train_capper = cap_outliers(train,train.select_dtypes(include='number').columns)
    display(train_capper.describe())
    
    return train_capper

### Finds pairs columns with high correlation (higher then threshold)
def high_correlation_pairs(train, threshold=0.95):
    # Calculate the correlation matrix
    corr_matrix = train.select_dtypes(include='number').corr("pearson")
    
    # Select pairs of features with correlations above the threshold
    high_corr_pairs = (
        corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))  # Upper triangle without diagonal
        .stack()  # Convert to Series
        .reset_index()
    )
    
    # Rename columns for readability
    high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    # Filter by the correlation threshold (both positive and negative)
    high_corr_pairs = high_corr_pairs[high_corr_pairs['Correlation'].abs() > threshold]
    
    # Display the high correlation pairs
    print("Highly correlated feature pairs (|correlation| > {}):".format(threshold))
    print(high_corr_pairs.to_string(index=False))
    
    return high_corr_pairs


# Find features low correlation in regrad to the target (lower correlation than threshold)
def low_correlated_features(train, target_column, threshold=0.1):
    '''
    input:
    train - dataframe 
    target_column - the target column name
    threshold - the minimum correlation score which keeps the column in the final subset of features
    returns: 
    '''
    corr_matrix = train.corr()
    target_correlations = corr_matrix[target_column].abs()
    low_correlated_features = target_correlations[target_correlations < threshold].index
    
    return low_correlated_features

##### Feature Engineering Function
def feature_engineering(df,tst,Target_series):
    '''
        This function is used to clean the data and make sure it is ready for modeling
        input:
            train: the test dataframe
            test: the train dataframe
            
    '''
    train = df.copy()
    test= tst.copy()
    ### Drop unique columns for train dataframe (later add the target series as it was dropped)
    cols_diff = feature_difference(train, test) 
    train = train.drop(list(cols_diff),axis=1)
    train = concat_df_a_target(train, Target_series)
    
    display(train)
    ### Feature creation 
    #feature_creation(train,test)
    
    ### Handle missing values

    
    ### Drop high correlation pairs
    high_corr_pairs = high_correlation_pairs(train)
    # take the second feature from each pair and drop them from the dataframe
    features_to_remove = high_corr_pairs['Feature 2'].tolist()
    train = train.drop(features_to_remove, axis=1)
    test = test.drop(features_to_remove, axis=1)

    ### Deal with outliers
    train_outliers = handle_outliers(train)
    
    # Drop features with low correlation to target
    low_corr_cols = low_correlated_features(train_outliers.select_dtypes(include='number'),'sii')
    train = train.drop(low_corr_cols,axis=1)
    test = test.drop(low_corr_cols,axis=1)

    ### Drop categorical columns 
    cat_cols = train.select_dtypes(exclude='number').columns
    train = train.drop(cat_cols,axis=1)
    test = test.drop(cat_cols,axis=1)
    
    
    return train, test 
