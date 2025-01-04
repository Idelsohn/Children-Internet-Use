# Children Internet Use 
## Goal
Develop a predictive model that analyzes children's physical activity and fitness data to identify early signs of problematic internet use.

## Dataset
Healthy Brain Network (HBN) dataset - consists of a clinical sample of about 3800 children who have undergone clinical and research screenings.
The dataset is compiled into two sources, parquet files containing the accelerometer (actigraphy) time-series and CSV files containing the remaining tabular data.
There are 80 unique features in the tabular data that are divided into 11 different categories which are detailed here:
![image](https://github.com/user-attachments/assets/8568f300-789f-4cbc-a0dc-d4111b09c750)

## Project Structure
- data				
    - processed	 	- location for processed files
      - train_processed.csv
      - test_processed.csv
    - raw	 	- location for parquet files containing raw data (train, test, sample submission)
      - data_dictionary.csv
      - sample_submission.csv
      - test.csv
      - train.csv
- notebooks - all of the notebooks that were used for the project
  - Kaggle_Submission.ipynb - the notebook that was used for submission in Kaggle ()
  - Problematic_Internet_Use_HyperTuning.ipynb - tuned hyperparameters based on the processed train set
  - Problematic_Internet_Use_EDA.ipynb
  - TimeSeries_EDA.ipynb 
  - submission.csv - the submission file consisting of the final prediction on the test set
  
## Solution
## EDA 
As detailed  
The data is extremely messy and has features that can have over 80% missing features, 
some features have extreme outliers that exceed the human limit (such as a BMI score of 0)
There are a lot of cases of Multicollinearity, that can cause overfitting
## The Model
For the final solution I have used an ensemble of model consist of XGBoost, LightGBM and CatBoost. After experimenting with each one alone I.
The models' performances were measured by the QWK metric (quadratic weighted kappa) Which measures the agreement between two outcomes. It typically varies from 0 (random agreement) to 1 (complete agreement). The metric is well suited for the task because of the significance of the error, classifying falsely over 2 categories is worse that 
Preprocessing - consists of: handling outliers by both percentile and domain knowledge capping, imputing missing values using KNNImputer, dropping Nan values in target feature, creating new features based on domain knowledge, and dropping highly correlated features and features with low correlation to target.
My best.

