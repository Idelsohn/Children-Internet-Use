# Children Internet Use 
## Goal
find biological markers that will improve the diagnosis and treatment of mental health and learning disorders from an objective biological perspective.

## Data
Initial data Consists of 

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
Preprocessing - consists of: handling outliers by both percentile and domain knowledge capping, imputing missing values using KNNImputer, dropping Nan values in target feature, creating new features based on domain knowledge, and dropping highly correlated features and features with low correlation to target.
My best.

