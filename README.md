# Children Internet Use
## Competition Goal and Results
Goal: Develop a predictive model that analyzes children's physical activity and fitness data to identify early signs of problematic internet use.
</br>
The results were to my satisfaction as I managed to earn my first medal and was in the top 3% of all submissions (out of 3600 groups) 
![image](https://github.com/user-attachments/assets/6ad8d31f-5fe8-4f50-8917-861399525884)

## Dataset
Healthy Brain Network (HBN) dataset - consists of a clinical sample of about 3800 children who have undergone clinical and research screenings.
The dataset is compiled into two sources, parquet files containing the accelerometer (actigraphy) time series and CSV files containing the remaining tabular data.
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
  - Kaggle_Submission.ipynb - the notebook that was used for submission in Kaggle
  - Problematic_Internet_Use_HyperTuning.ipynb - tuned hyperparameters based on the processed train set
  - Problematic_Internet_Use_EDA.ipynb - The detailed EDA process
  - TimeSeries_EDA.ipynb - EDA + transformation process of the timeseries
  - submission.csv - file consisting the prediction on the test set
  
## Solution
## EDA 
My EDA process which is detailed in this [notebook](https://github.com/Idelsohn/Children-Internet-Use/blob/main/notebooks/Problematic_Internet_Use_EDA.ipynb)
The data is extremely messy and has features that can have over 80% missing features, 
some features have extreme outliers that exceed the human limit (such as a BMI score of 0)
and there are a lot of cases of Multicollinearity, which can cause overfitting. Therefore the first action after starting to work on the project is to clean the data and try to capture valuable insights. For example: the features BMI, Height and Weight have the biggest connection and correlation to the target which can be latter used in feature engineering.
![image](https://github.com/user-attachments/assets/6693eec8-a76a-4959-9f20-291a8ffdf22f)

## The Model
For the final solution, I used an ensemble of models consisting of XGBoost, LightGBM, and CatBoost. After experimenting with each on its own, I concluded that this composition provides the most robust solution.
The models' performances were measured by the QWK metric (quadratic weighted kappa) Which measures the agreement between two outcomes. It typically varies from 0 (random agreement) to 1 (complete agreement). The metric is well suited for the task because it takes into account the size of the error.
I have tuned the hyperparameters for each model individually using optuna in this [notebook](https://github.com/Idelsohn/Children-Internet-Use/blob/main/notebooks/Problematic_Interent_Use_HyperParametersTuning.ipynb).

## Prediction and Evaluation
To evaluate the results of the model I have used two techniques
![image](https://github.com/user-attachments/assets/5e6c4dbc-90ca-41ec-a22c-ec9bc80e8415)


