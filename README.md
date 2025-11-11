# [Python] Data Analysis

🗓️ **Date**: 2024.09.05 ~ 2024.12.18

<br/>

📊 **Objective**
- Distinguish several feature engineering methods according to the data type and purpose to improve the performance of predictive models.
- Identify and practice the application of basic machine learning algorithms for predictive analytics.
- Experiment different aspects of data and predictive analytics using actual data generated from real world industrial scenarios.
<br/>

🧩 **Table of Contents**
|Num|Content|
|---|-------|
|01|[Python Basics](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_01.ipynb)|
|02|[Data Visualization](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_02.ipynb)|
|03|[Time Series Data](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_03.ipynb)|
|04|[Exploratory Data Analysis (EDA)](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_04.ipynb)|
|05|[Data Preprocessing](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_05.ipynb)|
|06|[Machine Learning Basics](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_06.ipynb)|
|07|[Project (1)](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_Project_1.ipynb)|
||[Project (2)](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_Project_2.ipynb)|
||[Project (3)](https://github.com/git-jihyunpark/Data-Analysis/blob/main/DA_Project_3.ipynb)|

<br/><br/>



## 🔷 Final Project: Predict Energy Consumption

📌 **Introduction**
- Analyze the correlation between variables affecting peak demand power in the manufacturing process and identify the variables influencing peak demand power.
- Predict energy consumption using various algorithms.
<br/><br/>


📂 **Dataset**
- Korea Electric Power Corporation (KEPCO) imposes electricity fees for a year based on the peak power usage during a three-month period in manufacturing facilities. Therefore, unintentional peak power surges in the manufacturing process lead to increased electricity costs, subsequently raising overall manufacturing expenses.
    - Peak power refers to the accumulated power measured over set time intervals through transformers installed in the factory.
- If fees are based on the maximum power used at specific moments—such as during equipment failures or the operation of large-capacity equipment—this can result in unfair charges for users. Therefore, an average power calculation over a designated period (15-minute intervals for KEPCO) is used to determine peak power.
<br/><br/>


### 1. Preprocessing
**1) Data Overview**
- column: total of 18
    - integer type: `날짜[date]`, `시간[hour]`, `15분[min_15]`, `30분[min_30]`, `45분[min_45]`, `60분[min_60]`, `평균[average]`, `생산량[production]`, `day`, `d`, `m`
    - float type: `기온[temperature]`, `풍속[wind_speed]`, `습도[humidity]`, `강수량[precipitation]`, `전기요금[electricity_cost]`, `공정직원[factory_staff]`, `인건비[labor_cost]`
- row: total of 6,168 
```python
df = pd.read_csv('./okm_augumented_2021.csv')
df.info()
```
<img width="338" height="393" alt="image" src="https://github.com/user-attachments/assets/789ef073-d6e7-4245-ac03-3b1ff1e3a096" /> <br/>



**2) Missing Value Check**
- The dataset has missing valuses
    - `풍속(wind_speed)`: 3 missing values
    - `강수량(precipitation)`: 1 missing values
    - `공정직원(factory_staff)`: 17 missing values
- All missing values replaced with 0.
```python
df.isnull().sum()
```
<img width="173" height="302" alt="image" src="https://github.com/user-attachments/assets/1b82c09d-08af-4020-8bf9-f5bb6ce0c496" /> <br/>


**3) Data Preprocessing**
- All column names changed to English. <br/>
```python
# Change Column Name to English
df_c.rename(columns={
    '날짜': 'date',
    '시간': 'hour',
    '15분': 'min_15',
    '30분': 'min_30',
    '45분': 'min_45',
    '60분': 'min_60',
    '평균': 'average',
    '생산량': 'production',
    '기온': 'temperature',
    '풍속': 'wind_speed',
    '습도': 'humidity',
    '강수량': 'precipitation',
    '전기요금(계절)': 'electricity_cost',
    'day': 'day',
    'd': 'd',
    'm': 'm',
    '공장인원': 'factory_staff',
    '인건비': 'labor_cost'
}, inplace=True)
```
<br/>
<br/>


**4) Selection of Features**
1. `전기요금[electricity_cost]` : Selected as a feature to examine its relationship with seasons, as there are three distinct electricity cost levels.
2. `날짜[date]`, `시간[hour]` : Selected as features to investigate differences in peak-time electricity usage distribution by month, time of day, and weekday/weekend.
3. `시간대별 피트타임 전력사용량[min15, min30, min45, min60]` : Selected as a feature to identify outliers through scaling and to analyze distributions by morning/afternoon and by season.
4. `전력생산량[production]` : Shows a long-tail distribution with most data concentrated near 0. Selected as a feature to analyze seasonal distributions after scaling and to evaluate production efficiency (productivity relative to cost).
5. `기온[temperature]` : Correlation analysis revealed a relationship between temperature and peak-time electricity usage. Selected as a feature to perform k-means binning and analyze peak-time electricity usage by temperature range.
6. `공장직원[factory_staff]` : Shows a long-tail distribution with most data concentrated near 0. Selected as a feature to examine monthly, time-based, and production efficiency (production relative to staff count) distributions after scaling.
7. `인건비[labor_cost]`: Selected as a feature to identify optimal distributions by analyzing variations in data distribution across time periods day/night.

<br/>
<br/>



### 2. EDA
**1) Box Plot**
- The median is around 100 for all peak electricity usage.
- The IQR is similar across all peak electricity usage.
- There are no outliers.
- There is no significant difference in peak power usage across different time intervals.
<img width="725" height="606" alt="image" src="https://github.com/user-attachments/assets/2e8f2b7a-1c9f-4a17-ba3c-0d4a3bcb2667" /> <br/><br/>


**2) Heatmap**
- Variables like 2 and 3 can be utilized for resource optimization.
    - Positive Correlation: `Peaks Usage (0.9)`, `Electricity Cost and Temperature (0.81)`, `Production and Factory Staff (0.79)`
    - Negative Correlation: `Peaks Usage and Day of the week (-0.43)`
- However, the only variable that showed a strong correlation with electricity consumption was the usage by time interval.
<img width="723" height="616" alt="image" src="https://github.com/user-attachments/assets/4a6c7fa0-aaaa-4e78-92ff-2b2d00ec7e4b" /> <br/><br/>


**3-1) Line Chart: Average of Peak Electricity Usage by Month**
- January, May, and August has decreased.
<img width="837" height="556" alt="image" src="https://github.com/user-attachments/assets/af6250e6-7591-417b-a83b-fef2c2d57251" /> <br/><br/>


**3-2) Line Chart: Average of Peak Electricity Usage for Weekday and Weekend**
- Weekends is lower in all months except January.
<img width="860" height="556" alt="image" src="https://github.com/user-attachments/assets/a58fe8a7-1fe8-4f1d-9882-a1ef1cba849b" /> <br/><br/>


**3-3) Line Chart: Average of Peak Electricity Usage for Morning and Afternoon**
- Morning is lower in all months except February.
<img width="860" height="556" alt="image" src="https://github.com/user-attachments/assets/77df9c4e-e60c-4bf7-b4bc-5d2f6125bc08" /> <br/><br/>


**3-4) Line Chart: Average of Peak Electricity Usage by Season**
- There is a clear tendency to increase power usage in the summer
<img width="784" height="584" alt="image" src="https://github.com/user-attachments/assets/9aefd3bd-0f8b-467c-a62b-c356512eae48" /> <br/><br/>



### 3. Prediction
**1) Select Model**
- For prediction, three basic models (Regression Tree, Logistic Regression, and Random Forest) and two advanced models suitable for time series or sequence data (Prophet and RNN) were used.

| **Model** | **Regression Tree** | **Logistic Regression** | **Random Forest** | **Prophet** | **RNN** |
|------------|----------------------|--------------------------|-------------------|--------------|----------|
| **Pipeline()** | O | O | O | O | X |
| **Cross Validation** | K-fold | K-fold | K-fold | TimeSplit | TimeSplit |

<br/><br/>


**2) Modeling**
- Average value was used to observe the overall trend of usage across different time intervals.
- `Weekend` includes Friday
- Train/test split = 0.8:0.2
- Scaler : Min-max Scaler, Standard Scaler, Robust Scaler, Polynomial Features, Spline Transformer

| **Model** | **Input (X)** | **Target (y)** |
|------------|----------------|----------------|
| **Regression Tree** | Temperature, Wind Speed, Humidity, Precipitation, Day, Month, Time, Weekend, Holiday | Average consumption |
| **Logistic Regression** | Temperature, Wind Speed, Humidity, Precipitation, Day, Month, Time, Weekend, Holiday | Average consumption |
| **Random Forest** | Temperature, Wind Speed, Humidity, Precipitation, Day, Month, Time, Weekend, Holiday | Average consumption |
| **Prophet** | Average consumption [t-n], Weekend, Holiday | Average consumption |
| **RNN** | Average consumption [t-n], Weekend, Holiday | Average consumption [t] |

<br/><br/>



### 4. Evaluation

**1) Regression Tree**
- Robust Scaler: RMSE=23.06
<br/>
<img width="853" height="556" alt="image" src="https://github.com/user-attachments/assets/93f30f6b-7b18-46e5-9ce1-120ca1491535" /> <br/>
<img width="853" height="556" alt="image" src="https://github.com/user-attachments/assets/a4e05701-b81b-4c9b-a244-af2ec732ca72" /> <br/>
<br/><br/>


**2) Logistic Regression**
- Min-max Scaler: RMSE=23.09
<br/>
<img width="862" height="556" alt="image" src="https://github.com/user-attachments/assets/b7bd87db-4b60-4879-8370-2979655b1d1a" /> <br/>
<img width="853" height="556" alt="image" src="https://github.com/user-attachments/assets/929757dd-7220-4031-8979-becc186a9ff3" /> <br/>
<br/><br/>


**3) Random Forest**
- Min-max Scaler: RMSE=22.4
<br/>
<img width="877" height="556" alt="image" src="https://github.com/user-attachments/assets/f4f835a8-c72b-4afe-a59b-e3cf6001bd8b" /> <br/>
<img width="696" height="479" alt="image" src="https://github.com/user-attachments/assets/46275adb-36aa-4e09-be37-4a3e66d1acdd" /> <br/>
<br/><br/>


**4) Prophet**
- TimeSeriesSplit=5
    - K-fold is not a good way in Timeseries data.
    - So we tried ‘TimeSeriesSplit’.
- The RMSE values for each fold across all the scalers were the same.
    - RMSE=44.36
    - It is difficult to consider the performance as good. Because the prediction error is high.
<br/>
<img width="711" height="502" alt="image" src="https://github.com/user-attachments/assets/9959ec7e-a0dc-4ecb-887f-33347ffdcc7d" /> <br/>
<img width="1184" height="584" alt="image" src="https://github.com/user-attachments/assets/3abaa4c0-918f-47a7-aa2b-eca73e4312dc" /> <br/>
<br/><br/>


**5) RNN**
- TimeSeriesSplit=5
    - It did not work as expected.
    - As a result, only a single RMSE value was calculated per scaler.
- Robust Scaler: RMSE=9.32
    - It predicts the test data well, especially on weekends. 
<br/>
<img width="868" height="556" alt="image" src="https://github.com/user-attachments/assets/d94360a6-4df1-4ebd-bffa-d21939b0238f" /> <br/>
<img width="1635" height="875" alt="image" src="https://github.com/user-attachments/assets/975bad57-5e5e-49ba-826e-608b04df16b6" /> <br/>
<br/><br/>



### 5. Conclusion
- The best performance model : Random Forest(Robust scaler)
    - Electricity usage can be predicted by time and weather variables.


---


💖 **Lesson & Learn**
1. Proficiency in using the `Python` Language
   > numpy, pandas, sklearn, seaborn, matplot
2. Understanding and Development Skills in Machine Learning 
   > Logistic Regression, Random Forest, Random Forest, Prophet, RNN 
3. Training Predictive Models According to the Type and Purpose of Industrial Data 
   > Feature Selection, Scaler, Evaluation 
4. Data Visualization Skills
   > Box plot, Heatmap, Line Chart
