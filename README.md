# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
from scipy.stats import chi2_contingency
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import seaborn as sns

df = pd.read_csv("/content/bmi.csv")

df.head()

df = df.dropna()

max_val = np.max(np.abs(df[['Height', 'Weight']]))

sc = StandardScaler()
df[['Height', 'Weight']] = sc.fit_transform(df[['Height', 'Weight']])

df.head(10)

nm = Normalizer()
df[['Height', 'Weight']] = nm.fit_transform(df[['Height', 'Weight']])

df

mas = MaxAbsScaler()
df[['Height', 'Weight']] = mas.fit_transform(df[['Height', 'Weight']])

df

rs = RobustScaler()
df[['Height', 'Weight']] = rs.fit_transform(df[['Height', 'Weight']])

df.head(5)

tips = sns.load_dataset('tips')

tips.head()

contingency_table = pd.crosstab(tips['sex'], tips['time'])

contingency_table

chi2, p, _, _ = chi2_contingency(contingency_table)

print('Chi-square statistic:', chi2)
print('p-value:', p)

data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': ['A', 'B', 'C', 'A', 'B'],
    'Feature3': [0, 1, 1, 0, 1],
    'Target': [0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

df

x = df[['Feature1', 'Feature3']]
y = df['Target']

selector = SelectKBest(score_func=mutual_info_classif, k=1)

x_new = selector.fit_transform(x, y)

print('Selected features:', x_new)

selectedFeatureIndices = selector.get_support(indices=True)

selectedFeatures = x.columns[selectedFeatureIndices]

print('Selected features:', selectedFeatures)
```
<img width="345" height="219" alt="image" src="https://github.com/user-attachments/assets/0ea7fd3b-04a9-41ae-b5b1-44c36109bb23" />

<img width="355" height="462" alt="image" src="https://github.com/user-attachments/assets/76b8e99d-9aee-4a84-a269-7b56abeadc5f" />
<img width="115" height="32" alt="image" src="https://github.com/user-attachments/assets/143f8941-0f03-4098-b958-8326b593a755" />

<img width="393" height="399" alt="image" src="https://github.com/user-attachments/assets/7ad1dcd8-79b1-4c08-9735-975192239929" />

<img width="395" height="463" alt="image" src="https://github.com/user-attachments/assets/b19790e1-f326-44ad-a827-b86a7bc548ce" />

<img width="385" height="463" alt="image" src="https://github.com/user-attachments/assets/ade966b3-6615-429e-808b-6fc7e278df70" />

<img width="372" height="226" alt="image" src="https://github.com/user-attachments/assets/a534da47-3a26-421d-a2a2-ec2a50a9b0a3" />

<img width="527" height="222" alt="image" src="https://github.com/user-attachments/assets/75702827-00a5-4ddf-b28a-092edf9ecd00" />

<img width="248" height="148" alt="image" src="https://github.com/user-attachments/assets/f34b79a2-8875-4d88-b2e9-9c7f1ea5c7d0" />

<img width="399" height="47" alt="image" src="https://github.com/user-attachments/assets/bc9a1fab-21e2-4643-b973-52cbb49f82db" />

<img width="396" height="226" alt="image" src="https://github.com/user-attachments/assets/1c13ba0e-f5c9-4ece-9cf0-9e1d020be095" />

<img width="276" height="113" alt="image" src="https://github.com/user-attachments/assets/4373ee41-d43a-4efd-9ef1-ab13574970eb" />

<img width="544" height="43" alt="image" src="https://github.com/user-attachments/assets/3b5b5288-22dc-480c-be3d-d74f45049547" />

# RESULT:
       # INCLUDE YOUR RESULT HERE
