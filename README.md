# IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes
## Introduction: 
Welcome to my submission for the Data Science Specialization with IBM. This project, titled "Exchange Rate Prediction of Guatemalan Quetzal," aims to forecast the exchange rate from the Guatemalan Quetzal to the American Dollar for the upcoming week. Utilizing the skills and knowledge acquired throughout the course, I have developed a predictive model to achieve this goal.

This report will provide a comprehensive explanation of the project, detailing the methodology, implementation, and results. Additionally, a video presentation is included at the end to further illustrate the project's development and functionality.

## Why did I choose this project?
The reason I chose this project is quite personal and relevant to my everyday life. As a Guatemalan, I regularly deal with the Guatemalan Quetzal. Recently, I needed to exchange some of my Quetzales to American Dollars for an upcoming trip. The experience made me acutely aware of how exchange rate fluctuations could impact my travel budget. This practical concern inspired me to leverage the skills I gained in this Data Science Specialization to predict the exchange rate for the week of my travel. By developing a predictive model, I aimed to gain better insight into potential exchange rate movements, making my financial planning more precise and informed. This project, therefore, not only showcases my technical abilities but also addresses a real-world problem that I personally encountered.

### Code Explanation:
To begin, we import our necessary libraries. We use Pandas and Numpy for data manipulation, Scikit-learn for scaling and metrics, TensorFlow for building and training our neural network, Matplotlib for visualization, and PySpark for handling large datasets:
```Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pyspark.sql import SparkSession
```
Next, we initialize a Spark session. This helps in efficiently handling large datasets that might not fit into memory:
```Initialize Spark
```javascript I'm A tab
console.log('Code Tab A');
```
Then, we load and explore our data. The initial_data_exploration function reads the CSV file, filters out rows with missing dates, and selects the relevant columns:
```Data Exploration
def initial_data_exploration(data_path):
    df = spark.read.csv(data_path, header=True, inferSchema=True, sep=",", multiLine=True)
    df = df.filter(df['Fecha'].isNotNull())
    df = df.select('Fecha', 'TCR 1/').withColumnRenamed('TCR 1/', 'TCR_1')
    return df
```
In the etl_process function, we clean the data by dropping rows with missing values.
