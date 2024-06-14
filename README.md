# IBM-Coursera-Advanced-Data-Science-Capstone
### Exchange Rate Prediction of Guatemalan Quetzal
#### By: Rodrigo Reyes
Table Of Contents:
* Introduction
* Why this Project
* How to Run the Code
* Code Explanation
* Results
* Presantation
* Video


## Introduction: 
Welcome to my submission for the Data Science Specialization with IBM. This project, titled "Exchange Rate Prediction of Guatemalan Quetzal," aims to forecast the exchange rate from the Guatemalan Quetzal to the American Dollar for the upcoming week. Utilizing the skills and knowledge acquired throughout the course, I have developed a predictive model to achieve this goal.

This report will provide a comprehensive explanation of the project, detailing the methodology, implementation, and results. Additionally, a video presentation is included at the end to further illustrate the project's development and functionality.

## Why did I choose this project?
The reason I chose this project is quite personal and relevant to my everyday life. As a Guatemalan, I regularly deal with the Guatemalan Quetzal. Recently, I needed to exchange some of my Quetzales to American Dollars for an upcoming trip. The experience made me acutely aware of how exchange rate fluctuations could impact my travel budget. This practical concern inspired me to leverage the skills I gained in this Data Science Specialization to predict the exchange rate for the week of my travel. By developing a predictive model, I aimed to gain better insight into potential exchange rate movements, making my financial planning more precise and informed. This project, therefore, not only showcases my technical abilities but also addresses a real-world problem that I personally encountered.

## How to Run the code
This code was Done in google colab, so you can copy the code, then you should download the CSV file and copy the path to the file under the:
```py
if __name__ == "__main__":
    data_path = "/content/historico_rango.csv"  # Change the path to the uploaded file
```
If you did this correctly you should be able to run the code succesfuly.

## Code Explanation:
To begin, we import our necessary libraries. We use Pandas and Numpy for data manipulation, Scikit-learn for scaling and metrics, TensorFlow for building and training our neural network, Matplotlib for visualization, and PySpark for handling large datasets:
```py
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
```py
spark = SparkSession.builder \
    .appName("Capstone Project") \
    .getOrCreate()
```
Then, we load and explore our data. The initial_data_exploration function reads the CSV file, filters out rows with missing dates, and selects the relevant columns:
```py
def initial_data_exploration(data_path):
    df = spark.read.csv(data_path, header=True, inferSchema=True, sep=",", multiLine=True)
    df = df.filter(df['Fecha'].isNotNull())
    df = df.select('Fecha', 'TCR 1/').withColumnRenamed('TCR 1/', 'TCR_1')
    return df
```
In the etl_process function, we clean the data by dropping rows with missing values.
```py
def etl_process(df):
    df_clean = df.dropna()
    return df_clean
```
We then create features for our model in the feature_creation function. This involves converting the DataFrame to a Pandas DataFrame, scaling the TCR_1 values, creating lag features, and scaling these lag features:
```py
def feature_creation(df):
    pd_df = df.toPandas()
    pd_df['Fecha'] = pd.to_datetime(pd_df['Fecha'], format='%d/%m/%Y')

    # Scale TCR_1
    tcr_scaler = StandardScaler()
    pd_df['TCR_1_scaled'] = tcr_scaler.fit_transform(pd_df[['TCR_1']])

    # Create lag features
    for lag in range(1, 8):
        pd_df[f'TCR_1_lag_{lag}'] = pd_df['TCR_1_scaled'].shift(lag)

    pd_df.dropna(inplace=True)

    # Scale lag features
    numerical_cols = [f'TCR_1_lag_{lag}' for lag in range(1, 8)]
    scaler = StandardScaler()
    pd_df[numerical_cols] = scaler.fit_transform(pd_df[numerical_cols])

    return pd_df, scaler, tcr_scaler
```
The define_model function defines a neural network model using TensorFlow's Keras API:
```py
def define_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mean_squared_error'])
    return model
```
We train the model using the train_model function, which also validates the model on a validation set:
```py
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history
```
The evaluate_model function evaluates the trained model on the test set:
```py
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
```
To predict the TCR_1 values for the next week, we use the predict_next_week function, which iteratively generates predictions for the next 7 days:
```py
def predict_next_week(model, last_week_data, scaler, feature_columns, tcr_scaler):
    predictions = []
    for _ in range(7):
        # Ensure last_week_data has the correct shape (2D array)
        last_week_df = pd.DataFrame(last_week_data, columns=feature_columns)
        scaled_data = scaler.transform(last_week_df)
        next_prediction = model.predict(scaled_data)

        # Descale the prediction
        next_prediction_descaled = tcr_scaler.inverse_transform(next_prediction)
        predictions.append(next_prediction_descaled[0, 0])

        # Create new input by shifting last week data and adding the new prediction
        last_week_data = np.roll(last_week_data, -1)
        last_week_data[-1] = next_prediction[0, 0]

    return predictions
```
We also include functions to plot the data and the predictions:
```py
# Function to plot the data
def plot_data(processed_data, next_week_predictions, next_week_dates):
    plt.figure(figsize=(15, 10))

    # Plot all original data
    plt.subplot(2, 2, 1)
    plt.plot(processed_data['Fecha'], processed_data['TCR_1'], label='Original Data')
    plt.title('All Original Data')
    plt.xlabel('Date')
    plt.ylabel('TCR_1')
    plt.legend()

    # Plot last week's data
    plt.subplot(2, 2, 2)
    plt.plot(processed_data['Fecha'].tail(7), processed_data['TCR_1'].tail(7), label='Last Week')
    plt.title('Last Week Data')
    plt.xlabel('Date')
    plt.ylabel('TCR_1')
    plt.legend()

    # Plot next week's predictions
    plt.subplot(2, 2, 3)
    plt.plot(next_week_dates, next_week_predictions, label='Predictions', color='orange')
    plt.title('Next Week Predictions')
    plt.xlabel('Date')
    plt.ylabel('TCR_1')
    plt.legend()

    # Plot all data including predictions
    plt.subplot(2, 2, 4)
    plt.plot(processed_data['Fecha'], processed_data['TCR_1'], label='Original Data')
    plt.plot(next_week_dates, next_week_predictions, label='Predictions', color='orange')
    plt.title('All Data Including Predictions')
    plt.xlabel('Date')
    plt.ylabel('TCR_1')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to plot scatter data
def plot_scatter_data(processed_data, next_week_predictions, next_week_dates):
    plt.figure(figsize=(10, 6))

    # Scatter plot of all data including predictions
    plt.scatter(processed_data['Fecha'], processed_data['TCR_1'], label='Original Data', alpha=0.6)
    plt.scatter(next_week_dates, next_week_predictions, label='Predictions', color='orange', alpha=0.6)
    plt.title('All Data Including Predictions (Scatter Plot)')
    plt.xlabel('Date')
    plt.ylabel('TCR_1')
    plt.legend()

    plt.show()
```
To run the pipeline, we load the data, process it, train the model, evaluate it, and make predictions for the next week:
```py
if __name__ == "__main__":
    data_path = "/content/historico_rango.csv"  # Change the path to the uploaded file

    df = initial_data_exploration(data_path)
    df_clean = etl_process(df)
    processed_data, scaler, tcr_scaler = feature_creation(df_clean)

    # Debugging: Print processed_data to check the columns and data
    print("Processed Data:")
    print(processed_data.tail(10))  # Print the last 10 rows for inspection

    target_column = 'TCR_1_scaled'
    feature_columns = [col for col in processed_data.columns if col.startswith('TCR_1_lag')]

    X = processed_data[feature_columns]
    y = processed_data[target_column]

    print(f"Shapes of X and y: {X.shape}, {y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Shapes of X_train, X_val, y_train, y_val: {X_train.shape}, {X_val.shape}, {y_train.shape}, {y_val.shape}")

    input_shape = X_train.shape[1]
    model = define_model(input_shape)
    train_model(model, X_train, y_train, X_val, y_val)
    evaluate_model(model, X_val, y_val)

    # Save the model
    model.save("model.h5")
    print("Model saved as model.h5")

    # Predicting the next week's TCR_1
    last_week_data = processed_data[feature_columns].tail(7).values

    # Ensure last_week_data is a 2D array
    if last_week_data.shape[0] != 7:
        last_week_data = np.tile(last_week_data, (7, 1))

    print(f"Shape of last_week_data before prediction: {last_week_data.shape}")

    next_week_predictions = predict_next_week(model, last_week_data, scaler, feature_columns, tcr_scaler)
    next_week_dates = pd.date_range(start=processed_data['Fecha'].max() + pd.Timedelta(days=1), periods=7)

    # Displaying the prediction
    print("Next week's predictions:")
    for date, pred in zip(next_week_dates, next_week_predictions):
        print(f"Date: {date.date()}, Predicted TCR_1: {pred}")

    # Plotting the data
    plot_data(processed_data, next_week_predictions, next_week_dates)

    # Plotting scatter data
    plot_scatter_data(processed_data, next_week_predictions, next_week_dates)
```
## Results:
### So how does it look like?
Processed Data:
This table shows the historical data we used. Each row represents a day, and it includes the exchange rate (TCR_1) for that day along with some related information.
```yaml
          Fecha    TCR_1  TCR_1_scaled  TCR_1_lag_1  TCR_1_lag_2  TCR_1_lag_3  \
1615 2024-06-03  7.76374      0.004101     0.062953     0.063503     0.064045   
1616 2024-06-04  7.76850      0.081951     0.000503     0.063503     0.064045   
1617 2024-06-05  7.76761      0.067395     0.078320     0.001069     0.064045   
1618 2024-06-06  7.76844      0.080969     0.063770     0.078866     0.001627   
1619 2024-06-07  7.76638      0.047278     0.077339     0.064320     0.079404   
1620 2024-06-08  7.76638      0.047278     0.043662     0.077885     0.064862   
1621 2024-06-09  7.76638      0.047278     0.043662     0.044217     0.078423   
1622 2024-06-10  7.76241     -0.017651     0.043662     0.044217     0.044764   
1623 2024-06-11  7.76323     -0.004240    -0.021240     0.044217     0.044764   
1624 2024-06-12  7.76210     -0.022721    -0.007835    -0.020668     0.044764   

      TCR_1_lag_4  TCR_1_lag_5  TCR_1_lag_6  TCR_1_lag_7  
1615     0.089619     0.126122     0.063252     0.024438  
1616     0.064626     0.090194     0.126758     0.063932  
1617     0.064626     0.065207     0.090842     0.127416  
1618     0.064626     0.065207     0.065864     0.091513  
1619     0.002225     0.065207     0.065864     0.066543  
1620     0.079981     0.002823     0.065864     0.066543  
1621     0.065443     0.080559     0.003501     0.066543  
1622     0.079001     0.066024     0.081210     0.004202  
1623     0.045351     0.079579     0.066680     0.081884  
1624     0.045351     0.045937     0.080230     0.067359  
```
* Fecha: The date of the observation.
* TCR_1: The actual TCR_1 values.
* TCR_1_scaled: The scaled TCR_1 values, adjusted for training the neural network.
* TCR_1_lag_1 to TCR_1_lag_7: The TCR_1 values from the previous 1 to 7 days, used as features for training the model.

Shapes of X and y:
X is like a list of all the factors we use to predict (TCR_1), and y is the actual values we want to predict. Here, X has 1618 rows (days) and 7 columns (different factors from previous days), and y has 1618 values.
```scss
(1618, 7), (1618,)
```
* X: The feature matrix with 1618 rows (days) and 7 columns (lagged features).
* y: The target vector with 1618 values (TCR_1).

Shapes of X_train, X_val, y_train, y_val:
These are just smaller sets of the data that the computer uses to learn. X_train and y_train are what the computer learns from, and X_val and y_val are used to see how well it's learning.
(1294, 7), (324, 7), (1294,), (324,)
* X_train and y_train: The training set with 1294 samples and 7 features each.
* X_val and y_val: The validation set with 324 samples and 7 features each.

Model Training:
This part shows how the computer learns from the data. It's like when you practice something to get better at it. The computer tries to predict the exchange rate (TCR_1) and gets feedback on how good or bad its predictions are after each try (epoch).
```bash
Epoch 1/20
41/41 [==============================] - 1s 8ms/step - loss: 0.3665 - mean_squared_error: 0.3665 - val_loss: 0.0429 - val_mean_squared_error: 0.0429
Epoch 2/20
41/41 [==============================] - 0s 3ms/step - loss: 0.1198 - mean_squared_error: 0.1198 - val_loss: 0.0380 - val_mean_squared_error: 0.0380
Epoch 3/20
```
* Training and Validation Loss: Indicates the performance of the model on the training and validation datasets. Lower values suggest better performance.
* Epochs: The training process ran for 20 epochs, with loss and mean squared error metrics recorded for each epoch.

Mean Squared Error:
This number tells us how close the computer's predictions are to the actual exchange rates. A lower number means it's making better predictions.
```
0.07824650451860782
```
Mean Squared Error (MSE): A metric used to evaluate the model's performance. A lower MSE indicates better predictive accuracy.

Next week's predictions:
These are the computer's guesses for what the exchange rate will be over the next week. It uses what it learned from the historical data to make these predictions.
```yaml
Date: 2024-06-13, Predicted TCR_1: 7.7677130699157715
Date: 2024-06-14, Predicted TCR_1: 7.767287254333496
Date: 2024-06-15, Predicted TCR_1: 7.767825603485107
Date: 2024-06-16, Predicted TCR_1: 7.7689690589904785
Date: 2024-06-17, Predicted TCR_1: 7.768828392028809
Date: 2024-06-18, Predicted TCR_1: 7.769726753234863
Date: 2024-06-19, Predicted TCR_1: 7.768404483795166
```
Predictions: The model's forecasted TCR_1 values for the next seven days, starting from June 13, 2024, to June 19, 2024.

### Plots:
Plots would be generated to visually compare the historical data and the predicted values, providing a visual representation of the model's forecasting performance.

All Data:


![AllOriginalData](https://github.com/RodrigoReyes01/IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes/assets/71049819/85575774-9542-4de9-af72-afc8df514a77)


In this plot we can see all the Data that is in the Data Set.

Last Week:


![LastWeekData](https://github.com/RodrigoReyes01/IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes/assets/71049819/fd6b185b-08b0-446b-b1a6-cb1e07f33e72)


In this plot we show what last weeks TCR looked like more closely.

Next Week:


![NextWeekPredictions](https://github.com/RodrigoReyes01/IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes/assets/71049819/288e3dd7-a56e-47b2-9b6c-6cef97b51fb2)


In this plot we see how would our prediction look.

All Data Including Prediction:


![AllDataIncluded](https://github.com/RodrigoReyes01/IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes/assets/71049819/59cd37ae-2462-4418-b326-92a93e1d109a)


In this plot we can see now how all the past data looks like with the new data added in, you can see that it makes sense and looks somewhat acurrate.


![ScatterPlot](https://github.com/RodrigoReyes01/IBM-Coursera-Advanced-Data-Science-Capstone-Rodrigo-Reyes/assets/71049819/c9c2d8bb-6e10-4ac3-be66-22c1fb7011b4)


In this one you can have a closer look at where are the Dots Plotted.

## Presentation:
### Stakeholder:

https://docs.google.com/presentation/d/1j8afXtTIPqyfbLENv59swPRmIIqc1AeX/edit?usp=sharing&ouid=114712119144930753335&rtpof=true&sd=true

### Data Science Peers:

https://docs.google.com/presentation/d/1RL2xWkM2jJaW-r-jukcTP3UfY14TTrpX/edit?usp=sharing&ouid=114712119144930753335&rtpof=true&sd=true

### Video Presentation:

https://docs.google.com/presentation/d/1ufCJ5iJ6-ofTq2qDLjIxQ6wUi9u-0tn4/edit?usp=sharing&ouid=114712119144930753335&rtpof=true&sd=true

## Video

