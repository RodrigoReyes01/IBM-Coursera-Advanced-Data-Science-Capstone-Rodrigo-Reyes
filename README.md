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
``` Initialize Spark
spark = SparkSession.builder \
    .appName("Capstone Project") \
    .getOrCreate()
```
Then, we load and explore our data. The initial_data_exploration function reads the CSV file, filters out rows with missing dates, and selects the relevant columns:
``` Data Exploration
def initial_data_exploration(data_path):
    df = spark.read.csv(data_path, header=True, inferSchema=True, sep=",", multiLine=True)
    df = df.filter(df['Fecha'].isNotNull())
    df = df.select('Fecha', 'TCR 1/').withColumnRenamed('TCR 1/', 'TCR_1')
    return df
```
In the etl_process function, we clean the data by dropping rows with missing values.
``` ETL
def etl_process(df):
    df_clean = df.dropna()
    return df_clean
```
We then create features for our model in the feature_creation function. This involves converting the DataFrame to a Pandas DataFrame, scaling the TCR_1 values, creating lag features, and scaling these lag features:
``` Feature Creation
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
``` Define Model
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
``` Train model
def train_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history
```
The evaluate_model function evaluates the trained model on the test set:
``` Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
```
To predict the TCR_1 values for the next week, we use the predict_next_week function, which iteratively generates predictions for the next 7 days:
``` Predict next week
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
``` Plotting
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
``` Pipeline
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
