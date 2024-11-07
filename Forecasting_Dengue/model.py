import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
import seaborn as sns
import os
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

def mape(y_test, pred):
    y_test, pred = np.array(y_test), np.array(pred)
    return np.mean(np.abs((y_test - pred) / y_test))

class Model():
    def __init__(self, path):
        # Read Data
        if path.endswith("csv"):
            self.df_awal = pd.read_csv(path)  # CSV file
        else:
            self.df_awal = pd.read_excel(path)  # Excel file

        # Change Date format column to Month-Year
        self.df_awal['Date'] = pd.to_datetime(self.df_awal['Date'], format="%d.%m.%Y")
        self.df_awal['Date'] = self.df_awal['Date'].dt.to_period('M')

        # Extract date column into month and year
        self.df_awal['Month'] = self.df_awal['Date'].dt.month
        self.df_awal['Year'] = self.df_awal['Date'].dt.year

        # Set index column with date
        self.df = self.df_awal.set_index('Date')

        # Check and remove outliers
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        df_no_outlier = self.df[~((self.df < (Q1 - 1.5 * IQR)) | (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Feature Engineering: Add lag features
        df_no_outlier['Total_Cases_Lag1'] = df_no_outlier['Total_Cases'].shift(1)
        df_no_outlier['Total_Cases_Lag2'] = df_no_outlier['Total_Cases'].shift(2)
        df_no_outlier['Total_Cases_Lag3'] = df_no_outlier['Total_Cases'].shift(3)
        df_no_outlier.dropna(inplace=True)  # Drop rows with NaN after adding lag features

        # Splitting data
        X_train = df_no_outlier.iloc[:66, [0, 1, 2, 4, 5, 6]]  # Fitur termasuk lag
        y_train = df_no_outlier.iloc[:66, 3:4]  # Target variable
        self.X_test = df_no_outlier.iloc[66:, [0, 1, 2, 4, 5, 6]]  # Fitur untuk testing
        self.y_test = df_no_outlier.iloc[66:, 3:4]

        # Scaling with MinMaxScaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(self.X_test)

        # Random Forest Regressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train.values.ravel())
        self.predict = self.model.predict(X_test_scaled)

    def getDataResult(self):
        temp = pd.DataFrame(index=self.X_test.index)
        temp["Actual"] = self.y_test.values.flatten()
        temp["Predictions"] = self.predict.flatten()
        temp[['Actual', 'Predictions']].plot()
        plt.title("Actual vs Predictions")
        plt.xlabel("Date")
        plt.ylabel("Total Cases")
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/result.png"))
        plt.close()

    def pairplot(self):
        sns.pairplot(self.df)
        plt.title("Pair Plot of Features")
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/pairplot.png"))
        plt.close()

    def plotcase(self):
        plt.title('Total Cases Over Time')
        self.df['Total_Cases'].plot()
        plt.xlabel("Date")
        plt.ylabel("Total Cases")
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/plotcase.png"))
        plt.close()

    def corr_plot(self):
        sns.heatmap(self.df.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
        plt.title("Correlation Plot")
        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/corr.png"))
        plt.close()

    def data(self):
        self.df_awal["Date"] = self.df_awal["Date"].astype(str)
        return self.df_awal

    def evaluate(self):
        r2 = r2_score(self.y_test, self.predict)
        mae = mean_absolute_error(self.y_test, self.predict)
        rmse = sqrt(mean_squared_error(self.y_test, self.predict))
        mapes = mape(self.y_test, self.predict)
        return [r2, mae, mapes, rmse]

    def getModel(self):
        intercept = self.model.estimators_[0].feature_importances_
        return f"Model Coefficients: {intercept}"  # Update sesuai kebutuhan

    def movingaverage(self):
        ma = self.df['Total_Cases'].rolling(window=12, center=True, min_periods=6).mean()
        ax = self.df['Total_Cases'].plot(label='Total Cases', figsize=(10, 6))
        ma.plot(ax=ax, label='Moving Average', color='C1')
        plt.title('Total Cases and Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Total Cases')
        plt.legend()
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/movingaverage.png"))
        plt.close()

    def forecast(self):
        fourier = CalendarFourier(freq="A", order=10)
        dp = DeterministicProcess(index=self.df.index, constant=True, order=1, seasonal=True, additional_terms=[fourier], drop=True)
        X = dp.in_sample()
        y = self.df['Total_Cases']
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        y_pred = pd.Series(model.predict(X), index=y.index)
        X_fore = dp.out_of_sample(steps=24)
        self.hasilforecast = model.predict(X_fore)
        self.indexforecast = X_fore.index
        y_fore = pd.Series(self.hasilforecast, index=self.indexforecast)
        
        ax = y.plot(color='0.25', style='.', title="Predictions")
        y_pred.plot(ax=ax, label="Seasonal")
        y_fore.plot(ax=ax, label="Seasonal Forecast", color='C3')
        plt.savefig(os.path.join(os.getcwd(), "static/uploads/forecast.png"))
        plt.close()

    def forecastData(self):
        data = pd.DataFrame({"hasil": self.hasilforecast, "Date": self.indexforecast.astype(str)})
        return data

# Contoh penggunaan
# model = Model('path_to_your_data_file.csv')
# model.getDataResult()
# model.pairplot()
# model.plotcase()
# model.corr_plot()
# model.movingaverage()  # Tambahkan ini jika ingin memanggil metode movingaverage
# evaluations = model.evaluate()
# print(f"R2: {evaluations[0]}, MAE: {evaluations[1]}, MAPE: {evaluations[2]}, RMSE: {evaluations[3]}")
