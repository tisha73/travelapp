# import os
# import pandas as pd
# import numpy as np
# import pymysql
# import optuna  # For hyperparameter tuning

# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split, RandomizedSearchCV


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# import xgboost as xgb
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping

# import joblib

# # Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DB_URL")
# engine = create_engine( DB_URL)

# # Load Data
# df_car_rent = pd.read_sql("SELECT * FROM car_rent", engine)
# df_user = pd.read_sql("SELECT * FROM passenger", engine)

# df_user.rename(columns={'usercode': 'User_ID'}, inplace=True)
# # df = pd.merge(df_user, df_car_rent, left_on='correct_column_name', right_on='correct_column_name', how='inner')

# df = pd.merge(df_user, df_car_rent, on='User_ID', how='inner')
# # df.drop(columns=['name'], inplace=True)

# df['month'] = df['Rent_Date'].dt.month

# df.drop(['User_ID', 'TravelCode'], axis=1, inplace=True)

# X = df.drop(columns = ['Total_Rent_Price'])
# y = df['Total_Rent_Price']

# categorical_cols = df.select_dtypes(include=['object']).columns
# numerical_cols = df.select_dtypes(include=['number']).columns
# # Define transformers
# num_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# cat_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('encoder', OneHotEncoder(drop='first', sparse_output=False))
# ])

# # Preprocessing pipeline
# preprocessor = ColumnTransformer([
#     ('num', num_transformer, numerical_cols),
#     ('cat', cat_transformer, categorical_cols)
# ])

# # Feature selection pipeline
# feature_selector = Pipeline([
#     ('var_thresh', VarianceThreshold(threshold=0.01))
# ])

# # Full pipeline
# full_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('feature_selection', feature_selector)
# ])

# # # Apply transformations
# # df_processed = full_pipeline.fit_transform(df)

# # # Extract feature names for encoded categorical variables
# # encoded_feature_names = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
# # all_feature_names = np.concatenate([numerical_cols, encoded_feature_names])

# # Apply transformations
# df_processed = full_pipeline.fit_transform(df)

# # Convert transformed data back to DataFrame
# cat_transformer = full_pipeline.named_steps['preprocessor'].named_transformers_['cat']
# encoded_feature_names = cat_transformer.get_feature_names_out(categorical_cols)
# # encoded_feature_names = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
# all_feature_names = np.concatenate([numerical_cols, encoded_feature_names])
# df_processed = pd.DataFrame(df_processed, columns=[f"feature_{i}" for i in range(df_processed.shape[1])])
# # df_processed = pd.DataFrame(df_processed, columns=all_feature_names)

# # Remove highly correlated features
# corr_matrix = df_processed.corr().abs()
# upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]

# df_processed.drop(columns=to_drop, inplace=True)
# print(f"Removed highly correlated features: {to_drop}")

# # Define features and target variable
# target_col = 'Total_Rent_Price'  # Change this accordingly
# X = df_processed.drop(columns=[target_col], errors='ignore')
# y = df[target_col] if target_col in df else None

# # print("Preprocessing complete!")

# # # Convert transformed data back to DataFrame
# # df_processed = pd.DataFrame(df_processed, columns=all_feature_names)

# # print("Preprocessing complete!")
# # # Preprocessing Pipeline
# # num_features = X.select_dtypes(include=['int64', 'float64']).columns
# # cat_features = X.select_dtypes(include=['object']).columns

# # num_transformer = Pipeline([
# #     ('imputer', SimpleImputer(strategy='median')),
# #     ('scaler', StandardScaler())
# # ])

# # cat_transformer = Pipeline([
# #     ('imputer', SimpleImputer(strategy='most_frequent')),
# #     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# # ])

# # preprocessor = ColumnTransformer([
# #     ('num', num_transformer, num_features),
# #     ('cat', cat_transformer, cat_features)
# # ])

# # # Feature selection pipeline
# # feature_selector = Pipeline([
# #     ('var_thresh', VarianceThreshold(threshold=0.01))  # Remove low-variance features
# # ])

# # # Full pipeline
# # full_pipeline = Pipeline([
# #     ('preprocessor', preprocessor),
# #     ('feature_selection', feature_selector)
# # ])

# # # Apply transformations
# # df_processed = full_pipeline.fit_transform(df)

# # # Convert transformed data back to DataFrame
# # cat_transformer = full_pipeline.named_steps['preprocessor'].named_transformers_['cat']
# # encoded_feature_names = cat_transformer.get_feature_names_out(categorical_cols)
# # # encoded_feature_names = full_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
# # all_feature_names = np.concatenate([numerical_cols, encoded_feature_names])
# # df_processed = pd.DataFrame(df_processed, columns=all_feature_names)

# # # Remove highly correlated features
# # corr_matrix = df_processed.corr().abs()
# # upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# # to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]

# # df_processed.drop(columns=to_drop, inplace=True)
# # print(f"Removed highly correlated features: {to_drop}")

# # # Define features and target variable
# # target_col = 'rental_price'  # Change this accordingly
# # X = df_processed.drop(columns=[target_col], errors='ignore')
# # y = df[target_col] if target_col in df else None

# # print("Preprocessing complete!")
# # 4.6 Split & Scale Data*


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Use RobustScaler to handle outliers better than StandardScaler
# scaler = RobustScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# ### *5️⃣ Optimize Model Training*
# #### *5.1 Hyperparameter Tuning for Random Forest*
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# rf_model = RandomForestRegressor(random_state=42)
# grid_search = RandomizedSearchCV(rf_model, param_grid, n_iter=10, cv=3, scoring='r2', n_jobs=-1)
# grid_search.fit(X_train_scaled, y_train)

# rf_best = grid_search.best_estimator_
# print("Best RF Model:", grid_search.best_params_)


# #### *5.2 Hyperparameter Tuning for XGBoost*
# # xgb_model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, n_estimators=300, max_depth=5)
# # # xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], early_stopping_rounds=10, verbose=False)
# # xgb_model.fit(
# #     X_train_scaled, y_train, 
# #     eval_set=[(X_test_scaled, y_test)], 
# #     eval_metric="rmse",  # Change this depending on your task
# #     early_stopping_rounds=10, 
# #     verbose=False
# # )
# dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
# dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# params = {
#     "objective": "reg:squarederror",  # Use "reg:squarederror" for regression
#     "eval_metric": "logloss",
#     "max_depth": 10,
# }

# xgb_model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=300,
#     evals=[(dtest, "test")],
#     early_stopping_rounds=10
# )

# #### *5.3 Optimized Neural Network with Early Stopping*
# nn_model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
#     Dropout(0.2),  # Dropout to prevent overfitting
#     Dense(64, activation='relu'),
#     Dropout(0.2),
#     Dense(1)
# ])

# nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# nn_model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stop], verbose=1)




# ### *6️⃣ Evaluate Models (With MAE, MSE, R²)*

# def evaluate_model(model, X_test, y_test, model_name):
#     # y_pred = model.predict(X_test)
#     test = xgb.DMatrix(X_test)  # Convert to DMatrix
#     # y_pred = model.predict(X_test_scaled)
#     if isinstance(model, xgb.Booster):  # Check if it's an XGBoost Booster model
#         dtest = xgb.DMatrix(X_test_scaled)  # Convert test data to DMatrix
#         y_pred = model.predict(dtest)
#     else:
#      y_pred = model.predict(X_test_scaled)  # Use NumPy array for other models
#     mse = mean_squared_error(y_test, y_pred)
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     print(f"🔹 {model_name} Performance:")
#     print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
#     print(f"✅ Mean Squared Error (MSE): {mse:.4f}")
#     print(f"✅ R² Score: {r2:.4f}\n")

# # Evaluate optimized models
# evaluate_model(rf_best, X_test_scaled, y_test, "Optimized Random Forest")
# evaluate_model(xgb_model, X_test_scaled, y_test, "Optimized XGBoost")

# # Evaluate Neural Network separately
# y_pred_nn = nn_model.predict(X_test_scaled).flatten()
# mse_nn = mean_squared_error(y_test, y_pred_nn)
# mae_nn = mean_absolute_error(y_test, y_pred_nn)
# r2_nn = r2_score(y_test, y_pred_nn)
#  new attempt
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Load environment variables
load_dotenv()
DB_URL = os.getenv("DB_URL")
engine = create_engine(DB_URL)

# Load Data
df_car_rental = pd.read_sql("SELECT * FROM car_rent", engine)
df_user = pd.read_sql("SELECT * FROM passenger", engine)

df_user.rename(columns={'usercode': 'User_ID'}, inplace=True)
df = pd.merge(df_user, df_car_rental, on='User_ID', how='inner')
df.drop(columns=['name'], inplace=True)

# Feature Engineering
df['Rent_Date'] = pd.to_datetime(df['Rent_Date'])
df['month'] = df['Rent_Date'].dt.month
df['day_of_week'] = df['Rent_Date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df.drop(['Rent_Date', 'User_ID', 'TravelCode', 'Pickup_Location', 'Dropoff_Location', 'Rent_ID'], axis=1, inplace=True)

# Outlier handling using IQR
Q1 = df['Total_Rent_Price'].quantile(0.25)
Q3 = df['Total_Rent_Price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Total_Rent_Price'] >= (Q1 - 1.5 * IQR)) & (df['Total_Rent_Price'] <= (Q3 + 1.5 * IQR))]

# Define Features and Target
X = df.drop(columns=['Total_Rent_Price'])
y = df['Total_Rent_Price']

# Preprocessing Pipeline
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

# Train-Test Split
X_transformed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Hyperparameter Optimization for XGBoost
def objective(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 800, 1500, step=100),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
        max_depth=trial.suggest_int('max_depth', 5, 12),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        gamma=trial.suggest_float('gamma', 0, 5),
        reg_alpha=trial.suggest_float('reg_alpha', 0, 5),
        reg_lambda=trial.suggest_float('reg_lambda', 0, 5),
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=75)
best_params = study.best_params

# Train Best XGBoost Model
best_xgb = XGBRegressor(**best_params)
best_xgb.fit(X_train, y_train)
y_pred_xgb = best_xgb.predict(X_test)

# Optimized Neural Network Model
nn_model = Sequential([
    Dense(256, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dense(1)
])

nn_model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
nn_model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=1)

y_pred_nn = nn_model.predict(X_test).flatten()

# Model Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")

evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
evaluate_model("Optimized Neural Network", y_test, y_pred_nn)
