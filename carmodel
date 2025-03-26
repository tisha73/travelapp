# # import os
# # import tensorflow as tf
# # import pandas as pd
# # import numpy as np
# # import random
# # from dotenv import load_dotenv
# # from sqlalchemy import create_engine
# # from sklearn.pipeline import Pipeline
# # from sklearn.impute import SimpleImputer
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# # from xgboost import XGBRegressor
# # from xgboost import callback
# # import optuna
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# # from tensorflow.keras.optimizers import Adam
# # from tensorflow.keras.regularizers import l2
# # from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# # from tensorflow.keras.losses import Huber
# # from tensorflow.keras.layers import LeakyReLU
# # from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # # Set random seed for reproducibility
# # seed = 42
# # random.seed(seed)
# # np.random.seed(seed)
# # tf.random.set_seed(seed)

# # # Load environment variables
# # load_dotenv()
# # DB_URL = os.getenv("DB_URL")
# # engine = create_engine(DB_URL)

# # # Load Data
# # df_car_rental = pd.read_sql("SELECT * FROM car_rent", engine)
# # df_user = pd.read_sql("SELECT * FROM passenger", engine)

# # df_user.rename(columns={'usercode': 'User_ID'}, inplace=True)
# # df = pd.merge(df_user, df_car_rental, on='User_ID', how='inner')
# # df.drop(columns=['name'], inplace=True)

# # df['month'] = pd.to_datetime(df['Rent_Date']).dt.month
# # df['day_of_week'] = pd.to_datetime(df['Rent_Date']).dt.dayofweek
# # df.drop(['Rent_Date'], axis=1, inplace=True)
# # df.drop(['User_ID', 'TravelCode', 'Pickup_Location', 'Dropoff_Location', 'Rent_ID'], axis=1, inplace=True)

# # # Feature Engineering
# # # df['flight_date'] = pd.to_datetime(df['departure_date'])
# # # df['is_weekend_flight'] = (df['flight_date'].dt.weekday >= 5).astype(int)
# # # df['flight_month'] = df['flight_date'].dt.month
# # # df.drop(columns=['flight_date'], inplace=True)

# # X = df.drop(columns=['Total_Rent_Price'])
# # y = df['Total_Rent_Price']

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

# # # Train-Test Split
# # X_transformed = preprocessor.fit_transform(X)
# # X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# # # Hyperparameter Optimization
# # def objective(trial):
# #     model = XGBRegressor(
# #         n_estimators=trial.suggest_int('n_estimators', 200, 1000, step=100),
# #         learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
# #         max_depth=trial.suggest_int('max_depth', 3, 15),
# #         subsample=trial.suggest_float('subsample', 0.6, 1.0),
# #         colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
# #         gamma=trial.suggest_float('gamma', 0, 10),
# #         reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
# #         reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
# #         random_state=42
# #     )
# #     # early_stopping = callback.EarlyStopping(rounds=10, save_best=True, metric_name="rmse")

# #     # # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping])

# #     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
# #     y_pred = model.predict(X_test)
# #     return mean_squared_error(y_test, y_pred)

# # study = optuna.create_study(direction='minimize')
# # study.optimize(objective, n_trials=20)
# # best_params = study.best_params

# # # Train Best Model
# # best_xgb = XGBRegressor(**best_params)
# # best_xgb.fit(X_train, y_train)
# # y_pred_xgb = best_xgb.predict(X_test)
# # y_xgb = best_xgb.predict(X_train)
# # # Stacking Model
# # stacked_model = StackingRegressor(
# #     estimators=[
# #         ('xgb', best_xgb),
# #         ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
# #         ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
# #     ],
# #     final_estimator=XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
# # )
# # stacked_model.fit(X_train, y_train)
# # y_pred_stack = stacked_model.predict(X_test)
# # y_stack = stacked_model.predict(X_train)
# # # Neural Network Model
# # def build_nn():
# #     model = Sequential([
# #         Dense(256, kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
# #         LeakyReLU(alpha=0.1),
# #         BatchNormalization(),
# #         Dropout(0.4),

# #         Dense(128, kernel_regularizer=l2(0.001)),
# #         LeakyReLU(alpha=0.1),
# #         BatchNormalization(),
# #         Dropout(0.4),

# #         Dense(64, kernel_regularizer=l2(0.001)),
# #         LeakyReLU(alpha=0.1),
# #         BatchNormalization(),
# #         Dropout(0.3),

# #         Dense(1)
# #     ])
    
# #     model.compile(optimizer=Adam(learning_rate=0.003), loss=Huber(delta=1.0), metrics=['mae'])
# #     return model

# # nn_model = build_nn()
# # nn_callbacks = [
# #     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
# #     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # ]

# # history = nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
# #                         epochs=200, batch_size=32, verbose=1, callbacks=nn_callbacks)

# # y_pred_nn = nn_model.predict(X_test).flatten()
# # y_nn = nn_model.predict(X_train).flatten()
# # # Model Evaluation
# # def evaluate_model(name, y_true, y_pred):
# #     r2 = r2_score(y_true, y_pred)
# #     mae = mean_absolute_error(y_true, y_pred)
# #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# #     print(f"\n{name} Performance:")
# #     print(f"R² Score: {r2:.4f}")
# #     print(f"MAE: {mae:.4f}")
# #     print(f"RMSE: {rmse:.4f}")


# # evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
# # evaluate_model("Stacking Model", y_test, y_pred_stack)
# # evaluate_model("Neural Network", y_test, y_pred_nn) 
# # evaluate_model("Training xgb",y_train,y_xgb)
# # evaluate_model("Training stacking model",y_train,y_stack)
# # evaluate_model("Training nn",y_train,y_nn)
# import os
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import random
# from dotenv import load_dotenv
# from sqlalchemy import create_engine
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
# from xgboost import XGBRegressor
# from xgboost import callback
# import optuna
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.losses import Huber
# from tensorflow.keras.layers import LeakyReLU
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Set random seed for reproducibility
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# # Load environment variables
# load_dotenv()
# DB_URL = os.getenv("DB_URL")
# engine = create_engine(DB_URL)

# # Load Data
# df_car_rental = pd.read_sql("SELECT * FROM car_rent", engine)
# df_user = pd.read_sql("SELECT * FROM passenger", engine)

# df_user.rename(columns={'usercode': 'User_ID'}, inplace=True)
# df = pd.merge(df_user, df_car_rental, on='User_ID', how='inner')
# df.drop(columns=['name'], inplace=True)

# df['month'] = pd.to_datetime(df['Rent_Date']).dt.month
# df['day_of_week'] = pd.to_datetime(df['Rent_Date']).dt.dayofweek
# df.drop(['Rent_Date'], axis=1, inplace=True)
# df.drop(['User_ID', 'TravelCode', 'Pickup_Location', 'Dropoff_Location', 'Rent_ID'], axis=1, inplace=True)

# # Feature Engineering
# # df['flight_date'] = pd.to_datetime(df['departure_date'])
# # df['is_weekend_flight'] = (df['flight_date'].dt.weekday >= 5).astype(int)
# # df['flight_month'] = df['flight_date'].dt.month
# # df.drop(columns=['flight_date'], inplace=True)

# X = df.drop(columns=['Total_Rent_Price'])
# y = df['Total_Rent_Price']

# # Preprocessing Pipeline
# num_features = X.select_dtypes(include=['int64', 'float64']).columns
# cat_features = X.select_dtypes(include=['object']).columns

# num_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# cat_transformer = Pipeline([
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
# ])

# preprocessor = ColumnTransformer([
#     ('num', num_transformer, num_features),
#     ('cat', cat_transformer, cat_features)
# ])

# # Train-Test Split
# X_transformed = preprocessor.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# # Hyperparameter Optimization
# def objective(trial):
#     model = XGBRegressor(
#         n_estimators=trial.suggest_int('n_estimators', 200, 1000, step=100),
#         learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2),
#         max_depth=trial.suggest_int('max_depth', 3, 15),
#         subsample=trial.suggest_float('subsample', 0.6, 1.0),
#         colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         gamma=trial.suggest_float('gamma', 0, 10),
#         reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
#         reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
#         random_state=42
#     )
#     # early_stopping = callback.EarlyStopping(rounds=10, save_best=True, metric_name="rmse")

#     # # model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[early_stopping])

#     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
#     y_pred = model.predict(X_test)
#     return mean_squared_error(y_test, y_pred)

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)
# best_params = study.best_params

# # Train Best Model
# best_xgb = XGBRegressor(**best_params)
# best_xgb.fit(X_train, y_train)
# y_pred_xgb = best_xgb.predict(X_test)
# y_xgb = best_xgb.predict(X_train)
# # Stacking Model
# stacked_model = StackingRegressor(
#     estimators=[
#         ('xgb', best_xgb),
#         ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
#         ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
#     ],
#     final_estimator=XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
# )
# stacked_model.fit(X_train, y_train)
# y_pred_stack = stacked_model.predict(X_test)
# y_stack = stacked_model.predict(X_train)
# # Neural Network Model
# def build_nn():
#     model = Sequential([
#         Dense(256, kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(0.4),

#         Dense(128, kernel_regularizer=l2(0.001)),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(0.4),

#         Dense(64, kernel_regularizer=l2(0.001)),
#         LeakyReLU(alpha=0.1),
#         BatchNormalization(),
#         Dropout(0.3),

#         Dense(1)
#     ])
   
#     model.compile(optimizer=Adam(learning_rate=0.003), loss=Huber(delta=1.0), metrics=['mae'])
#     return model
                                                                      
# nn_model = build_nn()
# nn_callbacks = [
#     ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
#     EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# ]

# history = nn_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
#                         epochs=200, batch_size=32, verbose=1, callbacks=nn_callbacks)
                                         
# y_pred_nn = nn_model.predict(X_test).flatten()
# y_nn = nn_model.predict(X_train).flatten()
# # Model Evaluation
# def evaluate_model(name, y_true, y_pred):
#     r2 = r2_score(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     print(f"\n{name} Performance:")
#     print(f"R² Score: {r2:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"RMSE: {rmse:.4f}")


# evaluate_model("Optimized XGBoost", y_test, y_pred_xgb)
# evaluate_model("Stacking Model", y_test, y_pred_stack)
# evaluate_model("Neural Network", y_test, y_pred_nn) 
# evaluate_model("Training xgb",y_train,y_xgb)
# evaluate_model("Training stacking model",y_train,y_stack)
# evaluate_model("Training nn",y_train,y_nn)

import optuna
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
DB_url = os.getenv('DB_url')
engine = create_engine(DB_url)

# Load dataset 
df = pd.read_sql("SELECT * FROM car_rent LIMIT 20000", engine)
df['month'] = pd.to_datetime(df['Rent_Date']).dt.month
df['day_of_week'] = pd.to_datetime(df['Rent_Date']).dt.dayofweek
df.drop(['Rent_Date'], axis=1, inplace=True)
df.drop(['User_ID', 'TravelCode', 'Pickup_Location', 'Dropoff_Location', 'Rent_ID'], axis=1, inplace=True)

# Define features and target
X = df.drop(columns=['Total_Rent_Price'])
y = df['Total_Rent_Price']

# Numerical and categorical features
num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

# Pipelines
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

X_transformed = preprocessor.fit_transform(X)

cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_features)
all_feature_names = np.concatenate([num_features, cat_feature_names])

columns_to_drop = ['Rental_Agency_Budget', 'Fuel_Policy_Partial', 'Rental_Agency_Enterprise', 'Car_BookingStatus_Pending']
columns_to_keep = [col for col in all_feature_names if col not in columns_to_drop]
indices_to_keep = [np.where(all_feature_names == col)[0][0] for col in columns_to_keep]

X_filtered = X_transformed[:, indices_to_keep]

# Hyperparameter tuning
def objective(trial, model_name, X_train, y_train):
    params = {}
    if model_name == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'max_features': trial.suggest_float('max_features', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestRegressor(**params)

    elif model_name == 'Gradient Boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
            'random_state': 42
        }
        model = GradientBoostingRegressor(**params)

    elif model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.01, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)

    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    return -scores.mean()

best_params = {}
models = {}
for model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name, X_filtered, y), n_trials=20)
    best_params[model_name] = study.best_params
    models[model_name] = (RandomForestRegressor if model_name == 'Random Forest' else
                           GradientBoostingRegressor if model_name == 'Gradient Boosting' else
                           xgb.XGBRegressor)(**best_params[model_name])
    models[model_name].fit(X_filtered, y)

def meta_objective(trial):
    alpha = trial.suggest_loguniform('alpha', 0.01, 10.0)
    meta_model = Ridge(alpha=alpha)
    stacking_model = StackingRegressor(
        estimators=[('rf', models['Random Forest']), ('gb', models['Gradient Boosting']), ('xgb', models['XGBoost'])],
        final_estimator=meta_model
    )
    scores = cross_val_score(stacking_model, X_filtered, y, scoring='neg_mean_squared_error', cv=5)
    return -scores.mean()

meta_study = optuna.create_study(direction='minimize')
meta_study.optimize(meta_objective, n_trials=10)

best_alpha = meta_study.best_params['alpha']
stacking_model = StackingRegressor(
    estimators=[('rf', models['Random Forest']), ('gb', models['Gradient Boosting']), ('xgb', models['XGBoost'])],
    final_estimator=Ridge(alpha=best_alpha)
)
stacking_model.fit(X_filtered, y)
print(f"Best alpha for Ridge meta-learner: {best_alpha}")

y_pred = stacking_model.predict(X_filtered)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
           

joblib.dump(stacking_model, "best_model_Car.pkl")
joblib.dump(preprocessor, "preprocessor_Car.pkl")
print("Model saved as best_model_Car.pkl")