import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
keras = tf.keras
import joblib

np.random.seed(42)
tf.random.set_seed(42)

fires = pd.read_csv("./sanbul2district-divby100.csv")
fires['burned_area'] = np.log(fires['burned_area'] + 1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(fires, fires["month"]):
    strat_train_set = fires.loc[train_idx].reset_index(drop=True)
    strat_test_set = fires.loc[test_idx].reset_index(drop=True)

fires = strat_train_set.drop(["burned_area"], axis=1)
fires_labels = strat_train_set["burned_area"].copy()

num_attribs = ["longitude", "latitude", "avg_temp", "max_temp", "max_wind_speed", "avg_wind"]
cat_attribs = ["month", "day"]

full_pipeline = ColumnTransformer([
    ("num", Pipeline([("std_scaler", StandardScaler())]), num_attribs),
    ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_attribs),
])

fires_prepared = full_pipeline.fit_transform(fires)
joblib.dump(full_pipeline, "fires_pipeline.pkl")

fires_test = strat_test_set.drop(["burned_area"], axis=1)
fires_test_prepared = full_pipeline.transform(fires_test)

X_train, X_valid, y_train, y_valid = train_test_split(fires_prepared, fires_labels, test_size=0.2, random_state=42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=[fires_prepared.shape[1]]),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(learning_rate=1e-3))
model.fit(X_train, y_train, epochs=200, validation_data=(X_valid, y_valid))

model.save('fires_model.keras')
print("✅ 모델 학습 완료!")
