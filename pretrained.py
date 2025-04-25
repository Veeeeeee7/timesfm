import pandas as pd
import numpy as np
from collections import defaultdict
import timesfm
import time
import datetime
import os
import torch
# Data pipelining
def get_batched_data_fn(
    batch_size: int = 128, 
    context_len: int = 168, 
    horizon_len: int = 24,
    df: pd.DataFrame = None,
):
    entries = defaultdict(list)

    num_entries = 0
    for station_id in df["station_id"].unique():
        sub_df = df[df["station_id"] == station_id]
        for start in range(0, len(sub_df) - (context_len + horizon_len), horizon_len):
            num_entries += 1
            # constant variables
            entries["station_id"].append(station_id)
            entries["latitude"].append(sub_df["latitude"].iloc[0])
            entries["longitude"].append(sub_df["longitude"].iloc[0])
            
            # inputs/outputs
            entries["PM25_Concentration_inputs"].append(sub_df["PM25_Concentration"][start:(context_end := start + context_len)].tolist())
            entries["PM25_Concentration_outputs"].append(sub_df["PM25_Concentration"][context_end:(context_end + horizon_len)].tolist())
            
            # dynamic numerical variables
            entries["NO2_Concentration"].append(sub_df["NO2_Concentration"][start:context_end + horizon_len].tolist())
            entries["CO_Concentration"].append(sub_df["CO_Concentration"][start:context_end + horizon_len].tolist())
            entries["O3_Concentration"].append(sub_df["O3_Concentration"][start:context_end + horizon_len].tolist())
            entries["SO2_Concentration"].append(sub_df["SO2_Concentration"][start:context_end + horizon_len].tolist())
            entries["temperature"].append(sub_df["temperature"][start:context_end + horizon_len].tolist())
            entries["pressure"].append(sub_df["pressure"][start:context_end + horizon_len].tolist())
            entries["humidity"].append(sub_df["humidity"][start:context_end + horizon_len].tolist())
            entries["wind_speed"].append(sub_df["wind_speed"][start:context_end + horizon_len].tolist())
            entries["wind_direction"].append(sub_df["wind_direction"][start:context_end + horizon_len].tolist())

            # dynamic categorical variables
            entries["weather"].append(sub_df["weather"][start:context_end + horizon_len].tolist())
    
    log(f"Number of batches: {1 + (num_entries - 1) // batch_size}")

    def data_fn():
        for i in range(1 + (num_entries - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in entries.items()}
    
    return data_fn

# Define metrics
def mse(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

def mae(y_pred, y_true):
  y_pred = np.array(y_pred)
  y_true = np.array(y_true)
  return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

def log(message):
    with open(f'{os.path.join(output_folder, log_file_name)}', 'a') as f:
        f.write(f'{message}\n')
    print(message)

output_folder = '/users/vmli3/timesfm/logs/'
start_time = datetime.datetime.now().strftime("%Y-%-m-%d_%H:%M:%S")
log_file_name = f'log_{start_time}'

if torch.cuda.is_available():
    log(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        log(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    log("No CUDA devices available")

# loading data
start_time = time.time()
log("Loading data...")
df = pd.read_csv("/users/vmli3/timesfm/data/train_data.csv")
df["time"] = pd.to_datetime(df["time"])
df = df.fillna(0)
input_data = get_batched_data_fn(df=df)
metrics = defaultdict(list)
horizon_len = 24
num_layers = 50
context_len = 168
log(f"Data loaded in {int(time.time() - start_time)} seconds")

# loading model
start_time = time.time()
log("Loading model...")
model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=128,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)
log(f"Model loaded in {int(time.time() - start_time)} seconds")

# running model
log("Running model...")
for i, batch in enumerate(input_data()):
    raw_forecast, _ = model.forecast(inputs=batch["PM25_Concentration_inputs"], freq=[0] * len(batch["PM25_Concentration_inputs"]))
    start_time = time.time()
    cov_forecast, ols_forecast = model.forecast_with_covariates(  
        inputs=batch["PM25_Concentration_inputs"],
        dynamic_numerical_covariates={
            "NO2_Concentration": batch["NO2_Concentration"],
            "CO_Concentration": batch["CO_Concentration"],
            "O3_Concentration": batch["O3_Concentration"],
            "SO2_Concentration": batch["SO2_Concentration"],
            "temperature": batch["temperature"],
            "pressure": batch["pressure"],
            "humidity": batch["humidity"],
            "wind_speed": batch["wind_speed"],
            "wind_direction": batch["wind_direction"],
        },
        dynamic_categorical_covariates={
            "weather": batch["weather"],
        },
        static_numerical_covariates={
            "latitude": batch["latitude"],
            "longitude": batch["longitude"],
        },
        static_categorical_covariates={},
        freq=[0] * len(batch["PM25_Concentration_inputs"]),
        xreg_mode="xreg + timesfm",
        ridge=0.0,
        force_on_cpu=False,
        normalize_xreg_target_per_input=True,
    )

    log(f"Finished batch {i} in {round(time.time() - start_time, 2)} seconds")

    metrics["eval_mae_timesfm"].extend(mae(raw_forecast[:, :horizon_len], batch["PM25_Concentration_outputs"]))
    metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, batch["PM25_Concentration_outputs"]))
    metrics["eval_mae_xreg"].extend(mae(ols_forecast, batch["PM25_Concentration_outputs"]))
    metrics["eval_mse_timesfm"].extend(mse(raw_forecast[:, :horizon_len], batch["PM25_Concentration_outputs"]))
    metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, batch["PM25_Concentration_outputs"]))
    metrics["eval_mse_xreg"].extend(mse(ols_forecast, batch["PM25_Concentration_outputs"]))

for metric, value in metrics.items():
    log(f"{metric}: {np.mean(value)}")