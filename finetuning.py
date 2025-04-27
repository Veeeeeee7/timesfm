import pandas as pd
import numpy as np
from collections import defaultdict
import timesfm
import time
import datetime
import os
import torch
from autogluon.tabular import TabularPredictor
# Data pipelining
def get_batched_data_fn(
    batch_size: int = 128, 
    context_len: int = 160, 
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
            entries["PM10_Concentration"].append(sub_df["PM10_Concentration"][start:context_end + horizon_len].tolist())
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
    
    log(f"Number of batches: {(num_entries - 1) // batch_size}")

    def data_fn():
        # for i in range(1 + (num_entries - 1) // batch_size):
        # throw away the last batch becasuse it is not full
        for i in range((num_entries - 1) // batch_size):
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

# loading training data
start_time = time.time()
log("Loading training data...")
# df_train = pd.read_csv("/users/vmli3/timesfm/data/train_data.csv").head(10000)
df_train = pd.read_csv("/users/vmli3/timesfm/data/train_data.csv")
df_train["time"] = pd.to_datetime(df_train["time"])

# Normalize the PM25_Concentration column
df_train["PM25_Concentration"] = (df_train["PM25_Concentration"] - df_train["PM25_Concentration"].mean()) / df_train["PM25_Concentration"].std()

df_train = df_train.fillna(0)
batch_size = 128
per_core_batch_size = 32
horizon_len = 24
num_layers = 50
context_len = 160
input_patch_length = 32
output_patch_length = 128
input_data = get_batched_data_fn(batch_size=batch_size, horizon_len=horizon_len, context_len=context_len, df=df_train)
log(f"Data loaded in {int(time.time() - start_time)} seconds")

# loading model
start_time = time.time()
log("Loading model...")
model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=per_core_batch_size,
          horizon_len=horizon_len,
          num_layers=num_layers,
          use_positional_embedding=False,
          context_len=context_len,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)
log(f"Model loaded in {int(time.time() - start_time)} seconds")

# Creating Train Embeddings Using TimesFM
log("Creating Train Embeddings using TimesFM...")
X_train = []
y_train = []
for i, batch in enumerate(input_data()):
    start_time = time.time()
    patch_embeddings, xregs, xregs_on_context = model.embed_with_covariates(  
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

    num_core_batches = len(patch_embeddings)
    num_patches = len(patch_embeddings[0])
    # log(f"Number of steps: {num_core_batches}")
    # log(f"Number of patches: {num_patches}")
    
    # combine per_core_batch embeddings into batches
    batches = [torch.cat(col_tensors, dim=0)
          for col_tensors in zip(*patch_embeddings)]
    
    # make xregs into tensor (batch_size, horizon_len) and xregs_on_context (batch_size, context_len)
    xregs = torch.stack([
        torch.as_tensor(a, dtype=batches[0].dtype, device=batches[0].device)
        for a in xregs
    ], dim=0)

    xregs_on_context = torch.stack([
        torch.as_tensor(a, dtype=batches[0].dtype, device=batches[0].device)
        for a in xregs_on_context
    ], dim=0)

    # combine output patches and xregs into horizon length
    num_batches, num_input_patches, embedding_dim = batches[0].shape
    num_output_patches = len(batches)
    embeddings = []
    outputs = []
    for output_patch_idx in range(num_output_patches):
        # takes current output patch
        embedding_idx = batches[output_patch_idx]

        # use the last patch of the input and combine with xregs_on_context
        # ** DONT ADD XREGS AS THEY ARE BASED ON TRUE VALUES WE DONT KNOW **
        combined = torch.cat([embedding_idx[:, -1, :], xregs_on_context], dim=1)

        embeddings.append(combined.cpu().numpy())
        outputs.append(batch["PM25_Concentration_outputs"][output_patch_idx * output_patch_length:(output_patch_idx + 1) * output_patch_length])

    X_train.append(np.concatenate(embeddings, axis=0))
    y_train.append(np.concatenate(outputs, axis=0))

    log(f"Finished batch {i} in {round(time.time() - start_time, 2)} seconds")

X_train = np.concatenate(X_train, axis=0)
y_train = np.concatenate(y_train, axis=0)
log(f"X shape: {X_train.shape}")
log(f"y shape: {y_train.shape}")

# training autogluon
train_data = pd.DataFrame(X_train)
for i in range(len(y_train[0])):
    log(f"Training Autogluon for time {i}...")
    start_time = time.time()
    train_data['target'] = y_train[:, i]

    model_folder = '/scratch/vmli3/timesfm/models'
    model_path = os.path.join(model_folder, f"{i}")
    predictor = TabularPredictor(problem_type='regression', label='target', eval_metric='mean_squared_error', path=model_path)
    predictor.fit(train_data, memory_limit=256, num_cpus=96, num_gpus=4, time_limit=60*60*24*2 / 24, fit_strategy='sequential', presets='good_quality')

    log(f"Autogluon training for time {i} finished in {int(time.time() - start_time)} seconds")

start_time = time.time()
log("Loading testing data...")
# df_test = pd.read_csv("/users/vmli3/timesfm/data/test_data.csv").head(10000)
df_test = pd.read_csv("/users/vmli3/timesfm/data/test_data.csv")
df_test["time"] = pd.to_datetime(df_test["time"])

# Normalize the PM25_Concentration column
df_test["PM25_Concentration"] = (df_test["PM25_Concentration"] - df_test["PM25_Concentration"].mean()) / df_test["PM25_Concentration"].std()

df_test = df_test.fillna(0)
batch_size = 128
per_core_batch_size = 32
horizon_len = 24
num_layers = 50
context_len = 160
input_patch_length = 32
output_patch_length = 128
input_data = get_batched_data_fn(batch_size=batch_size, horizon_len=horizon_len, context_len=context_len, df=df_test)
log(f"Data loaded in {int(time.time() - start_time)} seconds")

# Creating Test Embeddings Using TimesFM
log("Creating Test Embeddings using TimesFM...")
X_test = []
y_test = []
for i, batch in enumerate(input_data()):
    start_time = time.time()
    patch_embeddings, xregs, xregs_on_context = model.embed_with_covariates(  
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

    num_core_batches = len(patch_embeddings)
    num_patches = len(patch_embeddings[0])
    # log(f"Number of steps: {num_core_batches}")
    # log(f"Number of patches: {num_patches}")
    
    # combine per_core_batch embeddings into batches
    batches = [torch.cat(col_tensors, dim=0)
          for col_tensors in zip(*patch_embeddings)]
    
    # make xregs into tensor (batch_size, horizon_len) and xregs_on_context (batch_size, context_len)
    xregs = torch.stack([
        torch.as_tensor(a, dtype=batches[0].dtype, device=batches[0].device)
        for a in xregs
    ], dim=0)

    xregs_on_context = torch.stack([
        torch.as_tensor(a, dtype=batches[0].dtype, device=batches[0].device)
        for a in xregs_on_context
    ], dim=0)

    # combine output patches and xregs into horizon length
    num_batches, num_input_patches, embedding_dim = batches[0].shape
    num_output_patches = len(batches)
    embeddings = []
    outputs = []
    for output_patch_idx in range(num_output_patches):
        # takes current output patch
        embedding_idx = batches[output_patch_idx]

        # use the last patch of the input and combine with xregs_on_context
        # ** DONT ADD XREGS AS THEY ARE BASED ON TRUE VALUES WE DONT KNOW **
        combined = torch.cat([embedding_idx[:, -1, :], xregs_on_context], dim=1)

        embeddings.append(combined.cpu().numpy())
        outputs.append(batch["PM25_Concentration_outputs"][output_patch_idx * output_patch_length:(output_patch_idx + 1) * output_patch_length])

    X_test.append(np.concatenate(embeddings, axis=0))
    y_test.append(np.concatenate(outputs, axis=0))

    log(f"Finished batch {i} in {round(time.time() - start_time, 2)} seconds")

X_test = np.concatenate(X_test, axis=0)
y_test = np.concatenate(y_test, axis=0)
log(f"X shape: {X_test.shape}")
log(f"y shape: {y_test.shape}")

# Predicting using Autogluon for each target column
test_data = pd.DataFrame(X_test)
metrics_list = []
for i in range(len(y_test[0])):
    log(f"Predicting for time {i}...")
    start_time = time.time()
    test_data['target'] = y_test[:, i]
    
    model_folder = '/scratch/vmli3/timesfm/models'
    model_path = os.path.join(model_folder, f"{i}")
    predictor = TabularPredictor.load(model_path)
    
    metrics = predictor.evaluate(test_data)
    metrics_list.append(metrics)
    log(f"Metrics for time {i}: {metrics}")
    log(f"Autogluon training for time {i} finished in {int(time.time() - start_time)} seconds")
