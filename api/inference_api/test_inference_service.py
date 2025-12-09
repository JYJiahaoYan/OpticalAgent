import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
import logging
import json  # Used to check files such as config_list.json

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# current_dir: graph_AI_optics/OpticalAgent/api/inference_api/
# project_root: graph_AI_optics/
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from api.inference_api.inference_service import InferenceService
from api.inference_api.schemas import FlatOpticalParamsInput

# Define paths (consistent with those in main.py)
CHECKPOINT_PATH = project_root / "code" / "examples" / "ckpts" / "checkpoint_best.pt"
USER_DATA_DIR = project_root / "code" / "graphormer" / "data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- Debugging Paths ---")
print(f"Project Root: {project_root}")
print(f"Checkpoint Path: {CHECKPOINT_PATH}")
print(f"User Data Directory: {USER_DATA_DIR}")
print(f"Inference Device: {DEVICE}")
print("-" * 30)


# Check existence of critical files
def check_file_existence(file_path, description):
    if not file_path.exists():
        logger.error(f"ERROR: {description} not found at {file_path}")
        sys.exit(1)
    else:
        logger.info(f"SUCCESS: {description} found at {file_path}")


check_file_existence(CHECKPOINT_PATH, "Model Checkpoint")
check_file_existence(USER_DATA_DIR, "User Data Directory")
check_file_existence(USER_DATA_DIR / "blocks_list.txt", "blocks_list.txt")
check_file_existence(USER_DATA_DIR / "others_list.txt", "others_list.txt")
check_file_existence(USER_DATA_DIR / "downT.txt", "downT.txt")

# Check config_list.json, which may be read during setup_task
config_list_path = USER_DATA_DIR / "config_list.json"
if not config_list_path.exists():
    logger.warning(f"WARNING: {config_list_path} not found. This might cause issues during task setup if it's expected.")
    # If this file is required, you may need to create an empty or minimal valid JSON file
    # Example:
    # with open(config_list_path, 'w') as f:
    #     json.dump([], f)
    # logger.info(f"Created an empty {config_list_path} for testing purposes.")
else:
    logger.info(f"SUCCESS: {config_list_path} found.")
    try:
        with open(config_list_path, 'r') as f:
            test_config_data = json.load(f)
            logger.info(f"Successfully loaded {config_list_path}. Contains {len(test_config_data)} entries.")
    except json.JSONDecodeError as e:
        logger.error(f"ERROR: Failed to parse {config_list_path}: {e}")
        sys.exit(1)


# --- 1. Test InferenceService Initialization ---
print("\n--- Testing InferenceService Initialization ---")
service: InferenceService = None
try:
    service = InferenceService(
        checkpoint_path=str(CHECKPOINT_PATH),
        user_data_dir=str(USER_DATA_DIR),
        device=DEVICE
    )
    logger.info("InferenceService initialized successfully.")
    logger.info(f"Model num_classes: {service.num_classes}")
    logger.info(f"Model max_nodes: {service.max_nodes}")
except Exception as e:
    logger.error(f"ERROR: InferenceService initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) # Exit the test if model loading fails

# --- 2. Prepare Sample Input Data ---
print("\n--- Preparing Sample Input Data ---")
# Create an instance of FlatOpticalParamsInput.
# Be sure to use a real, representative input that falls within the range of the training data.
# Include at least one Block, and test the scenario where Optional fields (e.g., Block2, Block3, Block4) are set to None.
sample_input_data = FlatOpticalParamsInput(
    active_thick=50.0,
    active_voltage=0.1, # Ensure this value is present in VOLTAGE_MAPPING
    light_polar=45.0,
    light_start_lambda=400.0,
    light_points=100,
    light_stop_lambda=800.0,
    periodic=1,
    gap=100.0,
    material1="Si", # Ensure this material is present in blocks_list.txt or others_list.txt
    height1=100.0,
    pitch1=200.0,
    x_expand1=50.0,
    y_expand1=50.0,
    x_loc1=0.0,
    y_loc1=0.0,
    rotate1=0.0,
    # Optional: Test the second Block
    material2="Ag",
    height2=80.0,
    pitch2=250.0,
    x_expand2=40.0,
    y_expand2=40.0,
    x_loc2=50.0,
    y_loc2=50.0,
    rotate2=90.0,
    # Optional: Test the third Block set to None
    material3=None,
    height3=None,
    pitch3=None,
    x_expand3=None,
    y_expand3=None,
    x_loc3=None,
    y_loc3=None,
    rotate3=None,
    # Optional: Test the fourth Block set to None
    material4=None,
    height4=None,
    pitch4=None,
    x_expand4=None,
    y_expand4=None,
    x_loc4=None,
    y_loc4=None,
    rotate4=None,
)

# Convert the list of Pydantic models to a pandas DataFrame (simulating the logic in main.py)
data_dicts = [sample_input_data.dict(exclude_unset=True)]

all_block_cols = []
for i in range(1, 5):
    all_block_cols.extend([
        f"material{i}", f"height{i}", f"pitch{i}", f"x_expand{i}",
        f"y_expand{i}", f"x_loc{i}", f"y_loc{i}", f"rotate{i}"
    ])

base_cols = [
    "active_thick", "active_voltage", "light_polar", "light_start_lambda",
    "light_points", "light_stop_lambda", "periodic", "gap"
]

df_columns = base_cols + all_block_cols

# Create a DataFrame from the list of dictionaries. from_records can handle cases where keys are missing from the dictionaries
df = pd.DataFrame.from_records(data_dicts, columns=df_columns)

# Convert NaN values in the DataFrame (caused by missing data) to None
for i in range(1, 5):
    material_col = f"material{i}"
    if material_col in df.columns:
        df[material_col] = df[material_col].replace({np.nan: None})

logger.info("Sample DataFrame created:")
logger.info(df)

# --- 3. Test Data Transformation (_dataframe_to_configs) ---
print("\n--- Testing Data Transformation (_dataframe_to_configs) ---")
try:
    configs = service._dataframe_to_configs(df)
    logger.info(f"Successfully converted DataFrame to {len(configs)} structured configs.")
    if configs:
        logger.info("First config dictionary (truncated):")
        # Print a portion to avoid excessive output length
        logger.info(json.dumps(configs[0], indent=2)[:500] + "...")
except Exception as e:
    logger.error(f"ERROR: Data transformation failed in _dataframe_to_configs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- 4. Test Prediction Logic (predict_spectrum) ---
print("\n--- Testing Prediction Logic (predict_spectrum) ---")
try:
    predictions = service.predict_spectrum(df)
    logger.info(f"Prediction successful. Received {len(predictions)} predictions.")
    if predictions:
        logger.info(f"First prediction (truncated): {predictions[0][:10]}...")
        logger.info(f"Length of first spectrum: {len(predictions[0])}")
        # Verify if the length of the predicted spectrum matches the model's expected num_classes
        if len(predictions[0]) != service.num_classes:
            logger.warning(f"WARNING: Predicted spectrum length ({len(predictions[0])}) does not match expected num_classes ({service.num_classes}).")
    else:
        logger.warning("No predictions returned.")
except Exception as e:
    logger.error(f"ERROR: Prediction failed in predict_spectrum: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n--- All local InferenceService tests completed successfully (if no ERROR messages above) ---")