import os
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from typing import List, Optional
import logging
import traceback

from api.inference_api.schemas import FlatOpticalParamsInput, SpectrumOutput
from api.inference_api.inference_service import InferenceService


project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Get the project root directory, pointing to graph_AI_optics/

# Path to the model checkpoint (.pt file)
CHECKPOINT_PATH = os.path.join(project_root, "code", "examples", "ckpts", "checkpoint_best.pt")

# Path to the data directory (contains optical_dataset.py and related data files such as config_list.json)
USER_DATA_DIR = os.path.join(project_root, "code", "graphormer", "data")

# Inference device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. FastAPI Application Initialization ---
app = FastAPI(
    title="Graphormer Optical Spectrum Prediction API",
    description="REST API for predicting optical spectra using a Fairseq/Graphormer model.",
    version="1.0.0",
)

# Initialize InferenceService
# This will load the model once when the application starts.
# If model loading fails, the application will not start.
inference_service: Optional[InferenceService] = None


@app.on_event("startup")
async def startup_event():
    # After declaring 'global inference_service', assignment operations to inference_service within the function
    # will directly modify the variable in the global scope. This ensures that after the application starts,
    # other components can access this initialized service instance.
    global inference_service
    try:
        logging.info("Initializing InferenceService...")
        inference_service = InferenceService(
            checkpoint_path=CHECKPOINT_PATH,
            user_data_dir=USER_DATA_DIR,
            device=DEVICE
        )
        logging.info("InferenceService initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize InferenceService: {e}")
        # Print detailed stack trace to assist with debugging
        traceback.print_exc()
        # Prevent application startup as the model is a core dependency
        raise RuntimeError(f"Failed to load model during startup: {e}")


# API definition

@app.post("/predict", response_model=SpectrumOutput, status_code=status.HTTP_200_OK)
async def predict_spectrum_endpoint(inputs: List[FlatOpticalParamsInput]):

    if not inputs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No input data provided.")

    # Convert the list of Pydantic models to a pandas DataFrame
    data_dicts = [input_obj.dict(exclude_unset=True) for input_obj in inputs]

    # Ensure the DataFrame contains all possible block columns, even if they are not provided in the input (will be filled with None/NaN)
    all_block_cols = []
    for i in range(1, 5):
        all_block_cols.extend([
            f"material{i}", f"height{i}", f"pitch{i}", f"x_expand{i}",
            f"y_expand{i}", f"x_loc{i}", f"y_loc{i}", f"rotate{i}"
        ])

    # Base parameter columns
    base_cols = [
        "active_thick", "active_voltage", "light_polar", "light_start_lambda",
        "light_points", "light_stop_lambda", "periodic", "gap"
    ]

    # Build a complete list of column names to ensure order consistency
    df_columns = base_cols + all_block_cols

    # Create a DataFrame from the list of dictionaries. from_records can handle cases where keys are missing from the dictionaries
    df = pd.DataFrame.from_records(data_dicts, columns=df_columns)

    # Convert NaN values in the DataFrame (caused by missing data) to None,
    # because the `_dataframe_to_configs` function expects `None` to determine whether a block exists.
    # Especially for the 'material' columns, `pd.DataFrame` may convert `None` to `np.nan`.
    for i in range(1, 5):
        material_col = f"material{i}"
        if material_col in df.columns:
            df[material_col] = df[material_col].replace({np.nan: None})

    try:
        if inference_service is None:
            raise RuntimeError("InferenceService is not initialized.")

        predictions = inference_service.predict_spectrum(df)
        return SpectrumOutput(predictions=predictions)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        traceback.print_exc()  # Print detailed stack trace
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {e}")