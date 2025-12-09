
import sys
import os
from pathlib import Path
import logging
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Any


# folder structure:
# root/
#   OpticalAgent/
#       api/
#           inference_api/
#               inference_service.py (current file)
#   OpticalGraphormer/
#       code/
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent   # point to root/
code_dir = project_root / "OpticalGraphormer" / "code"


if not code_dir.is_dir():
    raise FileNotFoundError(f"'{code_dir}' directory not found. Please check your project structure.")
if str(code_dir) not in sys.path:
    sys.path.insert(0, str(code_dir))  # If the path is not in sys.path, insert it at position 0 (the front) of the list

from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

# Note: These import paths are relative to `graph_AI_optics/code/`
from graphormer.data.collator import collator
from graphormer.data.optical_dataset import _build_graph_from_config
from graphormer.tasks.graph_prediction import GraphPredictionConfig, GraphPredictionTask
from graphormer.data.wrapper import preprocess_item
import graphormer.models.graphormer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- borrow from optical_dataset.py) ---
HEIGHT_RANGE = (50, 500)
HEIGHT_BINS = 46
PITCH_RANGE = (200, 800)
PITCH_BINS = 61
ROTATE_RANGE = (0, 90)
ROTATE_BINS = 19
SPAN_RANGE = (100, 300)
SPAN_BINS = 200
POLAR_RANGE = (0, 170)
POLAR_BINS = 18
LAMBDA_RANGE = (300, 1590)
LAMBDA_BINS = 150
POINT_RANGE = (1, 399)
POINT_BINS = 200
GAP_RANGE = (50, 400)
GAP_BINS = 36
THICK_RANGE = (10, 100)
THICK_BINS = 10

VOLTAGE_MAPPING = {
    -0.2: 0,
    0.1: 1,
    0.15: 2,
    0.2: 3,
    0.5: 4
}


class InferenceService:
    def __init__(self, checkpoint_path: str, user_data_dir: str, device: str = "cuda"):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.model = None
        self.task = None
        self.cfg = None

        # Retrieve default values from GraphPredictionConfig, which are determined during model training
        # If different parameters were used during your model training, ensure to update them here
        self.max_nodes = GraphPredictionConfig.max_nodes  # 128
        self.multi_hop_max_dist = GraphPredictionConfig.multi_hop_max_dist  # 5
        self.spatial_pos_max = GraphPredictionConfig.spatial_pos_max  # 1024
        self.num_classes = GraphPredictionConfig.num_classes  # 400 (length of spectrum)

        self._load_model(checkpoint_path, user_data_dir)

    def _load_model(self, checkpoint_path: str, user_data_dir: str):
        logger.info(f"Loading model from: {checkpoint_path}")
        logger.info(f"User data directory: {user_data_dir}")

        # Manually construct an argparse.Namespace object based on the parameters in evaluate.sh
        # This is required for Fairseq's setup_task and build_model to function properly.
        # We first get an empty parser, then add the necessary parameters.
        # Additionally, add the extra parameters defined in evaluate.py
        parser = options.get_training_parser()
        parser.add_argument("--split", type=str, default="test")
        parser.add_argument("--metric", type=str, default="mse")
        parser.add_argument("--visual", action="store_true")

        args = parser.parse_args([])

        # common args
        args.update_epoch_batch_itr = False
        args.arch = "graphormer_base"
        args.encoder_embed_dim = 64
        args.encoder_layers = 4
        args.encoder_attention_heads = 4
        args.encoder_ffn_embed_dim = 256
        args.num_workers = 0
        args.fp16 = False
        args.data_buffer_size = 20
        args.log_format = "simple"
        args.log_interval = 100
        args.no_progress_bar = True
        args.seed = 1

        args.num_atoms = GraphPredictionConfig.num_atoms
        args.num_edges = GraphPredictionConfig.num_edges
        args.num_in_degree = GraphPredictionConfig.num_in_degree
        args.num_out_degree = GraphPredictionConfig.num_out_degree
        args.num_spatial = GraphPredictionConfig.num_spatial
        args.num_edge_dis = GraphPredictionConfig.num_edge_dis
        args.multi_hop_max_dist = GraphPredictionConfig.multi_hop_max_dist
        args.spatial_pos_max = GraphPredictionConfig.spatial_pos_max
        args.edge_type = GraphPredictionConfig.edge_type
        args.num_classes = self.num_classes

        # task-specific args
        args.dataset_name = "optical_system_dataset"
        args.dataset_source = "pyg"
        args.task = "graph_prediction"
        args.user_data_dir = user_data_dir  # graph_AI_optics/code/graphormer/data
        args.pretrained_model_name = "none"
        args.load_pretrained_model_output_layer = False
        args.train_epoch_shuffle = False
        args.save_dir = os.path.dirname(checkpoint_path)

        self.cfg = convert_namespace_to_omegaconf(args)

        # Ensure key parameters in the task configuration match those used during training
        # GraphPredictionConfig default values: num_classes=400, max_nodes=128, multi_hop_max_dist=5, spatial_pos_max=1024
        # If these values differed during training, you need to manually update `self.cfg.task` here
        self.cfg.task.num_classes = self.num_classes
        self.cfg.task.max_nodes = self.max_nodes
        self.cfg.task.multi_hop_max_dist = self.multi_hop_max_dist
        self.cfg.task.spatial_pos_max = self.spatial_pos_max

        # Initialize the task (this will trigger a call to create_optical_system_dataset in optical_dataset.py)
        # Note: create_optical_system_dataset attempts to read config_list.json and topT.txt/downT.txt
        # These files must exist in user_data_dir even during inference.
        self.task = tasks.setup_task(self.cfg.task)

        # build model
        self.model = self.task.build_model(self.cfg.model)

        # load checkpoint
        model_state = torch.load(checkpoint_path, map_location=self.device)["model"]
        self.model.load_state_dict(model_state, strict=True, model_cfg=self.cfg.model)
        del model_state

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully and set to evaluation mode.")


    def _dataframe_to_configs(self, raw_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert optical parameters in the form of a flattened pandas DataFrame into a structured list of configuration dictionaries,
        which is suitable for the `_build_graph_from_config` function.
        """
        configs = []
        for _, row in raw_data.iterrows():
            config = {
                "active": {
                    "thickness": row["active_thick"],
                    "voltage": row["active_voltage"]
                },
                "light": {
                    "polar": row["light_polar"],
                    "start_lambda": row["light_start_lambda"],
                    "points": row["light_points"],
                    "stop_lambda": row["light_stop_lambda"]
                },
                "boundary": {
                    "periodic": row["periodic"],
                    "gap": row["gap"]
                },
                "blocks": []
            }

            # Process up to 4 blocks
            # This means that my input DataFrame must be in the following format
            # [active_thick, active_voltage, light_polar, light_start_lambda, light_points, light_stop_lambda, periodic, gap,
            # material1, height1, pitch1, x_expand1, y_expand1, x_loc1, y_loc1, rotate1,
            # material2, height2, pitch2, x_expand2, y_expand2, x_loc2, y_loc2, rotate2,
            # material3, height3, pitch3, x_expand3, y_expand3, x_loc3, y_loc3, rotate3,
            # material4, height4, pitch4, x_expand4, y_expand4, x_loc4, y_loc4, rotate4,]
            for i in range(1, 5):
                material_key = f"material{i}"
                # Check if the material field exists and is not None/NaN to determine if the block exists
                if pd.notna(row.get(material_key)) and row.get(material_key) is not None:
                    block = {
                        "material": row[material_key],
                        "height": row[f"height{i}"],
                        "pitch": row[f"pitch{i}"],
                        "x_span": row[f"x_expand{i}"],
                        "y_span": row[f"y_expand{i}"],
                        "x_loc": row[f"x_loc{i}"],
                        "y_loc": row[f"y_loc{i}"],
                        "rotate": row[f"rotate{i}"]
                    }
                    config["blocks"].append(block)
            configs.append(config)
        return configs

    def predict_spectrum(self, raw_data: pd.DataFrame) -> List[List[float]]:

        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model first.")

        # 1. Convert input DataFrame to a structured list of configuration dictionaries
        configs = self._dataframe_to_configs(raw_data)
        logger.info(f"Converted {len(configs)} input configurations to structured configs.")

        # 2. Convert each configuration dictionary to a PyG Data object
        pyg_data_list = []
        # Real spectrum labels and masks are not required during inference, so dummy tensors are passed in
        dummy_spectrum = torch.zeros(self.num_classes, dtype=torch.float32)
        dummy_mask = torch.zeros(self.num_classes, dtype=torch.bool)

        for i, config in enumerate(configs):
            try:
                # _build_graph_from_config is from optical_dataset.py
                pyg_data = _build_graph_from_config(config, dummy_spectrum, dummy_mask)
                # Add a unique index to each PyG Data object
                pyg_data.idx = torch.tensor(i, dtype=torch.long)
                pyg_data = preprocess_item(pyg_data)  # Introduce attributes such as attn_bias
                pyg_data_list.append(pyg_data)
            except Exception as e:
                logger.error(f"Error building graph for config: {config}. Error: {e}")
                # If graph construction fails for a sample, you can choose to skip the sample or return an error
                continue

        if not pyg_data_list:
            logger.warning("No valid graphs could be built from the input data. Returning empty predictions.")
            return []

        # 3. Collate the list of PyG Data objects into a batch of tensors
        # The collator function is from graphormer.data.collator
        collated_batch = collator(
            pyg_data_list,
            max_node=self.max_nodes,
            multi_hop_max_dist=self.multi_hop_max_dist,
            spatial_pos_max=self.spatial_pos_max,
        )
        logger.info(f"Collated batch with {len(pyg_data_list)} samples.")

        # 4. Prepare batch data and perform model inference
        # According to the forward method signature of the Fairseq Graphormer model,
        # it expects a dictionary containing the 'batched_data' key as the parameter.
        # 'batched_data' itself is a dictionary containing all tensors returned by the collator.
        batched_data_on_device = {
            k: v.to(self.device) for k, v in collated_batch.items() if k not in ["idx", "y", "mask"]
        }
        net_input = {"batched_data": batched_data_on_device}

        with torch.no_grad():
            logits = self.model(**net_input)
        logger.info(f"Model inference completed. Output shape: {logits.shape}")

        # 5. Convert model outputs to the required format (List[List[float]])
        predictions = logits.detach().cpu().numpy().tolist()
        return predictions
