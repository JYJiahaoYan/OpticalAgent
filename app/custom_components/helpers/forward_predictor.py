from langflow.custom import Component
from langflow.io import StrInput, DataInput, Output
from langflow.schema import Data
from typing import Union, List, Optional, Dict, Any
import json
import requests
import logging


logger = logging.getLogger(__name__)

# see building customized component based on langflow https://docs.langflow.org/components-custom-components


class ForwardPredictor(Component):
    """
    A Langflow custom component to predict optical spectra by calling a FastAPI service.
    It takes optical system configurations and returns predicted spectra.
    """
    display_name = "ForwardPredictor"  # will be the widget name on langflow GUI
    description = "Put the input Data representing structural parameters into self-made trained neural network and output corresponding optical spectrum"
    icon = "Hammer"
    name = "ForwardPredictor"

    inputs = [
        StrInput(
            name="api_url",
            display_name="FastAPI Base URL",
            info="The base URL of the FastAPI service (e.g., http://localhost:8000). The component will append '/predict' to this URL.",
            required=True,
            value="http://localhost:8000"  # Default value for local development
        ),
        DataInput(
            name="optical_params",
            display_name="Optical Parameters",
            info="Input data containing optical system configurations. Can be a single configuration or list of configurations. "
                 "Should include fields like active_thick, active_voltage, light_polar, material1, height1, etc.",
            required=True,
            tool_mode=True,    # Important for Agent to know this is a tool input
        )
    ]

    outputs = [
        Output(name="prediction_result", display_name="Prediction Result", method="get_prediction_result"),
    ]

    # Internal state to store results and errors after the build method runs
    _predictions: Optional[List[List[float]]] = None
    _status_message: str = ""
    _error_details: Optional[str] = None

    def _process_input_data(self, input_value: Union[Data, str, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process the input value, which can be a Data object, a raw string, a dict, or a list,
        and convert it to the format expected by the API.
        Handles both single configurations and lists of configurations.
        """
        raw_data = None
        logger.debug(f"Processing input data. Type: {type(input_value)}")

        if isinstance(input_value, Data):
            logger.debug("Input value is a Data object.")
            if hasattr(input_value, 'data') and input_value.data is not None:
                raw_data = input_value.data
            elif hasattr(input_value, 'text') and input_value.text is not None:
                try:
                    raw_data = json.loads(input_value.text)
                    logger.debug("Successfully parsed Data.text as JSON.")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Data.text as JSON: {input_value.text[:100]}...")
                    raise ValueError("Data object's text attribute must be valid JSON")
            else:
                logger.error("Data object is empty (neither 'data' nor 'text' populated).")
                raise ValueError("Data object is empty (neither 'data' nor 'text' populated)")
        elif isinstance(input_value, str):
            logger.debug("Input value is a string. Attempting to parse as JSON.")
            try:
                raw_data = json.loads(input_value)
                logger.debug("Successfully parsed input string as JSON.")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse input string as JSON: {input_value[:100]}...")
                raise ValueError("Input string must be valid JSON")
        elif isinstance(input_value, (dict, list)):
            logger.debug("Input value is already a dict or list.")
            raw_data = input_value
        else:
            logger.error(f"Unsupported input type: {type(input_value)}. Expected Data, str, dict, or list.")
            raise ValueError(f"Unsupported input type: {type(input_value)}. Expected Data, str, dict, or list.")

        if raw_data is None:
            logger.error("Failed to extract raw data from input, raw_data is None.")
            raise ValueError("Failed to extract raw data from input.")

        # Convert to list format expected by API
        if isinstance(raw_data, dict):
            return [raw_data]
        elif isinstance(raw_data, list):
            # Ensure all items in the list are dictionaries
            if not all(isinstance(item, dict) for item in raw_data):
                raise ValueError("Input list must contain only dictionaries.")
            return raw_data
        else:
            raise ValueError("Processed raw data must be a dictionary or list of dictionaries")

    def _validate_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and ensure required fields are present in the configuration.
        """
        required_fields = [
            "active_thick", "active_voltage", "light_polar",
            "light_start_lambda", "light_points", "light_stop_lambda",
            "periodic", "gap", "material1", "height1", "pitch1",
            "x_expand1", "y_expand1", "x_loc1", "y_loc1", "rotate1"
        ]

        # Check for required fields
        missing_fields = [field for field in required_fields if field not in config or config[field] is None]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        return config

    def build(self):
        """
        Core logic: process inputs, call API, and store results.
        This method populates the internal state (_predictions, _status_message, _error_details).
        """
        # Reset internal state for each execution of the component
        self._predictions = None
        self._status_message = "Component started."
        self._error_details = None
        logger.info("ForwardPredictor component build started.")

        # Construct the full API endpoint URL
        api_url = self.api_url.rstrip('/')  # Remove trailing slash if present
        predict_endpoint = f"{api_url}/predict"
        logger.debug(f"API endpoint: {predict_endpoint}")

        try:
            input_data_list = self._process_input_data(self.optical_params)
            logger.debug(f"Processed input data: {len(input_data_list)} configurations.")

            # Validate each configuration
            validated_configs = []
            for i, config in enumerate(input_data_list):
                try:
                    validated_config = self._validate_configuration(config)
                    validated_configs.append(validated_config)
                except ValueError as e:
                    raise ValueError(f"Configuration {i+1} validation failed: {e}")

            if not validated_configs:
                self._status_message = "Warning: No valid configurations to process. No API call made."
                logger.warning(self._status_message)
                return

            self._status_message = f"Calling API at: {predict_endpoint} with {len(validated_configs)} configurations."
            logger.info(self._status_message)

            # Make API call
            headers = {"Content-Type": "application/json"}
            response = requests.post(
                predict_endpoint,
                json=validated_configs,
                headers=headers,
                timeout=120
            )

            # Check response
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"API response received. Keys: {response_data.keys()}")

            # Extract predictions
            self._predictions = response_data.get("predictions")
            if self._predictions is None:
                raise ValueError("API response missing 'predictions' key")

            self._status_message = f"Prediction successful. Generated {len(self._predictions)} spectra."
            logger.info(self._status_message)

        except ValueError as e:
            self._status_message = "Input validation error."
            self._error_details = str(e)
            logger.error(f"{self._status_message}: {self._error_details}")
        except requests.exceptions.Timeout:
            self._status_message = "API request timed out."
            self._error_details = "Request timeout after 120 seconds."
            logger.error(f"{self._status_message}: {self._error_details}")
        except requests.exceptions.ConnectionError as e:
            self._status_message = "Connection error."
            self._error_details = f"Cannot connect to {predict_endpoint}. Ensure API is running. Details: {e}"
            logger.error(f"{self._status_message}: {self._error_details}")
        except requests.exceptions.HTTPError as e:
            self._status_message = f"HTTP error ({e.response.status_code})."
            try:
                error_response = e.response.json()
                self._error_details = f"API Error: {error_response.get('detail', e.response.text)}"
            except:
                self._error_details = f"API Error: {e.response.text}"
            logger.error(f"{self._status_message}: {self._error_details}")
        except Exception as e:
            self._status_message = "Unexpected error occurred."
            self._error_details = str(e)
            logger.exception(f"{self._status_message}: {self._error_details}") # Use exception for full traceback

    def get_prediction_result(self) -> Data:
        """
        Consolidates prediction results, status, and error details into a single Data object.
        This method is called when the 'prediction_result' output is accessed.
        """
        self.build()
        result_data = {
            "status": self._status_message,
            "predictions": self._predictions if self._predictions is not None else [],
            "error": self._error_details,
        }

        # Construct a summary text for the Data object
        if self._error_details:
            result_text = f"Prediction failed: {self._status_message}. Details: {self._error_details}"
        elif self._predictions:
            result_text = f"Prediction successful. Generated {len(self._predictions)} optical spectra."
        else:
            result_text = self._status_message  # e.g., "Warning: No valid configurations to process."

        logger.debug(f"Returning prediction result: {result_text}")
        return Data(text=result_text, data=result_data)
