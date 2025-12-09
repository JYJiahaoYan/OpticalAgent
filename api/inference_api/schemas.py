# Dependencies required only for API deployment

from typing import List, Optional
from pydantic import BaseModel, Field


class FlatOpticalParamsInput(BaseModel):
    """
    Input model for predicting optical spectra, reflecting the flat DataFrame structure.
    Represents a single configuration of an optical system.
    """
    active_thick: float = Field(..., description="Thickness of the active layer in nm.")
    active_voltage: float = Field(..., description="Voltage applied to the active layer in V (e.g., -0.2, 0.1, 0.15, 0.2, 0.5).")
    light_polar: float = Field(..., description="Polarization angle of the incident light in degrees.")
    light_start_lambda: float = Field(..., description="Starting wavelength of the incident light in nm.")
    light_points: int = Field(..., description="Number of wavelength points for the incident light.")
    light_stop_lambda: float = Field(..., description="Stopping wavelength of the incident light in nm.")
    periodic: int = Field(..., description="Periodicity of the boundary (0 or 1).")
    gap: float = Field(..., description="Gap size in nm.")

    # Block 1 parameters
    material1: Optional[str] = Field(..., description="Material of Block 1 (e.g., 'Si', 'Ag', 'Au').")
    height1: Optional[float] = Field(..., description="Height of Block 1 in nm.")
    pitch1: Optional[float] = Field(..., description="Pitch of Block 1 in nm.")
    x_expand1: Optional[float] = Field(..., description="X-expansion of Block 1 in nm.")
    y_expand1: Optional[float] = Field(..., description="Y-expansion of Block 1 in nm.")
    x_loc1: Optional[float] = Field(..., description="X-location offset of Block 1 (e.g., -half_p, 0, half_p).")
    y_loc1: Optional[float] = Field(..., description="Y-location offset of Block 1 (e.g., -half_p, 0, half_p).")
    rotate1: Optional[float] = Field(..., description="Rotation angle of Block 1 in degrees.")

    # Block 2 parameters (Optional, as a system might have 1-4 blocks)
    material2: Optional[str] = Field(None, description="Material of Block 2.")
    height2: Optional[float] = Field(None, description="Height of Block 2.")
    pitch2: Optional[float] = Field(None, description="Pitch of Block 2.")
    x_expand2: Optional[float] = Field(None, description="X-expansion of Block 2.")
    y_expand2: Optional[float] = Field(None, description="Y-expansion of Block 2.")
    x_loc2: Optional[float] = Field(None, description="X-location offset of Block 2.")
    y_loc2: Optional[float] = Field(None, description="Y-location offset of Block 2.")
    rotate2: Optional[float] = Field(None, description="Rotation angle of Block 2.")

    # Block 3 parameters
    material3: Optional[str] = Field(None, description="Material of Block 3.")
    height3: Optional[float] = Field(None, description="Height of Block 3.")
    pitch3: Optional[float] = Field(None, description="Pitch of Block 3.")
    x_expand3: Optional[float] = Field(None, description="X-expansion of Block 3.")
    y_expand3: Optional[float] = Field(None, description="Y-expansion of Block 3.")
    x_loc3: Optional[float] = Field(None, description="X-location offset of Block 3.")
    y_loc3: Optional[float] = Field(None, description="Y-location offset of Block 3.")
    rotate3: Optional[float] = Field(None, description="Rotation angle of Block 3.")

    # Block 4 parameters
    material4: Optional[str] = Field(None, description="Material of Block 4.")
    height4: Optional[float] = Field(None, description="Height of Block 4.")
    pitch4: Optional[float] = Field(None, description="Pitch of Block 4.")
    x_expand4: Optional[float] = Field(None, description="X-expansion of Block 4.")
    y_expand4: Optional[float] = Field(None, description="Y-expansion of Block 4.")
    x_loc4: Optional[float] = Field(None, description="X-location offset of Block 4.")
    y_loc4: Optional[float] = Field(None, description="Y-location offset of Block 4.")
    rotate4: Optional[float] = Field(None, description="Rotation angle of Block 4.")


class SpectrumOutput(BaseModel):
    """
    Output model for predicted optical spectra.
    Each element in the list is a predicted spectrum (list of floats).
    """
    predictions: List[List[float]] = Field(..., description="List of predicted spectra, each spectrum is a list of float values.")
