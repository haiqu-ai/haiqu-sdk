from typing import Literal, Optional
from pydantic import BaseModel


class CompressionOptions(BaseModel):
    """Options for compression-in-training in variational optimization."""

    compression_level: Literal["low", "balanced", "high"] = "balanced"
    noise_profile: str = "default"
    fine_tuning: Literal["disabled", "low", "heavy"] = "disabled"
    approximation_level: Optional[int] = None
