from typing import Optional

from pydantic import BaseModel, Field

from ..constants import (
    CompressionFineTuning,
    CompressionLevel,
    MAX_COMPRESSION_APPROXIMATION_LEVEL,
    MIN_COMPRESSION_APPROXIMATION_LEVEL,
)


class CompressionOptions(BaseModel):
    """Options for compression-in-training in variational optimization."""

    compression_level: CompressionLevel = CompressionLevel.BALANCED
    noise_profile: str = "default"
    fine_tuning: CompressionFineTuning = CompressionFineTuning.DISABLED
    approximation_level: Optional[int] = Field(
        default=None, ge=MIN_COMPRESSION_APPROXIMATION_LEVEL, le=MAX_COMPRESSION_APPROXIMATION_LEVEL
    )
