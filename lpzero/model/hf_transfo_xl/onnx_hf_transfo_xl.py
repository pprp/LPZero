# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Huggingface's Transformer-XL for ONNX.
"""

from typing import Any, Dict

from lpzero.model.config_base import OnnxConfig


class HfTransfoXLOnnxConfig(OnnxConfig):
    """Huggingface's Transformer-XL ONNX-based configuration.

    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        """Initializes the class by setting missing keys on incoming
            model's configuration.

        Args:
            model_config: Configuration of the model that will be exported.

        """

        super().__init__(model_config, model_type='transfo-xl')
