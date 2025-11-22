"""
Configuration file for Natural Language Rationale Generation Agent
"""
import torch

# Black-box model for interpretation
MODEL_NAME = "textattack/bert-base-uncased-SST-2"

# Rationale generator model
GENERATOR_MODEL = "t5-small"

# Maximum sequence length
MAX_LENGTH = 512

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

