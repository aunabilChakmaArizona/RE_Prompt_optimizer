"""
Configuration for cluster evolution system.
"""

AZURE_API_KEY = None  # Will be loaded from config
AZURE_ENDPOINT = None  # Will be loaded from config
AZURE_API_VERSION = "2024-05-01-preview"
# BACKBONE_MODEL = "gpt-4.1-nano"
BACKBONE_MODEL = "gpt-4.1-mini"
# BACKBONE_MODEL = "gpt-4o-mini"
# BACKBONE_MODEL = "openai/gpt-oss-20b"
# BACKBONE_MODEL = "deepseek/deepseek-chat-v3.1"
# BACKBONE_MODEL = "openai/gpt-oss-120b"
# BACKBONE_MODEL = "gemma-3-27b-it"
# BACKBONE_MODEL = "gemma-3-12b-it"
# BACKBONE_MODEL = "gemini-2.0-flash-lite"
# BACKBONE_MODEL = "gemini-2.0-flash"
# BACKBONE_MODEL = "gemini-2.5-flash-lite"
# BACKBONE_MODEL = "gemini-2.5-flash"
# AGENT_MODEL = "gemini-2.5-flash"
AGENT_MODEL = "gpt-4.1"
# AGENT_MODEL = "gpt-5"
JUDGE_MODEL = "gpt-5"
RETRIES = 3

USE_OPENAI = False

# Per-model worker token budget configuration
WORKER_TOKEN_CONFIG = {
    # Model-specific token budgets (tokens per worker per minute)
    "model_budgets": {
        "gpt-4.1-nano": 80_000,    # Higher budget for nano (cheaper, faster model)
        "gpt-4o-mini": 80_000,
        "openai/gpt-oss-20b": 80_000,
        "openai/gpt-oss-120b": 200_000,
        "gemma-3-27b-it": 80_000,
        "gemma-3-12b-it": 80_000,
        "gemini-2.0-flash-lite": 200_000,
        "gemini-2.0-flash": 200_000,
        "gemini-2.5-flash-lite": 200_000,
        "gemini-2.5-flash": 200_000,
        "deepseek/deepseek-chat-v3.1": 200_000,
        "gpt-4.1-mini": 200_000,    # Medium budget for mini
        "gpt-4.1": 200_000,          # Lower budget for full GPT-4.1 (expensive)
        "gpt-5": 200_000,
        "default": 15_000          # Default for unknown models
    },
    # Alert thresholds per model (optional, can also be model-specific)
    "large_token_alert_thresholds": {
        "gpt-4.1-nano": 70_000,
        "gpt-4o-mini": 70_000,
        "openai/gpt-oss-20b": 70_000,
        "openai/gpt-oss-120b": 190_000,
        "gemma-3-27b-it": 70_000,
        "gemma-3-12b-it": 70_000,
        "gemini-2.0-flash-lite": 190_000,
        "gemini-2.0-flash": 190_000,
        "gemini-2.5-flash-lite": 190_000,
        "gemini-2.5-flash": 190_000,
        "deepseek/deepseek-chat-v3.1": 190_000,
        "gpt-4.1-mini": 190_000,
        "gpt-4.1": 190_000,
        "gpt-5": 190_000,
        "default": 14_000
    }
}

# Model capacity configuration (tokens per minute)
MODEL_CAPACITY = {
    "gpt-4.1-nano": 10_000_000,
    "gpt-4o-mini": 10_000_000,
    "openai/gpt-oss-20b": 10_000_000,
    "openai/gpt-oss-120b": 25_000_000,
    "gemma-3-27b-it": 10_000_000,
    "gemma-3-12b-it": 10_000_000,
    "gemini-2.0-flash-lite": 25_000_000,
    "gemini-2.0-flash": 25_000_000,
    "gemini-2.5-flash-lite": 25_000_000,
    "gemini-2.5-flash": 25_000_000,
    "deepseek/deepseek-chat-v3.1": 25_000_000,
    "gpt-4.1-mini": 25_000_000,
    "gpt-4.1": 25_000_000,
    "gpt-5": 25_000_000,
    "default": 250_000  # Conservative fallback
}

"""
# Example for Azure configuration with lower limits:
USE_OPENAI = False

WORKER_TOKEN_CONFIG = {
    "model_budgets": {
        "gpt-4.1-nano": 80_000,
        "gpt-4o-mini": 128_000,
        "gpt-4.1-mini": 200_000,
        "gpt-4.1": 50_000,
        "gpt-5": 50_000,
        "default": 15_000
    },
    "large_token_alert_thresholds": {
        "gpt-4.1-nano": 70_000,
        "gpt-4o-mini": 118_000,
        "gpt-4.1-mini": 190_000,
        "gpt-4.1": 40_000,
        "gpt-5": 40_000,
        "default": 14_000
    }
}

MODEL_CAPACITY = {
    "gpt-4.1-nano": 2_500_000,
    "gpt-4o-mini": 16_000_000,
    "gpt-4.1-mini": 2_500_000,
    "gpt-4.1": 1_500_000,
    "gpt-5": 1_500_000,
    "default": 250_000
}
"""
