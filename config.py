"""
Configuration management for Gen AI API Gateway
"""
import os
from typing import Dict, Any
from enum import Enum

class SubscriptionTier(Enum):
    FREE = "free"
    PROFESSIONAL = "professional"
    PREMIUM = "premium"

class AppConfig:
    """Central configuration class"""

    # Database Configuration
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    # Portkey Configuration
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY", "")
    PORTKEY_GATEWAY_URL = "https://api.portkey.ai/v1"

    # FastAPI Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

    # Subscription Limits
    SUBSCRIPTION_LIMITS = {
        SubscriptionTier.FREE: {
            "api_calls_per_month": 1000,
            "models_access": ["gpt-3.5-turbo", "claude-3-haiku"],
            "features": ["basic_routing"],
            "cost_limit_usd": 10.0
        },
        SubscriptionTier.PROFESSIONAL: {
            "api_calls_per_month": 50000,
            "models_access": ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-haiku"],
            "features": ["basic_routing", "load_balancing", "fallback"],
            "cost_limit_usd": 500.0
        },
        SubscriptionTier.PREMIUM: {
            "api_calls_per_month": -1,  # Unlimited
            "models_access": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
            "features": ["basic_routing", "load_balancing", "fallback", "conditional_routing", "priority_support"],
            "cost_limit_usd": -1  # Unlimited
        }
    }

    # Subscription Pricing (USD per month)
    SUBSCRIPTION_PRICING = {
        SubscriptionTier.FREE: 0.0,
        SubscriptionTier.PROFESSIONAL: 29.99,
        SubscriptionTier.PREMIUM: 99.99
    }

    # OAuth Configuration
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
    APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID", "")
    APPLE_CLIENT_SECRET = os.getenv("APPLE_CLIENT_SECRET", "")

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "app.log")

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = {
        SubscriptionTier.FREE: 10,
        SubscriptionTier.PROFESSIONAL: 100,
        SubscriptionTier.PREMIUM: 1000
    }

    @classmethod
    def get_subscription_config(cls, tier: SubscriptionTier) -> Dict[str, Any]:
        """Get configuration for a specific subscription tier"""
        return cls.SUBSCRIPTION_LIMITS.get(tier, cls.SUBSCRIPTION_LIMITS[SubscriptionTier.FREE])

    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present"""
        required_configs = [
            "PORTKEY_API_KEY",
            "SECRET_KEY"
        ]

        missing_configs = []
        for config in required_configs:
            if not getattr(cls, config):
                missing_configs.append(config)

        if missing_configs:
            raise ValueError(f"Missing required configuration: {missing_configs}")

        return True

# Load environment variables
def load_env():
    """Load environment variables from .env file if it exists"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# Initialize configuration
load_env()
