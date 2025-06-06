"""
API Gateway Module
Core gateway functionality using Portkey for AI API management,
including routing, load balancing, and conditional routing
"""
import uuid
import time
import json
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from fastapi import HTTPException, status
from portkey_ai import Portkey, PORTKEY_GATEWAY_URL, createHeaders
from config import AppConfig, SubscriptionTier
from user_management import user_manager, User
from subscription import subscription_manager
from admin import admin_manager, AIProvider
from billing import billing_manager, UsageType
from logging_module import logging_manager, LogLevel, EventType

class RoutingStrategy(Enum):
    SINGLE = "single"
    LOAD_BALANCE = "loadbalance"
    FALLBACK = "fallback"
    CONDITIONAL = "conditional"

class RequestStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    UNAUTHORIZED = "unauthorized"

@dataclass
class RoutingConfig:
    strategy: RoutingStrategy
    targets: List[Dict[str, Any]]
    conditions: Optional[Dict[str, Any]] = None

@dataclass
class APIRequest:
    request_id: str
    user_id: str
    virtual_key: str
    provider: str
    model: str
    endpoint: str
    payload: Dict[str, Any]
    headers: Dict[str, str]
    timestamp: datetime

@dataclass
class APIResponse:
    request_id: str
    status_code: int
    response_data: Dict[str, Any]
    latency_ms: float
    tokens_used: Dict[str, int]
    cost_credits: float
    cost_usd: float
    provider_response_id: Optional[str] = None
    error_message: Optional[str] = None

class APIGateway:
    """Main API Gateway class integrating Portkey for AI API management"""

    def __init__(self):
        """Initialize API Gateway with Portkey client"""
        self.portkey_client = None
        if AppConfig.PORTKEY_API_KEY:
            self.portkey_client = Portkey(api_key=AppConfig.PORTKEY_API_KEY)

        # Rate limiting storage (in production, use Redis)
        self.rate_limit_storage = {}

        # Routing configurations cache
        self.routing_configs = {}

    async def process_request(
        self,
        virtual_key: str,
        endpoint: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        ip_address: Optional[str] = None
    ) -> APIResponse:
        """Process incoming API request through the gateway"""

        request_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Authenticate user by virtual key
            user = user_manager.get_user_by_virtual_key(virtual_key)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid virtual key"
                )

            # Create API request object
            api_request = APIRequest(
                request_id=request_id,
                user_id=user.user_id,
                virtual_key=virtual_key,
                provider="",  # Will be determined by routing
                model=payload.get("model", ""),
                endpoint=endpoint,
                payload=payload,
                headers=headers,
                timestamp=datetime.utcnow()
            )

            # Check rate limits
            if not self._check_rate_limits(user, ip_address):
                logging_manager.log_rate_limit_event(
                    user_id=user.user_id,
                    endpoint=endpoint,
                    current_requests=self._get_current_requests(user.user_id),
                    limit=AppConfig.RATE_LIMIT_REQUESTS_PER_MINUTE[user.subscription_tier],
                    ip_address=ip_address
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )

            # Check subscription limits
            subscription = subscription_manager.get_user_subscription(user.user_id)
            if not subscription or subscription.status != "active":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="No active subscription"
                )

            # Check available models for user's subscription
            available_models = subscription_manager.get_available_models(user.user_id)
            if api_request.model not in available_models:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Model {api_request.model} not available in your subscription tier"
                )

            # Check user credits
            user_credits = billing_manager.get_user_credits(user.user_id)
            if not user_credits or user_credits.remaining_credits <= 0:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail="Insufficient credits"
                )

            # Determine routing strategy and route request
            routing_config = self._get_routing_config(user, api_request.model)
            response = await self._route_request(api_request, routing_config)

            # Calculate and record usage
            await self._record_usage(api_request, response)

            # Log successful request
            logging_manager.log_api_call(
                user_id=user.user_id,
                virtual_key=virtual_key,
                provider=response.request_id,  # This would be set in routing
                model=api_request.model,
                endpoint=endpoint,
                request_tokens=response.tokens_used.get("input", 0),
                response_tokens=response.tokens_used.get("output", 0),
                latency_ms=response.latency_ms,
                status_code=response.status_code,
                cost_credits=response.cost_credits,
                cost_usd=response.cost_usd,
                request_metadata={"payload": payload, "headers": headers},
                response_metadata=response.response_data,
                request_id=request_id
            )

            return response

        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Handle unexpected errors
            latency_ms = (time.time() - start_time) * 1000

            error_response = APIResponse(
                request_id=request_id,
                status_code=500,
                response_data={"error": str(e)},
                latency_ms=latency_ms,
                tokens_used={"input": 0, "output": 0},
                cost_credits=0.0,
                cost_usd=0.0,
                error_message=str(e)
            )

            # Log error
            logging_manager.log_event(
                level=LogLevel.ERROR,
                event_type=EventType.SYSTEM_ERROR,
                message=f"API Gateway error: {str(e)}",
                details={"request_id": request_id, "endpoint": endpoint, "error": str(e)},
                user_id=getattr(user, 'user_id', None) if 'user' in locals() else None,
                request_id=request_id,
                source="api_gateway"
            )

            return error_response

    async def _route_request(self, api_request: APIRequest, routing_config: RoutingConfig) -> APIResponse:
        """Route request based on configuration"""

        if routing_config.strategy == RoutingStrategy.SINGLE:
            return await self._route_single(api_request, routing_config.targets[0])
        elif routing_config.strategy == RoutingStrategy.LOAD_BALANCE:
            return await self._route_load_balance(api_request, routing_config.targets)
        elif routing_config.strategy == RoutingStrategy.FALLBACK:
            return await self._route_fallback(api_request, routing_config.targets)
        elif routing_config.strategy == RoutingStrategy.CONDITIONAL:
            return await self._route_conditional(api_request, routing_config)
        else:
            raise ValueError(f"Unsupported routing strategy: {routing_config.strategy}")

    async def _route_single(self, api_request: APIRequest, target: Dict[str, Any]) -> APIResponse:
        """Route to a single target"""
        return await self._make_provider_request(api_request, target)

    async def _route_load_balance(self, api_request: APIRequest, targets: List[Dict[str, Any]]) -> APIResponse:
        """Route with load balancing using weighted distribution"""

        # Calculate total weight
        total_weight = sum(target.get("weight", 1.0) for target in targets)

        # Generate random number for weighted selection
        import random
        random_value = random.random() * total_weight

        # Select target based on weight
        current_weight = 0
        selected_target = targets[0]  # Default fallback

        for target in targets:
            current_weight += target.get("weight", 1.0)
            if random_value <= current_weight:
                selected_target = target
                break

        return await self._make_provider_request(api_request, selected_target)

    async def _route_fallback(self, api_request: APIRequest, targets: List[Dict[str, Any]]) -> APIResponse:
        """Route with fallback - try targets in order until success"""

        last_error = None

        for target in targets:
            try:
                response = await self._make_provider_request(api_request, target)
                if response.status_code == 200:
                    return response
                last_error = response.error_message
            except Exception as e:
                last_error = str(e)
                continue

        # All targets failed
        return APIResponse(
            request_id=api_request.request_id,
            status_code=500,
            response_data={"error": "All fallback targets failed"},
            latency_ms=0,
            tokens_used={"input": 0, "output": 0},
            cost_credits=0.0,
            cost_usd=0.0,
            error_message=last_error
        )

    async def _route_conditional(self, api_request: APIRequest, routing_config: RoutingConfig) -> APIResponse:
        """Route based on conditions (e.g., user tier, metadata)"""

        user = user_manager.get_user_by_id(api_request.user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Evaluate conditions
        for condition in routing_config.conditions.get("rules", []):
            if self._evaluate_condition(condition, user, api_request):
                target_index = condition["target_index"]
                if target_index < len(routing_config.targets):
                    return await self._make_provider_request(
                        api_request, 
                        routing_config.targets[target_index]
                    )

        # Default target if no condition matches
        default_target = routing_config.targets[0] if routing_config.targets else None
        if not default_target:
            raise HTTPException(status_code=500, detail="No routing target available")

        return await self._make_provider_request(api_request, default_target)

    async def _make_provider_request(self, api_request: APIRequest, target: Dict[str, Any]) -> APIResponse:
        """Make actual request to AI provider through Portkey"""

        start_time = time.time()

        try:
            # Get virtual key mapping for the provider
            provider = AIProvider(target["provider"])
            mapping = admin_manager.get_user_provider_mapping(api_request.user_id, provider)

            if not mapping or not mapping.portkey_virtual_key:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No virtual key mapping found for provider {provider.value}"
                )

            # Prepare Portkey request
            headers = createHeaders(
                api_key=AppConfig.PORTKEY_API_KEY,
                virtual_key=mapping.portkey_virtual_key,
                config=mapping.config_id
            )

            # Use appropriate Portkey client method based on endpoint
            if "chat/completions" in api_request.endpoint:
                response = await self._make_chat_completion_request(
                    api_request.payload, 
                    headers, 
                    target
                )
            elif "completions" in api_request.endpoint:
                response = await self._make_completion_request(
                    api_request.payload, 
                    headers, 
                    target
                )
            elif "embeddings" in api_request.endpoint:
                response = await self._make_embeddings_request(
                    api_request.payload, 
                    headers, 
                    target
                )
            else:
                # Generic request
                response = await self._make_generic_request(
                    api_request, 
                    headers, 
                    target
                )

            latency_ms = (time.time() - start_time) * 1000

            # Extract tokens and calculate cost
            tokens_used = self._extract_token_usage(response)
            cost_credits, cost_usd = self._calculate_request_cost(
                api_request.model, 
                tokens_used
            )

            return APIResponse(
                request_id=api_request.request_id,
                status_code=200,
                response_data=response,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost_credits=cost_credits,
                cost_usd=cost_usd,
                provider_response_id=response.get("id")
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            return APIResponse(
                request_id=api_request.request_id,
                status_code=500,
                response_data={"error": str(e)},
                latency_ms=latency_ms,
                tokens_used={"input": 0, "output": 0},
                cost_credits=0.0,
                cost_usd=0.0,
                error_message=str(e)
            )

    async def _make_chat_completion_request(self, payload: Dict[str, Any], headers: Dict[str, str], target: Dict[str, Any]) -> Dict[str, Any]:
        """Make chat completion request through Portkey"""

        if not self.portkey_client:
            raise HTTPException(status_code=500, detail="Portkey client not configured")

        # Use Portkey client for chat completion
        response = self.portkey_client.chat.completions.create(
            model=payload.get("model"),
            messages=payload.get("messages", []),
            temperature=payload.get("temperature", 1.0),
            max_tokens=payload.get("max_tokens"),
            top_p=payload.get("top_p", 1.0),
            stream=payload.get("stream", False)
        )

        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    async def _make_completion_request(self, payload: Dict[str, Any], headers: Dict[str, str], target: Dict[str, Any]) -> Dict[str, Any]:
        """Make completion request through Portkey"""

        if not self.portkey_client:
            raise HTTPException(status_code=500, detail="Portkey client not configured")

        # Use Portkey client for completion
        response = self.portkey_client.completions.create(
            model=payload.get("model"),
            prompt=payload.get("prompt", ""),
            max_tokens=payload.get("max_tokens", 100),
            temperature=payload.get("temperature", 1.0)
        )

        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    async def _make_embeddings_request(self, payload: Dict[str, Any], headers: Dict[str, str], target: Dict[str, Any]) -> Dict[str, Any]:
        """Make embeddings request through Portkey"""

        if not self.portkey_client:
            raise HTTPException(status_code=500, detail="Portkey client not configured")

        # Use Portkey client for embeddings
        response = self.portkey_client.embeddings.create(
            model=payload.get("model"),
            input=payload.get("input", "")
        )

        return response.model_dump() if hasattr(response, 'model_dump') else dict(response)

    async def _make_generic_request(self, api_request: APIRequest, headers: Dict[str, str], target: Dict[str, Any]) -> Dict[str, Any]:
        """Make generic request through Portkey"""

        # For other endpoints, use HTTP client with Portkey headers
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PORTKEY_GATEWAY_URL}/{api_request.endpoint}",
                json=api_request.payload,
                headers=headers,
                timeout=30.0
            )

            response.raise_for_status()
            return response.json()

    def _get_routing_config(self, user: User, model: str) -> RoutingConfig:
        """Get routing configuration based on user subscription and model"""

        # Check subscription features
        subscription_limits = subscription_manager.check_subscription_limits(
            user.user_id, 
            "conditional_routing"
        )

        if subscription_limits["allowed"] and user.subscription_tier == SubscriptionTier.PREMIUM:
            # Premium users get conditional routing
            return RoutingConfig(
                strategy=RoutingStrategy.CONDITIONAL,
                targets=[
                    {"provider": "openai", "weight": 0.7},
                    {"provider": "anthropic", "weight": 0.3}
                ],
                conditions={
                    "rules": [
                        {
                            "condition": {"subscription_tier": "premium"},
                            "target_index": 0
                        }
                    ]
                }
            )
        elif user.subscription_tier == SubscriptionTier.PROFESSIONAL:
            # Professional users get load balancing
            return RoutingConfig(
                strategy=RoutingStrategy.LOAD_BALANCE,
                targets=[
                    {"provider": "openai", "weight": 0.6},
                    {"provider": "anthropic", "weight": 0.4}
                ]
            )
        else:
            # Free users get single routing
            return RoutingConfig(
                strategy=RoutingStrategy.SINGLE,
                targets=[{"provider": "openai", "weight": 1.0}]
            )

    def _check_rate_limits(self, user: User, ip_address: Optional[str] = None) -> bool:
        """Check if user is within rate limits"""

        current_time = time.time()
        limit = AppConfig.RATE_LIMIT_REQUESTS_PER_MINUTE[user.subscription_tier]

        # Clean old entries (older than 1 minute)
        if user.user_id in self.rate_limit_storage:
            self.rate_limit_storage[user.user_id] = [
                timestamp for timestamp in self.rate_limit_storage[user.user_id]
                if current_time - timestamp < 60
            ]
        else:
            self.rate_limit_storage[user.user_id] = []

        # Check current count
        current_count = len(self.rate_limit_storage[user.user_id])

        if current_count >= limit:
            return False

        # Add current request
        self.rate_limit_storage[user.user_id].append(current_time)
        return True

    def _get_current_requests(self, user_id: str) -> int:
        """Get current request count for user"""
        if user_id not in self.rate_limit_storage:
            return 0
        return len(self.rate_limit_storage[user_id])

    def _evaluate_condition(self, condition: Dict[str, Any], user: User, api_request: APIRequest) -> bool:
        """Evaluate routing condition"""

        condition_rules = condition.get("condition", {})

        for key, value in condition_rules.items():
            if key == "subscription_tier":
                if user.subscription_tier.value != value:
                    return False
            elif key == "model":
                if api_request.model != value:
                    return False
            # Add more condition types as needed

        return True

    def _extract_token_usage(self, response: Dict[str, Any]) -> Dict[str, int]:
        """Extract token usage from provider response"""

        usage = response.get("usage", {})

        return {
            "input": usage.get("prompt_tokens", 0),
            "output": usage.get("completion_tokens", 0),
            "total": usage.get("total_tokens", 0)
        }

    def _calculate_request_cost(self, model: str, tokens_used: Dict[str, int]) -> Tuple[float, float]:
        """Calculate cost in credits and USD for the request"""

        # Use billing manager to calculate cost
        input_cost_usd, input_credits = billing_manager._calculate_cost(
            model, UsageType.TOKEN_INPUT, tokens_used.get("input", 0)
        )

        output_cost_usd, output_credits = billing_manager._calculate_cost(
            model, UsageType.TOKEN_OUTPUT, tokens_used.get("output", 0)
        )

        total_cost_usd = input_cost_usd + output_cost_usd
        total_credits = input_credits + output_credits

        return total_credits, total_cost_usd

    async def _record_usage(self, api_request: APIRequest, response: APIResponse):
        """Record usage in billing system"""

        if response.tokens_used.get("input", 0) > 0:
            billing_manager.record_usage(
                user_id=api_request.user_id,
                usage_type=UsageType.TOKEN_INPUT,
                provider=api_request.provider,
                model=api_request.model,
                quantity=response.tokens_used["input"],
                request_id=api_request.request_id
            )

        if response.tokens_used.get("output", 0) > 0:
            billing_manager.record_usage(
                user_id=api_request.user_id,
                usage_type=UsageType.TOKEN_OUTPUT,
                provider=api_request.provider,
                model=api_request.model,
                quantity=response.tokens_used["output"],
                request_id=api_request.request_id
            )

# Initialize global API gateway instance
api_gateway = APIGateway()
