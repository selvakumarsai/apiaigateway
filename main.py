"""
Main FastAPI Application
Provides web interface and API endpoints for the Gen AI API Gateway
"""
import os
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, Depends, status, Request, Form, File, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, EmailStr
import uvicorn

# Import our modules
from config import AppConfig, SubscriptionTier
from user_management import user_manager, User, AuthProvider
from subscription import subscription_manager, PaymentMethod
from admin import admin_manager, AIProvider
from billing import billing_manager
from logging_module import logging_manager
from api_gateway import api_gateway

# Initialize FastAPI app
app = FastAPI(
    title="Gen AI API Gateway",
    description="A comprehensive API gateway for AI models with subscription management",
    version="1.0.0"
)

# Setup static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static/templates")

# Security
security = HTTPBearer()

# Pydantic models for request/response
class UserRegistration(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class SubscriptionUpgrade(BaseModel):
    tier: SubscriptionTier
    payment_method: PaymentMethod

class ProviderKeyAdd(BaseModel):
    provider: AIProvider
    api_key: str
    description: str

class AIRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class CreditPurchase(BaseModel):
    amount: float
    payment_method: PaymentMethod

# Helper functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    payload = user_manager.verify_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

    user = user_manager.get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user and verify admin privileges"""
    # In a real application, you would check admin role
    # For now, assume premium users have admin access
    if current_user.subscription_tier != SubscriptionTier.PREMIUM:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# Web interface routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """User dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Admin panel"""
    return templates.TemplateResponse("admin.html", {"request": request})

# Authentication endpoints
@app.post("/api/auth/register")
async def register_user(user_data: UserRegistration):
    """Register a new user"""
    try:
        user = user_manager.register_user_email(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )

        # Initialize user credits
        billing_manager.initialize_user_credits(user.user_id, user.subscription_tier)

        # Create subscription
        subscription_manager.create_subscription(user.user_id, user.subscription_tier)

        # Generate access token
        token = user_manager.create_access_token(user)

        return {
            "message": "User registered successfully",
            "user_id": user.user_id,
            "access_token": token,
            "token_type": "bearer"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login_user(user_data: UserLogin):
    """Login user"""
    user = user_manager.authenticate_user(user_data.email, user_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    token = user_manager.create_access_token(user)

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user.user_id,
            "email": user.email,
            "full_name": user.full_name,
            "subscription_tier": user.subscription_tier.value
        }
    }

@app.get("/api/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    subscription = subscription_manager.get_user_subscription(current_user.user_id)
    credits = billing_manager.get_user_credits(current_user.user_id)

    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "subscription_tier": current_user.subscription_tier.value,
        "virtual_key": current_user.virtual_key,
        "subscription": {
            "status": subscription.status if subscription else "none",
            "end_date": subscription.end_date.isoformat() if subscription else None
        },
        "credits": {
            "total": credits.total_credits if credits else 0,
            "used": credits.used_credits if credits else 0,
            "remaining": credits.remaining_credits if credits else 0
        }
    }

# Subscription management endpoints
@app.post("/api/subscription/upgrade")
async def upgrade_subscription(
    upgrade_data: SubscriptionUpgrade,
    current_user: User = Depends(get_current_user)
):
    """Upgrade user subscription"""
    result = subscription_manager.upgrade_subscription(
        user_id=current_user.user_id,
        new_tier=upgrade_data.tier,
        payment_method=upgrade_data.payment_method
    )

    if result["success"]:
        # Update user subscription tier in user management
        user_manager.update_subscription(current_user.user_id, upgrade_data.tier)

        return {
            "message": "Subscription upgraded successfully",
            "new_tier": upgrade_data.tier.value,
            "payment_id": result["payment_id"]
        }
    else:
        raise HTTPException(status_code=400, detail=result["error"])

@app.get("/api/subscription/models")
async def get_available_models(current_user: User = Depends(get_current_user)):
    """Get available AI models for user's subscription"""
    models = subscription_manager.get_available_models(current_user.user_id)

    return {"available_models": models}

# Billing endpoints
@app.post("/api/billing/credits/purchase")
async def purchase_credits(
    purchase_data: CreditPurchase,
    current_user: User = Depends(get_current_user)
):
    """Purchase additional credits"""
    success = billing_manager.add_credits(
        user_id=current_user.user_id,
        amount=purchase_data.amount,
        description=f"Credit purchase - {purchase_data.amount} credits"
    )

    if success:
        return {
            "message": "Credits purchased successfully",
            "amount": purchase_data.amount
        }
    else:
        raise HTTPException(status_code=400, detail="Failed to purchase credits")

@app.get("/api/billing/usage")
async def get_usage_summary(
    days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Get usage summary for user"""
    summary = billing_manager.get_usage_summary(current_user.user_id, days)
    return summary

@app.get("/api/billing/report")
async def get_billing_report(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Get detailed billing report"""

    if start_date:
        start_dt = datetime.fromisoformat(start_date)
    else:
        start_dt = datetime.utcnow() - timedelta(days=30)

    if end_date:
        end_dt = datetime.fromisoformat(end_date)
    else:
        end_dt = datetime.utcnow()

    report = billing_manager.get_billing_report(current_user.user_id, start_dt, end_dt)
    return report

# AI API Gateway endpoints
@app.post("/api/ai/chat/completions")
async def chat_completions(
    request: Request,
    ai_request: AIRequest,
    current_user: User = Depends(get_current_user)
):
    """Chat completions endpoint"""

    # Get client IP
    client_ip = request.client.host

    # Process through API gateway
    response = await api_gateway.process_request(
        virtual_key=current_user.virtual_key,
        endpoint="chat/completions",
        payload=ai_request.dict(),
        headers=dict(request.headers),
        ip_address=client_ip
    )

    if response.status_code == 200:
        return response.response_data
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.error_message or "API request failed"
        )

@app.post("/api/ai/completions")
async def completions(
    request: Request,
    ai_request: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Completions endpoint"""

    client_ip = request.client.host

    response = await api_gateway.process_request(
        virtual_key=current_user.virtual_key,
        endpoint="completions",
        payload=ai_request,
        headers=dict(request.headers),
        ip_address=client_ip
    )

    if response.status_code == 200:
        return response.response_data
    else:
        raise HTTPException(
            status_code=response.status_code,
            detail=response.error_message or "API request failed"
        )

# Admin endpoints
@app.post("/api/admin/provider-keys")
async def add_provider_key(
    key_data: ProviderKeyAdd,
    admin_user: User = Depends(get_admin_user)
):
    """Add AI provider API key"""

    provider_key = admin_manager.add_provider_api_key(
        provider=key_data.provider,
        api_key=key_data.api_key,
        description=key_data.description,
        admin_user_id=admin_user.user_id
    )

    return {
        "message": "Provider API key added successfully",
        "key_id": provider_key.key_id,
        "provider": provider_key.provider.value
    }

@app.get("/api/admin/provider-keys")
async def list_provider_keys(
    provider: Optional[AIProvider] = None,
    admin_user: User = Depends(get_admin_user)
):
    """List provider API keys"""

    keys = admin_manager.list_provider_api_keys(provider)

    return {
        "provider_keys": [
            {
                "key_id": key.key_id,
                "provider": key.provider.value,
                "description": key.description,
                "is_active": key.is_active,
                "created_at": key.created_at.isoformat(),
                "usage_count": key.usage_count
            }
            for key in keys
        ]
    }

@app.post("/api/admin/virtual-mappings")
async def create_virtual_mapping(
    user_id: str,
    provider: AIProvider,
    provider_key_id: Optional[str] = None,
    admin_user: User = Depends(get_admin_user)
):
    """Create virtual key mapping for user"""

    # Get user's virtual key
    user = user_manager.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    mapping = admin_manager.create_virtual_key_mapping(
        user_virtual_key=user.virtual_key,
        user_id=user_id,
        provider=provider,
        admin_user_id=admin_user.user_id,
        provider_key_id=provider_key_id
    )

    return {
        "message": "Virtual key mapping created successfully",
        "mapping_id": mapping.mapping_id,
        "portkey_virtual_key": mapping.portkey_virtual_key
    }

@app.get("/api/admin/stats")
async def get_system_stats(admin_user: User = Depends(get_admin_user)):
    """Get system statistics"""

    stats = admin_manager.get_system_stats()

    # Add performance metrics
    performance_metrics = logging_manager.get_api_performance_metrics(24)
    error_summary = logging_manager.get_error_summary(24)

    return {
        "system_stats": stats,
        "performance_metrics": performance_metrics,
        "error_summary": error_summary
    }

# Logging endpoints
@app.get("/api/logs/api-calls")
async def get_api_call_logs(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get API call logs for current user"""

    logs = logging_manager.get_api_call_logs(
        user_id=current_user.user_id,
        limit=limit
    )

    return {
        "logs": [
            {
                "call_id": log.call_id,
                "provider": log.provider,
                "model": log.model,
                "status_code": log.status_code,
                "latency_ms": log.latency_ms,
                "cost_credits": log.cost_credits,
                "timestamp": log.timestamp.isoformat()
            }
            for log in logs
        ]
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logging_manager.log_event(
        level=logging_manager.LogLevel.ERROR,
        event_type=logging_manager.EventType.SYSTEM_ERROR,
        message=f"Unhandled exception: {str(exc)}",
        details={"path": str(request.url), "method": request.method},
        source="main_app"
    )

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Validate configuration
    AppConfig.validate_config()

    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
