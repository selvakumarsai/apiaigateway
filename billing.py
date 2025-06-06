"""
Billing Module
Handles credit tracking, usage monitoring, cost calculation, and billing reports
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException, status
from config import AppConfig, SubscriptionTier

class TransactionType(Enum):
    CREDIT_PURCHASE = "credit_purchase"
    API_USAGE = "api_usage"
    SUBSCRIPTION_PAYMENT = "subscription_payment"
    REFUND = "refund"
    BONUS_CREDIT = "bonus_credit"

class UsageType(Enum):
    API_CALL = "api_call"
    TOKEN_INPUT = "token_input"
    TOKEN_OUTPUT = "token_output"
    IMAGE_GENERATION = "image_generation"
    AUDIO_TRANSCRIPTION = "audio_transcription"

@dataclass
class CreditBalance:
    user_id: str
    total_credits: float
    used_credits: float
    remaining_credits: float
    last_updated: datetime
    subscription_tier: SubscriptionTier

@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    transaction_type: TransactionType
    amount: float
    currency: str
    description: str
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class UsageRecord:
    usage_id: str
    user_id: str
    usage_type: UsageType
    provider: str
    model: str
    credits_used: float
    cost_usd: float
    quantity: int
    created_at: datetime
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class BillingManager:
    """Manages billing, credits, and usage tracking"""

    def __init__(self):
        """Initialize ChromaDB client and create collections"""
        self.client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self._setup_collections()

        # Model pricing (credits per 1K tokens or per request)
        self.model_pricing = {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "dall-e-3": {"request": 0.04},
            "whisper-1": {"minute": 0.006}
        }

    def _setup_collections(self):
        """Setup ChromaDB collections for billing data"""
        try:
            self.credit_balances_collection = self.client.get_collection("credit_balances")
        except:
            self.credit_balances_collection = self.client.create_collection(
                name="credit_balances",
                metadata={"description": "User credit balance tracking"}
            )

        try:
            self.transactions_collection = self.client.get_collection("transactions")
        except:
            self.transactions_collection = self.client.create_collection(
                name="transactions",
                metadata={"description": "Financial transaction records"}
            )

        try:
            self.usage_records_collection = self.client.get_collection("usage_records")
        except:
            self.usage_records_collection = self.client.create_collection(
                name="usage_records",
                metadata={"description": "API usage tracking records"}
            )

    def initialize_user_credits(self, user_id: str, subscription_tier: SubscriptionTier) -> CreditBalance:
        """Initialize credit balance for new user"""

        # Initial credits based on subscription tier
        initial_credits = {
            SubscriptionTier.FREE: 10.0,
            SubscriptionTier.PROFESSIONAL: 100.0,
            SubscriptionTier.PREMIUM: 500.0
        }

        credits = initial_credits.get(subscription_tier, 10.0)

        balance = CreditBalance(
            user_id=user_id,
            total_credits=credits,
            used_credits=0.0,
            remaining_credits=credits,
            last_updated=datetime.utcnow(),
            subscription_tier=subscription_tier
        )

        # Store in ChromaDB
        self.credit_balances_collection.add(
            documents=[f"Credit balance for user {user_id}"],
            metadatas=[{
                "user_id": user_id,
                "total_credits": credits,
                "used_credits": 0.0,
                "remaining_credits": credits,
                "last_updated": datetime.utcnow().isoformat(),
                "subscription_tier": subscription_tier.value
            }],
            ids=[user_id]
        )

        # Record initial credit transaction
        self.record_transaction(
            user_id=user_id,
            transaction_type=TransactionType.BONUS_CREDIT,
            amount=credits,
            currency="CREDITS",
            description=f"Initial credits for {subscription_tier.value} subscription",
            metadata={"tier": subscription_tier.value}
        )

        return balance

    def get_user_credits(self, user_id: str) -> Optional[CreditBalance]:
        """Get current credit balance for user"""
        try:
            balance_data = self.credit_balances_collection.get(ids=[user_id])
            if not balance_data["ids"]:
                return None

            return self._metadata_to_credit_balance(balance_data["metadatas"][0])
        except:
            return None

    def add_credits(
        self, 
        user_id: str, 
        amount: float, 
        transaction_type: TransactionType = TransactionType.CREDIT_PURCHASE,
        description: str = "Credit purchase"
    ) -> bool:
        """Add credits to user account"""

        balance = self.get_user_credits(user_id)
        if not balance:
            return False

        try:
            # Update balance
            new_total = balance.total_credits + amount
            new_remaining = balance.remaining_credits + amount

            updated_metadata = {
                "user_id": user_id,
                "total_credits": new_total,
                "used_credits": balance.used_credits,
                "remaining_credits": new_remaining,
                "last_updated": datetime.utcnow().isoformat(),
                "subscription_tier": balance.subscription_tier.value
            }

            self.credit_balances_collection.update(
                ids=[user_id],
                metadatas=[updated_metadata]
            )

            # Record transaction
            self.record_transaction(
                user_id=user_id,
                transaction_type=transaction_type,
                amount=amount,
                currency="CREDITS",
                description=description
            )

            return True
        except:
            return False

    def deduct_credits(
        self, 
        user_id: str, 
        amount: float, 
        description: str = "API usage",
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Deduct credits from user account"""

        balance = self.get_user_credits(user_id)
        if not balance:
            return False

        if balance.remaining_credits < amount:
            return False

        try:
            # Update balance
            new_used = balance.used_credits + amount
            new_remaining = balance.remaining_credits - amount

            updated_metadata = {
                "user_id": user_id,
                "total_credits": balance.total_credits,
                "used_credits": new_used,
                "remaining_credits": new_remaining,
                "last_updated": datetime.utcnow().isoformat(),
                "subscription_tier": balance.subscription_tier.value
            }

            self.credit_balances_collection.update(
                ids=[user_id],
                metadatas=[updated_metadata]
            )

            # Record transaction
            self.record_transaction(
                user_id=user_id,
                transaction_type=TransactionType.API_USAGE,
                amount=-amount,  # Negative for deduction
                currency="CREDITS",
                description=description,
                metadata=metadata or {}
            )

            return True
        except:
            return False

    def record_usage(
        self,
        user_id: str,
        usage_type: UsageType,
        provider: str,
        model: str,
        quantity: int,
        request_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> UsageRecord:
        """Record API usage and calculate cost"""

        # Calculate cost and credits
        cost_usd, credits_used = self._calculate_cost(model, usage_type, quantity)

        usage_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        usage_record = UsageRecord(
            usage_id=usage_id,
            user_id=user_id,
            usage_type=usage_type,
            provider=provider,
            model=model,
            credits_used=credits_used,
            cost_usd=cost_usd,
            quantity=quantity,
            created_at=created_at,
            request_id=request_id,
            metadata=metadata or {}
        )

        # Store usage record
        self.usage_records_collection.add(
            documents=[f"Usage: {model} - {quantity} {usage_type.value}"],
            metadatas=[{
                "usage_id": usage_id,
                "user_id": user_id,
                "usage_type": usage_type.value,
                "provider": provider,
                "model": model,
                "credits_used": credits_used,
                "cost_usd": cost_usd,
                "quantity": quantity,
                "created_at": created_at.isoformat(),
                "request_id": request_id,
                "metadata": metadata or {}
            }],
            ids=[usage_id]
        )

        # Deduct credits from user balance
        self.deduct_credits(
            user_id=user_id,
            amount=credits_used,
            description=f"{model} usage - {quantity} {usage_type.value}",
            metadata={
                "usage_id": usage_id,
                "model": model,
                "provider": provider,
                "quantity": quantity
            }
        )

        return usage_record

    def get_usage_summary(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage summary for user over specified days"""

        start_date = datetime.utcnow() - timedelta(days=days)

        # Get usage records for the period
        usage_records = self.usage_records_collection.query(
            query_texts=[""],
            n_results=1000,
            where={"user_id": user_id}
        )

        if not usage_records["metadatas"][0]:
            return {
                "total_requests": 0,
                "total_cost_usd": 0.0,
                "total_credits_used": 0.0,
                "by_model": {},
                "by_provider": {},
                "period_days": days
            }

        # Filter by date and calculate summary
        total_requests = 0
        total_cost_usd = 0.0
        total_credits_used = 0.0
        by_model = {}
        by_provider = {}

        for metadata in usage_records["metadatas"][0]:
            created_at = datetime.fromisoformat(metadata["created_at"])
            if created_at >= start_date:
                total_requests += 1
                total_cost_usd += metadata["cost_usd"]
                total_credits_used += metadata["credits_used"]

                # Group by model
                model = metadata["model"]
                if model not in by_model:
                    by_model[model] = {"requests": 0, "cost_usd": 0.0, "credits_used": 0.0}
                by_model[model]["requests"] += 1
                by_model[model]["cost_usd"] += metadata["cost_usd"]
                by_model[model]["credits_used"] += metadata["credits_used"]

                # Group by provider
                provider = metadata["provider"]
                if provider not in by_provider:
                    by_provider[provider] = {"requests": 0, "cost_usd": 0.0, "credits_used": 0.0}
                by_provider[provider]["requests"] += 1
                by_provider[provider]["cost_usd"] += metadata["cost_usd"]
                by_provider[provider]["credits_used"] += metadata["credits_used"]

        return {
            "total_requests": total_requests,
            "total_cost_usd": round(total_cost_usd, 4),
            "total_credits_used": round(total_credits_used, 2),
            "by_model": by_model,
            "by_provider": by_provider,
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": datetime.utcnow().isoformat()
        }

    def get_billing_report(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive billing report for user"""

        # Get transactions in period
        transactions = self.transactions_collection.query(
            query_texts=[""],
            n_results=1000,
            where={"user_id": user_id}
        )

        # Get usage records in period
        usage_records = self.usage_records_collection.query(
            query_texts=[""],
            n_results=1000,
            where={"user_id": user_id}
        )

        # Process transactions
        total_credits_purchased = 0.0
        total_credits_used = 0.0
        total_amount_paid = 0.0

        transaction_summary = {
            TransactionType.CREDIT_PURCHASE.value: {"count": 0, "amount": 0.0},
            TransactionType.API_USAGE.value: {"count": 0, "amount": 0.0},
            TransactionType.SUBSCRIPTION_PAYMENT.value: {"count": 0, "amount": 0.0},
            TransactionType.REFUND.value: {"count": 0, "amount": 0.0},
            TransactionType.BONUS_CREDIT.value: {"count": 0, "amount": 0.0}
        }

        if transactions["metadatas"][0]:
            for metadata in transactions["metadatas"][0]:
                created_at = datetime.fromisoformat(metadata["created_at"])
                if start_date <= created_at <= end_date:
                    tx_type = metadata["transaction_type"]
                    amount = metadata["amount"]

                    transaction_summary[tx_type]["count"] += 1
                    transaction_summary[tx_type]["amount"] += amount

                    if tx_type == TransactionType.API_USAGE.value:
                        total_credits_used += abs(amount)
                    elif tx_type in [TransactionType.CREDIT_PURCHASE.value, TransactionType.BONUS_CREDIT.value]:
                        total_credits_purchased += amount
                    elif tx_type == TransactionType.SUBSCRIPTION_PAYMENT.value:
                        total_amount_paid += amount

        # Process usage records
        usage_by_day = {}
        total_api_calls = 0

        if usage_records["metadatas"][0]:
            for metadata in usage_records["metadatas"][0]:
                created_at = datetime.fromisoformat(metadata["created_at"])
                if start_date <= created_at <= end_date:
                    day_key = created_at.date().isoformat()
                    if day_key not in usage_by_day:
                        usage_by_day[day_key] = {"requests": 0, "credits_used": 0.0, "cost_usd": 0.0}

                    usage_by_day[day_key]["requests"] += 1
                    usage_by_day[day_key]["credits_used"] += metadata["credits_used"]
                    usage_by_day[day_key]["cost_usd"] += metadata["cost_usd"]
                    total_api_calls += 1

        # Get current balance
        current_balance = self.get_user_credits(user_id)

        return {
            "user_id": user_id,
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "current_balance": {
                "total_credits": current_balance.total_credits if current_balance else 0.0,
                "used_credits": current_balance.used_credits if current_balance else 0.0,
                "remaining_credits": current_balance.remaining_credits if current_balance else 0.0
            },
            "period_summary": {
                "total_credits_purchased": total_credits_purchased,
                "total_credits_used": total_credits_used,
                "total_amount_paid_usd": total_amount_paid,
                "total_api_calls": total_api_calls
            },
            "transaction_summary": transaction_summary,
            "daily_usage": usage_by_day,
            "generated_at": datetime.utcnow().isoformat()
        }

    def check_usage_limits(self, user_id: str, subscription_tier: SubscriptionTier) -> Dict[str, Any]:
        """Check if user is within subscription usage limits"""

        # Get current month usage
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        usage_summary = self.get_usage_summary(user_id, days=30)

        # Get subscription limits
        limits = AppConfig.get_subscription_config(subscription_tier)

        # Check API call limits
        monthly_limit = limits["api_calls_per_month"]
        current_usage = usage_summary["total_requests"]

        api_limit_exceeded = monthly_limit != -1 and current_usage >= monthly_limit

        # Check cost limits
        cost_limit = limits["cost_limit_usd"]
        current_cost = usage_summary["total_cost_usd"]

        cost_limit_exceeded = cost_limit != -1 and current_cost >= cost_limit

        return {
            "within_limits": not (api_limit_exceeded or cost_limit_exceeded),
            "api_calls": {
                "current": current_usage,
                "limit": monthly_limit,
                "exceeded": api_limit_exceeded,
                "percentage": (current_usage / monthly_limit * 100) if monthly_limit != -1 else 0
            },
            "cost": {
                "current_usd": current_cost,
                "limit_usd": cost_limit,
                "exceeded": cost_limit_exceeded,
                "percentage": (current_cost / cost_limit * 100) if cost_limit != -1 else 0
            }
        }

    def record_transaction(
        self,
        user_id: str,
        transaction_type: TransactionType,
        amount: float,
        currency: str,
        description: str,
        metadata: Dict[str, Any] = None
    ) -> Transaction:
        """Record a financial transaction"""

        transaction_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        transaction = Transaction(
            transaction_id=transaction_id,
            user_id=user_id,
            transaction_type=transaction_type,
            amount=amount,
            currency=currency,
            description=description,
            created_at=created_at,
            metadata=metadata or {}
        )

        # Store transaction
        self.transactions_collection.add(
            documents=[f"Transaction: {description}"],
            metadatas=[{
                "transaction_id": transaction_id,
                "user_id": user_id,
                "transaction_type": transaction_type.value,
                "amount": amount,
                "currency": currency,
                "description": description,
                "created_at": created_at.isoformat(),
                "metadata": metadata or {}
            }],
            ids=[transaction_id]
        )

        return transaction

    def _calculate_cost(self, model: str, usage_type: UsageType, quantity: int) -> Tuple[float, float]:
        """Calculate cost in USD and credits for usage"""

        if model not in self.model_pricing:
            # Default pricing for unknown models
            cost_usd = 0.001 * quantity
            credits_used = cost_usd * 10  # 1 USD = 10 credits
            return cost_usd, credits_used

        pricing = self.model_pricing[model]

        if usage_type == UsageType.TOKEN_INPUT:
            cost_usd = (quantity / 1000) * pricing.get("input", 0.001)
        elif usage_type == UsageType.TOKEN_OUTPUT:
            cost_usd = (quantity / 1000) * pricing.get("output", 0.002)
        elif usage_type == UsageType.IMAGE_GENERATION:
            cost_usd = quantity * pricing.get("request", 0.04)
        elif usage_type == UsageType.AUDIO_TRANSCRIPTION:
            cost_usd = quantity * pricing.get("minute", 0.006)
        else:
            # Default API call pricing
            cost_usd = quantity * 0.001

        # Convert USD to credits (1 USD = 10 credits)
        credits_used = cost_usd * 10

        return round(cost_usd, 6), round(credits_used, 4)

    def _metadata_to_credit_balance(self, metadata: Dict[str, Any]) -> CreditBalance:
        """Convert ChromaDB metadata to CreditBalance object"""
        return CreditBalance(
            user_id=metadata["user_id"],
            total_credits=metadata["total_credits"],
            used_credits=metadata["used_credits"],
            remaining_credits=metadata["remaining_credits"],
            last_updated=datetime.fromisoformat(metadata["last_updated"]),
            subscription_tier=SubscriptionTier(metadata["subscription_tier"])
        )

# Initialize global billing manager instance
billing_manager = BillingManager()
