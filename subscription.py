"""
Subscription Management Module
Handles subscription tiers, payment processing, and feature access control
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException, status
from config import AppConfig, SubscriptionTier

class PaymentStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    PAYPAL = "paypal"
    STRIPE = "stripe"
    MOCK = "mock"  # For testing

@dataclass
class Subscription:
    subscription_id: str
    user_id: str
    tier: SubscriptionTier
    status: str
    start_date: datetime
    end_date: datetime
    auto_renew: bool
    payment_method: Optional[PaymentMethod] = None
    last_payment_date: Optional[datetime] = None
    next_payment_date: Optional[datetime] = None
    created_at: Optional[datetime] = None

@dataclass
class Payment:
    payment_id: str
    user_id: str
    subscription_id: str
    amount: float
    currency: str
    payment_method: PaymentMethod
    status: PaymentStatus
    transaction_id: Optional[str]
    created_at: datetime
    processed_at: Optional[datetime] = None

class SubscriptionManager:
    """Manages user subscriptions and payments"""

    def __init__(self):
        """Initialize ChromaDB client and create collections"""
        self.client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self._setup_collections()

    def _setup_collections(self):
        """Setup ChromaDB collections for subscriptions and payments"""
        try:
            self.subscriptions_collection = self.client.get_collection("subscriptions")
        except:
            self.subscriptions_collection = self.client.create_collection(
                name="subscriptions",
                metadata={"description": "User subscription data"}
            )

        try:
            self.payments_collection = self.client.get_collection("payments")
        except:
            self.payments_collection = self.client.create_collection(
                name="payments",
                metadata={"description": "Payment transaction data"}
            )

    def create_subscription(
        self,
        user_id: str,
        tier: SubscriptionTier,
        payment_method: Optional[PaymentMethod] = None,
        auto_renew: bool = True
    ) -> Subscription:
        """Create a new subscription for user"""

        subscription_id = str(uuid.uuid4())
        start_date = datetime.utcnow()

        # Calculate end date based on tier
        if tier == SubscriptionTier.FREE:
            end_date = start_date + timedelta(days=365)  # Free tier lasts 1 year
            next_payment_date = None
        else:
            end_date = start_date + timedelta(days=30)  # Paid tiers are monthly
            next_payment_date = end_date if auto_renew else None

        subscription = Subscription(
            subscription_id=subscription_id,
            user_id=user_id,
            tier=tier,
            status="active",
            start_date=start_date,
            end_date=end_date,
            auto_renew=auto_renew,
            payment_method=payment_method,
            next_payment_date=next_payment_date,
            created_at=start_date
        )

        # Store in ChromaDB
        self.subscriptions_collection.add(
            documents=[f"Subscription for user {user_id} - {tier.value}"],
            metadatas=[{
                "subscription_id": subscription_id,
                "user_id": user_id,
                "tier": tier.value,
                "status": "active",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "auto_renew": auto_renew,
                "payment_method": payment_method.value if payment_method else None,
                "next_payment_date": next_payment_date.isoformat() if next_payment_date else None,
                "created_at": start_date.isoformat()
            }],
            ids=[subscription_id]
        )

        return subscription

    def get_user_subscription(self, user_id: str) -> Optional[Subscription]:
        """Get active subscription for user"""
        subscriptions = self.subscriptions_collection.query(
            query_texts=[""],
            n_results=1,
            where={"user_id": user_id, "status": "active"}
        )

        if not subscriptions["ids"][0]:
            # Create free subscription if none exists
            return self.create_subscription(user_id, SubscriptionTier.FREE)

        return self._metadata_to_subscription(subscriptions["metadatas"][0][0])

    def upgrade_subscription(
        self,
        user_id: str,
        new_tier: SubscriptionTier,
        payment_method: PaymentMethod
    ) -> Dict[str, Any]:
        """Upgrade user subscription to a higher tier"""

        if new_tier == SubscriptionTier.FREE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot upgrade to free tier"
            )

        # Get current subscription
        current_subscription = self.get_user_subscription(user_id)
        if not current_subscription:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No subscription found for user"
            )

        # Calculate payment amount
        amount = AppConfig.SUBSCRIPTION_PRICING[new_tier]

        # Process payment
        payment = self.process_payment(
            user_id=user_id,
            subscription_id=current_subscription.subscription_id,
            amount=amount,
            payment_method=payment_method
        )

        if payment.status == PaymentStatus.COMPLETED:
            # Update subscription
            self._update_subscription_tier(current_subscription.subscription_id, new_tier, payment_method)

            return {
                "success": True,
                "subscription_id": current_subscription.subscription_id,
                "payment_id": payment.payment_id,
                "new_tier": new_tier.value,
                "amount_paid": amount
            }
        else:
            return {
                "success": False,
                "error": "Payment failed",
                "payment_id": payment.payment_id
            }

    def downgrade_subscription(self, user_id: str, new_tier: SubscriptionTier) -> bool:
        """Downgrade user subscription to a lower tier"""
        current_subscription = self.get_user_subscription(user_id)
        if not current_subscription:
            return False

        # Update subscription
        return self._update_subscription_tier(current_subscription.subscription_id, new_tier)

    def cancel_subscription(self, user_id: str) -> bool:
        """Cancel user subscription"""
        current_subscription = self.get_user_subscription(user_id)
        if not current_subscription:
            return False

        try:
            # Update subscription status
            subscription_data = self.subscriptions_collection.get(ids=[current_subscription.subscription_id])
            metadata = subscription_data["metadatas"][0]
            metadata["status"] = "cancelled"
            metadata["auto_renew"] = False
            metadata["next_payment_date"] = None

            self.subscriptions_collection.update(
                ids=[current_subscription.subscription_id],
                metadatas=[metadata]
            )

            return True
        except:
            return False

    def process_payment(
        self,
        user_id: str,
        subscription_id: str,
        amount: float,
        payment_method: PaymentMethod,
        currency: str = "USD"
    ) -> Payment:
        """Process payment for subscription"""

        payment_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        # Mock payment processing (replace with real payment gateway)
        if payment_method == PaymentMethod.MOCK or True:  # For demonstration
            status = PaymentStatus.COMPLETED
            transaction_id = f"txn_{uuid.uuid4().hex[:12]}"
            processed_at = created_at
        else:
            # Here you would integrate with real payment processors
            # like Stripe, PayPal, etc.
            status = self._process_real_payment(amount, payment_method)
            transaction_id = f"real_txn_{uuid.uuid4().hex[:12]}"
            processed_at = datetime.utcnow() if status == PaymentStatus.COMPLETED else None

        payment = Payment(
            payment_id=payment_id,
            user_id=user_id,
            subscription_id=subscription_id,
            amount=amount,
            currency=currency,
            payment_method=payment_method,
            status=status,
            transaction_id=transaction_id,
            created_at=created_at,
            processed_at=processed_at
        )

        # Store payment in ChromaDB
        self.payments_collection.add(
            documents=[f"Payment {amount} {currency} for user {user_id}"],
            metadatas=[{
                "payment_id": payment_id,
                "user_id": user_id,
                "subscription_id": subscription_id,
                "amount": amount,
                "currency": currency,
                "payment_method": payment_method.value,
                "status": status.value,
                "transaction_id": transaction_id,
                "created_at": created_at.isoformat(),
                "processed_at": processed_at.isoformat() if processed_at else None
            }],
            ids=[payment_id]
        )

        return payment

    def get_payment_history(self, user_id: str) -> List[Payment]:
        """Get payment history for user"""
        payments = self.payments_collection.query(
            query_texts=[""],
            n_results=100,
            where={"user_id": user_id}
        )

        return [self._metadata_to_payment(metadata) for metadata in payments["metadatas"][0]]

    def check_subscription_limits(self, user_id: str, requested_feature: str) -> Dict[str, Any]:
        """Check if user's subscription allows access to requested feature"""
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return {"allowed": False, "reason": "No subscription found"}

        # Check if subscription is active
        if subscription.status != "active":
            return {"allowed": False, "reason": "Subscription not active"}

        # Check if subscription has expired
        if datetime.utcnow() > subscription.end_date:
            return {"allowed": False, "reason": "Subscription expired"}

        # Get subscription configuration
        config = AppConfig.get_subscription_config(subscription.tier)

        # Check feature access
        if requested_feature not in config["features"]:
            return {
                "allowed": False, 
                "reason": f"Feature '{requested_feature}' not available in {subscription.tier.value} tier",
                "current_tier": subscription.tier.value,
                "available_features": config["features"]
            }

        return {
            "allowed": True,
            "tier": subscription.tier.value,
            "features": config["features"],
            "limits": config
        }

    def get_available_models(self, user_id: str) -> List[str]:
        """Get list of AI models available to user based on subscription"""
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return AppConfig.get_subscription_config(SubscriptionTier.FREE)["models_access"]

        config = AppConfig.get_subscription_config(subscription.tier)
        return config["models_access"]

    def renew_subscription(self, subscription_id: str) -> bool:
        """Renew subscription (for auto-renewal)"""
        try:
            subscription_data = self.subscriptions_collection.get(ids=[subscription_id])
            if not subscription_data["ids"]:
                return False

            metadata = subscription_data["metadatas"][0]
            tier = SubscriptionTier(metadata["tier"])

            if tier != SubscriptionTier.FREE:
                # Process renewal payment
                amount = AppConfig.SUBSCRIPTION_PRICING[tier]
                payment = self.process_payment(
                    user_id=metadata["user_id"],
                    subscription_id=subscription_id,
                    amount=amount,
                    payment_method=PaymentMethod(metadata["payment_method"])
                )

                if payment.status != PaymentStatus.COMPLETED:
                    return False

            # Extend subscription
            current_end_date = datetime.fromisoformat(metadata["end_date"])
            new_end_date = current_end_date + timedelta(days=30)
            new_next_payment_date = new_end_date if metadata["auto_renew"] else None

            metadata["end_date"] = new_end_date.isoformat()
            metadata["next_payment_date"] = new_next_payment_date.isoformat() if new_next_payment_date else None
            metadata["last_payment_date"] = datetime.utcnow().isoformat()

            self.subscriptions_collection.update(
                ids=[subscription_id],
                metadatas=[metadata]
            )

            return True
        except:
            return False

    def _process_real_payment(self, amount: float, payment_method: PaymentMethod) -> PaymentStatus:
        """Process real payment through payment gateway (placeholder)"""
        # This is where you would integrate with real payment processors
        # For now, return completed for demonstration
        return PaymentStatus.COMPLETED

    def _update_subscription_tier(
        self, 
        subscription_id: str, 
        new_tier: SubscriptionTier,
        payment_method: Optional[PaymentMethod] = None
    ) -> bool:
        """Update subscription tier"""
        try:
            subscription_data = self.subscriptions_collection.get(ids=[subscription_id])
            if not subscription_data["ids"]:
                return False

            metadata = subscription_data["metadatas"][0]
            metadata["tier"] = new_tier.value

            if payment_method:
                metadata["payment_method"] = payment_method.value
                metadata["last_payment_date"] = datetime.utcnow().isoformat()

            # Update end date for paid tiers
            if new_tier != SubscriptionTier.FREE:
                new_end_date = datetime.utcnow() + timedelta(days=30)
                metadata["end_date"] = new_end_date.isoformat()
                if metadata.get("auto_renew"):
                    metadata["next_payment_date"] = new_end_date.isoformat()

            self.subscriptions_collection.update(
                ids=[subscription_id],
                metadatas=[metadata]
            )

            return True
        except:
            return False

    def _metadata_to_subscription(self, metadata: Dict[str, Any]) -> Subscription:
        """Convert ChromaDB metadata to Subscription object"""
        return Subscription(
            subscription_id=metadata["subscription_id"],
            user_id=metadata["user_id"],
            tier=SubscriptionTier(metadata["tier"]),
            status=metadata["status"],
            start_date=datetime.fromisoformat(metadata["start_date"]),
            end_date=datetime.fromisoformat(metadata["end_date"]),
            auto_renew=metadata["auto_renew"],
            payment_method=PaymentMethod(metadata["payment_method"]) if metadata.get("payment_method") else None,
            last_payment_date=datetime.fromisoformat(metadata["last_payment_date"]) if metadata.get("last_payment_date") else None,
            next_payment_date=datetime.fromisoformat(metadata["next_payment_date"]) if metadata.get("next_payment_date") else None,
            created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else None
        )

    def _metadata_to_payment(self, metadata: Dict[str, Any]) -> Payment:
        """Convert ChromaDB metadata to Payment object"""
        return Payment(
            payment_id=metadata["payment_id"],
            user_id=metadata["user_id"],
            subscription_id=metadata["subscription_id"],
            amount=metadata["amount"],
            currency=metadata["currency"],
            payment_method=PaymentMethod(metadata["payment_method"]),
            status=PaymentStatus(metadata["status"]),
            transaction_id=metadata["transaction_id"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            processed_at=datetime.fromisoformat(metadata["processed_at"]) if metadata.get("processed_at") else None
        )

# Initialize global subscription manager instance
subscription_manager = SubscriptionManager()
