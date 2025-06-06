"""
User Management Module
Handles user registration, login, authentication, and virtual key management
"""
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status
from config import AppConfig, SubscriptionTier

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthProvider(Enum):
    EMAIL = "email"
    GOOGLE = "google"
    APPLE = "apple"

@dataclass
class User:
    user_id: str
    email: str
    full_name: str
    auth_provider: AuthProvider
    subscription_tier: SubscriptionTier
    virtual_key: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    password_hash: Optional[str] = None
    provider_id: Optional[str] = None  # For OAuth providers

class UserManager:
    """Manages user registration, authentication, and virtual keys"""

    def __init__(self):
        """Initialize ChromaDB client and create collections"""
        self.client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self._setup_collections()

    def _setup_collections(self):
        """Setup ChromaDB collections for users and sessions"""
        try:
            # Users collection
            self.users_collection = self.client.get_collection("users")
        except:
            self.users_collection = self.client.create_collection(
                name="users",
                metadata={"description": "User registration and profile data"}
            )

        try:
            # Sessions collection
            self.sessions_collection = self.client.get_collection("user_sessions")
        except:
            self.sessions_collection = self.client.create_collection(
                name="user_sessions",
                metadata={"description": "User session and token data"}
            )

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)

    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)

    def _generate_virtual_key(self) -> str:
        """Generate a unique virtual key for the user"""
        return f"vk_{secrets.token_urlsafe(32)}"

    def _generate_user_id(self) -> str:
        """Generate a unique user ID"""
        return str(uuid.uuid4())

    def register_user_email(
        self, 
        email: str, 
        password: str, 
        full_name: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    ) -> User:
        """Register a new user with email and password"""

        # Check if user already exists
        existing_users = self.users_collection.query(
            query_texts=[email],
            n_results=1,
            where={"email": email}
        )

        if existing_users["ids"][0]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )

        # Create new user
        user_id = self._generate_user_id()
        virtual_key = self._generate_virtual_key()
        password_hash = self._hash_password(password)
        created_at = datetime.utcnow()

        user = User(
            user_id=user_id,
            email=email,
            full_name=full_name,
            auth_provider=AuthProvider.EMAIL,
            subscription_tier=subscription_tier,
            virtual_key=virtual_key,
            created_at=created_at,
            password_hash=password_hash
        )

        # Store in ChromaDB
        self.users_collection.add(
            documents=[f"User: {full_name}, Email: {email}"],
            metadatas=[{
                "user_id": user_id,
                "email": email,
                "full_name": full_name,
                "auth_provider": auth_provider.value,
                "subscription_tier": subscription_tier.value,
                "virtual_key": virtual_key,
                "created_at": created_at.isoformat(),
                "is_active": True,
                "password_hash": password_hash
            }],
            ids=[user_id]
        )

        return user

    def register_user_oauth(
        self,
        email: str,
        full_name: str,
        auth_provider: AuthProvider,
        provider_id: str,
        subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    ) -> User:
        """Register a new user via OAuth (Google/Apple)"""

        # Check if user already exists
        existing_users = self.users_collection.query(
            query_texts=[email],
            n_results=1,
            where={"email": email}
        )

        if existing_users["ids"][0]:
            # Update existing user with OAuth info
            return self._update_user_oauth(existing_users["ids"][0][0], provider_id, auth_provider)

        # Create new user
        user_id = self._generate_user_id()
        virtual_key = self._generate_virtual_key()
        created_at = datetime.utcnow()

        user = User(
            user_id=user_id,
            email=email,
            full_name=full_name,
            auth_provider=auth_provider,
            subscription_tier=subscription_tier,
            virtual_key=virtual_key,
            created_at=created_at,
            provider_id=provider_id
        )

        # Store in ChromaDB
        self.users_collection.add(
            documents=[f"User: {full_name}, Email: {email}"],
            metadatas=[{
                "user_id": user_id,
                "email": email,
                "full_name": full_name,
                "auth_provider": auth_provider.value,
                "subscription_tier": subscription_tier.value,
                "virtual_key": virtual_key,
                "created_at": created_at.isoformat(),
                "is_active": True,
                "provider_id": provider_id
            }],
            ids=[user_id]
        )

        return user

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""

        # Find user by email
        users = self.users_collection.query(
            query_texts=[email],
            n_results=1,
            where={"email": email, "auth_provider": AuthProvider.EMAIL.value}
        )

        if not users["ids"][0]:
            return None

        user_metadata = users["metadatas"][0][0]

        # Verify password
        if not self._verify_password(password, user_metadata["password_hash"]):
            return None

        # Update last login
        self._update_last_login(user_metadata["user_id"])

        return self._metadata_to_user(user_metadata)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            user_data = self.users_collection.get(ids=[user_id])
            if not user_data["ids"]:
                return None

            return self._metadata_to_user(user_data["metadatas"][0])
        except:
            return None

    def get_user_by_virtual_key(self, virtual_key: str) -> Optional[User]:
        """Get user by virtual key"""
        users = self.users_collection.query(
            query_texts=[""],
            n_results=1,
            where={"virtual_key": virtual_key}
        )

        if not users["ids"][0]:
            return None

        return self._metadata_to_user(users["metadatas"][0][0])

    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user"""
        to_encode = {
            "sub": user.user_id,
            "email": user.email,
            "subscription_tier": user.subscription_tier.value,
            "exp": datetime.utcnow() + timedelta(minutes=AppConfig.ACCESS_TOKEN_EXPIRE_MINUTES)
        }

        encoded_jwt = jwt.encode(to_encode, AppConfig.SECRET_KEY, algorithm=AppConfig.ALGORITHM)

        # Store session in ChromaDB
        session_id = str(uuid.uuid4())
        self.sessions_collection.add(
            documents=[f"Session for user {user.email}"],
            metadatas=[{
                "session_id": session_id,
                "user_id": user.user_id,
                "token": encoded_jwt,
                "created_at": datetime.utcnow().isoformat(),
                "expires_at": (datetime.utcnow() + timedelta(minutes=AppConfig.ACCESS_TOKEN_EXPIRE_MINUTES)).isoformat()
            }],
            ids=[session_id]
        )

        return encoded_jwt

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, AppConfig.SECRET_KEY, algorithms=[AppConfig.ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
            return payload
        except JWTError:
            return None

    def update_subscription(self, user_id: str, new_tier: SubscriptionTier) -> bool:
        """Update user subscription tier"""
        try:
            # Get current user data
            user_data = self.users_collection.get(ids=[user_id])
            if not user_data["ids"]:
                return False

            # Update metadata
            metadata = user_data["metadatas"][0]
            metadata["subscription_tier"] = new_tier.value

            # Update in ChromaDB
            self.users_collection.update(
                ids=[user_id],
                metadatas=[metadata]
            )

            return True
        except:
            return False

    def regenerate_virtual_key(self, user_id: str) -> Optional[str]:
        """Regenerate virtual key for user"""
        try:
            # Get current user data
            user_data = self.users_collection.get(ids=[user_id])
            if not user_data["ids"]:
                return None

            # Generate new virtual key
            new_virtual_key = self._generate_virtual_key()

            # Update metadata
            metadata = user_data["metadatas"][0]
            metadata["virtual_key"] = new_virtual_key

            # Update in ChromaDB
            self.users_collection.update(
                ids=[user_id],
                metadatas=[metadata]
            )

            return new_virtual_key
        except:
            return None

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account"""
        try:
            # Get current user data
            user_data = self.users_collection.get(ids=[user_id])
            if not user_data["ids"]:
                return False

            # Update metadata
            metadata = user_data["metadatas"][0]
            metadata["is_active"] = False

            # Update in ChromaDB
            self.users_collection.update(
                ids=[user_id],
                metadatas=[metadata]
            )

            return True
        except:
            return False

    def _update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        try:
            user_data = self.users_collection.get(ids=[user_id])
            if user_data["ids"]:
                metadata = user_data["metadatas"][0]
                metadata["last_login"] = datetime.utcnow().isoformat()

                self.users_collection.update(
                    ids=[user_id],
                    metadatas=[metadata]
                )
        except:
            pass

    def _update_user_oauth(self, user_id: str, provider_id: str, auth_provider: AuthProvider) -> User:
        """Update existing user with OAuth information"""
        user_data = self.users_collection.get(ids=[user_id])
        metadata = user_data["metadatas"][0]
        metadata["auth_provider"] = auth_provider.value
        metadata["provider_id"] = provider_id

        self.users_collection.update(
            ids=[user_id],
            metadatas=[metadata]
        )

        return self._metadata_to_user(metadata)

    def _metadata_to_user(self, metadata: Dict[str, Any]) -> User:
        """Convert ChromaDB metadata to User object"""
        return User(
            user_id=metadata["user_id"],
            email=metadata["email"],
            full_name=metadata["full_name"],
            auth_provider=AuthProvider(metadata["auth_provider"]),
            subscription_tier=SubscriptionTier(metadata["subscription_tier"]),
            virtual_key=metadata["virtual_key"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            last_login=datetime.fromisoformat(metadata["last_login"]) if metadata.get("last_login") else None,
            is_active=metadata.get("is_active", True),
            password_hash=metadata.get("password_hash"),
            provider_id=metadata.get("provider_id")
        )

# Initialize global user manager instance
user_manager = UserManager()
