"""
Admin Module
Handles administrative functions including AI provider API key management,
virtual key associations, and system configuration
"""
import uuid
import secrets
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException, status
from portkey_ai import Portkey
from config import AppConfig, SubscriptionTier

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure-openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"

class VirtualKeyStatus(Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

@dataclass
class ProviderAPIKey:
    key_id: str
    provider: AIProvider
    api_key: str
    description: str
    is_active: bool
    created_at: datetime
    created_by: str
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class VirtualKeyMapping:
    mapping_id: str
    user_virtual_key: str
    provider_key_id: str
    user_id: str
    provider: AIProvider
    status: VirtualKeyStatus
    created_at: datetime
    portkey_virtual_key: Optional[str] = None
    config_id: Optional[str] = None

class AdminManager:
    """Manages administrative functions for the API gateway"""

    def __init__(self):
        """Initialize ChromaDB client and Portkey client"""
        self.client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize Portkey client
        if AppConfig.PORTKEY_API_KEY:
            self.portkey_client = Portkey(api_key=AppConfig.PORTKEY_API_KEY)
        else:
            self.portkey_client = None

        self._setup_collections()

    def _setup_collections(self):
        """Setup ChromaDB collections for admin data"""
        try:
            self.provider_keys_collection = self.client.get_collection("provider_api_keys")
        except:
            self.provider_keys_collection = self.client.create_collection(
                name="provider_api_keys",
                metadata={"description": "AI provider API keys storage"}
            )

        try:
            self.virtual_mappings_collection = self.client.get_collection("virtual_key_mappings")
        except:
            self.virtual_mappings_collection = self.client.create_collection(
                name="virtual_key_mappings",
                metadata={"description": "Virtual key to provider key mappings"}
            )

        try:
            self.admin_logs_collection = self.client.get_collection("admin_logs")
        except:
            self.admin_logs_collection = self.client.create_collection(
                name="admin_logs",
                metadata={"description": "Administrative action logs"}
            )

    def add_provider_api_key(
        self,
        provider: AIProvider,
        api_key: str,
        description: str,
        admin_user_id: str
    ) -> ProviderAPIKey:
        """Add a new AI provider API key"""

        key_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        # Validate API key by testing with provider
        if not self._validate_api_key(provider, api_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid API key for provider {provider.value}"
            )

        provider_key = ProviderAPIKey(
            key_id=key_id,
            provider=provider,
            api_key=api_key,
            description=description,
            is_active=True,
            created_at=created_at,
            created_by=admin_user_id
        )

        # Store in ChromaDB (encrypt the API key in production)
        self.provider_keys_collection.add(
            documents=[f"API Key for {provider.value}: {description}"],
            metadatas=[{
                "key_id": key_id,
                "provider": provider.value,
                "api_key": self._encrypt_api_key(api_key),  # Encrypt in production
                "description": description,
                "is_active": True,
                "created_at": created_at.isoformat(),
                "created_by": admin_user_id,
                "usage_count": 0
            }],
            ids=[key_id]
        )

        # Log admin action
        self._log_admin_action(
            admin_user_id,
            "ADD_PROVIDER_KEY",
            f"Added API key for {provider.value}: {description}"
        )

        return provider_key

    def create_virtual_key_mapping(
        self,
        user_virtual_key: str,
        user_id: str,
        provider: AIProvider,
        admin_user_id: str,
        provider_key_id: Optional[str] = None
    ) -> VirtualKeyMapping:
        """Create virtual key mapping and Portkey virtual key"""

        # Get provider API key
        if not provider_key_id:
            provider_key_id = self._get_best_provider_key(provider)

        if not provider_key_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No available API key for provider {provider.value}"
            )

        provider_key_data = self.provider_keys_collection.get(ids=[provider_key_id])
        if not provider_key_data["ids"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Provider API key not found"
            )

        provider_metadata = provider_key_data["metadatas"][0]
        decrypted_api_key = self._decrypt_api_key(provider_metadata["api_key"])

        # Create Portkey virtual key
        portkey_virtual_key = None
        config_id = None

        if self.portkey_client:
            try:
                # Create virtual key in Portkey
                virtual_key_response = self.portkey_client.virtual_keys.create(
                    name=f"User_{user_id}_{provider.value}",
                    provider=provider.value,
                    key=decrypted_api_key
                )
                portkey_virtual_key = virtual_key_response.get("data", {}).get("slug")

                # Create basic config for the virtual key
                config_response = self.portkey_client.configs.create(
                    name=f"Config_{user_id}_{provider.value}",
                    config={
                        "virtual_key": portkey_virtual_key,
                        "provider": provider.value
                    }
                )
                config_id = config_response.get("data", {}).get("id")

            except Exception as e:
                print(f"Failed to create Portkey virtual key: {e}")

        # Create mapping
        mapping_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        mapping = VirtualKeyMapping(
            mapping_id=mapping_id,
            user_virtual_key=user_virtual_key,
            provider_key_id=provider_key_id,
            user_id=user_id,
            provider=provider,
            status=VirtualKeyStatus.ACTIVE,
            created_at=created_at,
            portkey_virtual_key=portkey_virtual_key,
            config_id=config_id
        )

        # Store mapping
        self.virtual_mappings_collection.add(
            documents=[f"Virtual key mapping for user {user_id} - {provider.value}"],
            metadatas=[{
                "mapping_id": mapping_id,
                "user_virtual_key": user_virtual_key,
                "provider_key_id": provider_key_id,
                "user_id": user_id,
                "provider": provider.value,
                "status": VirtualKeyStatus.ACTIVE.value,
                "created_at": created_at.isoformat(),
                "portkey_virtual_key": portkey_virtual_key,
                "config_id": config_id
            }],
            ids=[mapping_id]
        )

        # Update provider key usage
        self._increment_key_usage(provider_key_id)

        # Log admin action
        self._log_admin_action(
            admin_user_id,
            "CREATE_VIRTUAL_MAPPING",
            f"Created virtual key mapping for user {user_id} with {provider.value}"
        )

        return mapping

    def get_user_virtual_key_mappings(self, user_id: str) -> List[VirtualKeyMapping]:
        """Get all virtual key mappings for a user"""
        mappings = self.virtual_mappings_collection.query(
            query_texts=[""],
            n_results=100,
            where={"user_id": user_id}
        )

        return [self._metadata_to_mapping(metadata) for metadata in mappings["metadatas"][0]]

    def get_user_provider_mapping(self, user_id: str, provider: AIProvider) -> Optional[VirtualKeyMapping]:
        """Get virtual key mapping for specific user and provider"""
        mappings = self.virtual_mappings_collection.query(
            query_texts=[""],
            n_results=1,
            where={"user_id": user_id, "provider": provider.value, "status": VirtualKeyStatus.ACTIVE.value}
        )

        if not mappings["ids"][0]:
            return None

        return self._metadata_to_mapping(mappings["metadatas"][0][0])

    def deactivate_virtual_key_mapping(self, mapping_id: str, admin_user_id: str) -> bool:
        """Deactivate a virtual key mapping"""
        try:
            mapping_data = self.virtual_mappings_collection.get(ids=[mapping_id])
            if not mapping_data["ids"]:
                return False

            metadata = mapping_data["metadatas"][0]
            metadata["status"] = VirtualKeyStatus.INACTIVE.value

            self.virtual_mappings_collection.update(
                ids=[mapping_id],
                metadatas=[metadata]
            )

            # Log admin action
            self._log_admin_action(
                admin_user_id,
                "DEACTIVATE_VIRTUAL_MAPPING",
                f"Deactivated virtual key mapping {mapping_id}"
            )

            return True
        except:
            return False

    def list_provider_api_keys(self, provider: Optional[AIProvider] = None) -> List[ProviderAPIKey]:
        """List all provider API keys, optionally filtered by provider"""
        where_clause = {}
        if provider:
            where_clause["provider"] = provider.value

        keys = self.provider_keys_collection.query(
            query_texts=[""],
            n_results=100,
            where=where_clause
        )

        return [self._metadata_to_provider_key(metadata) for metadata in keys["metadatas"][0]]

    def update_provider_api_key(
        self,
        key_id: str,
        api_key: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        admin_user_id: str = None
    ) -> bool:
        """Update provider API key"""
        try:
            key_data = self.provider_keys_collection.get(ids=[key_id])
            if not key_data["ids"]:
                return False

            metadata = key_data["metadatas"][0]

            if api_key is not None:
                # Validate new API key
                provider = AIProvider(metadata["provider"])
                if not self._validate_api_key(provider, api_key):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid API key"
                    )
                metadata["api_key"] = self._encrypt_api_key(api_key)

            if description is not None:
                metadata["description"] = description

            if is_active is not None:
                metadata["is_active"] = is_active

            self.provider_keys_collection.update(
                ids=[key_id],
                metadatas=[metadata]
            )

            # Log admin action
            if admin_user_id:
                self._log_admin_action(
                    admin_user_id,
                    "UPDATE_PROVIDER_KEY",
                    f"Updated API key {key_id}"
                )

            return True
        except:
            return False

    def delete_provider_api_key(self, key_id: str, admin_user_id: str) -> bool:
        """Delete provider API key"""
        try:
            # Check if key is in use
            mappings = self.virtual_mappings_collection.query(
                query_texts=[""],
                n_results=1,
                where={"provider_key_id": key_id, "status": VirtualKeyStatus.ACTIVE.value}
            )

            if mappings["ids"][0]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot delete API key that is currently in use"
                )

            self.provider_keys_collection.delete(ids=[key_id])

            # Log admin action
            self._log_admin_action(
                admin_user_id,
                "DELETE_PROVIDER_KEY",
                f"Deleted API key {key_id}"
            )

            return True
        except HTTPException:
            raise
        except:
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics for admin dashboard"""
        # Get provider key stats
        provider_keys = self.list_provider_api_keys()
        provider_stats = {}

        for provider in AIProvider:
            provider_stats[provider.value] = {
                "total_keys": len([k for k in provider_keys if k.provider == provider]),
                "active_keys": len([k for k in provider_keys if k.provider == provider and k.is_active])
            }

        # Get virtual key mapping stats
        all_mappings = self.virtual_mappings_collection.query(
            query_texts=[""],
            n_results=1000
        )

        total_mappings = len(all_mappings["ids"][0]) if all_mappings["ids"][0] else 0
        active_mappings = 0

        if all_mappings["metadatas"][0]:
            active_mappings = len([
                m for m in all_mappings["metadatas"][0] 
                if m.get("status") == VirtualKeyStatus.ACTIVE.value
            ])

        return {
            "provider_keys": provider_stats,
            "virtual_mappings": {
                "total": total_mappings,
                "active": active_mappings
            },
            "last_updated": datetime.utcnow().isoformat()
        }

    def _validate_api_key(self, provider: AIProvider, api_key: str) -> bool:
        """Validate API key by testing with provider (simplified)"""
        # In production, implement actual validation for each provider
        # For now, just check if key is not empty and has reasonable format
        if not api_key or len(api_key) < 10:
            return False

        # Basic format validation
        if provider == AIProvider.OPENAI and not api_key.startswith("sk-"):
            return False
        elif provider == AIProvider.ANTHROPIC and not api_key.startswith("sk-ant-"):
            return False

        return True

    def _encrypt_api_key(self, api_key: str) -> str:
        """Encrypt API key (simplified - use proper encryption in production)"""
        # In production, use proper encryption like Fernet
        import base64
        return base64.b64encode(api_key.encode()).decode()

    def _decrypt_api_key(self, encrypted_key: str) -> str:
        """Decrypt API key (simplified - use proper decryption in production)"""
        # In production, use proper decryption
        import base64
        return base64.b64decode(encrypted_key.encode()).decode()

    def _get_best_provider_key(self, provider: AIProvider) -> Optional[str]:
        """Get the best available provider key for the given provider"""
        keys = self.provider_keys_collection.query(
            query_texts=[""],
            n_results=10,
            where={"provider": provider.value, "is_active": True}
        )

        if not keys["ids"][0]:
            return None

        # Return the key with lowest usage count
        best_key_id = keys["ids"][0][0]
        min_usage = float('inf')

        for i, metadata in enumerate(keys["metadatas"][0]):
            usage_count = metadata.get("usage_count", 0)
            if usage_count < min_usage:
                min_usage = usage_count
                best_key_id = keys["ids"][0][i]

        return best_key_id

    def _increment_key_usage(self, key_id: str):
        """Increment usage count for provider key"""
        try:
            key_data = self.provider_keys_collection.get(ids=[key_id])
            if key_data["ids"]:
                metadata = key_data["metadatas"][0]
                metadata["usage_count"] = metadata.get("usage_count", 0) + 1
                metadata["last_used"] = datetime.utcnow().isoformat()

                self.provider_keys_collection.update(
                    ids=[key_id],
                    metadatas=[metadata]
                )
        except:
            pass

    def _log_admin_action(self, admin_user_id: str, action: str, description: str):
        """Log administrative action"""
        log_id = str(uuid.uuid4())

        self.admin_logs_collection.add(
            documents=[f"Admin action: {action} - {description}"],
            metadatas=[{
                "log_id": log_id,
                "admin_user_id": admin_user_id,
                "action": action,
                "description": description,
                "timestamp": datetime.utcnow().isoformat()
            }],
            ids=[log_id]
        )

    def _metadata_to_provider_key(self, metadata: Dict[str, Any]) -> ProviderAPIKey:
        """Convert ChromaDB metadata to ProviderAPIKey object"""
        return ProviderAPIKey(
            key_id=metadata["key_id"],
            provider=AIProvider(metadata["provider"]),
            api_key=self._decrypt_api_key(metadata["api_key"]),
            description=metadata["description"],
            is_active=metadata["is_active"],
            created_at=datetime.fromisoformat(metadata["created_at"]),
            created_by=metadata["created_by"],
            last_used=datetime.fromisoformat(metadata["last_used"]) if metadata.get("last_used") else None,
            usage_count=metadata.get("usage_count", 0)
        )

    def _metadata_to_mapping(self, metadata: Dict[str, Any]) -> VirtualKeyMapping:
        """Convert ChromaDB metadata to VirtualKeyMapping object"""
        return VirtualKeyMapping(
            mapping_id=metadata["mapping_id"],
            user_virtual_key=metadata["user_virtual_key"],
            provider_key_id=metadata["provider_key_id"],
            user_id=metadata["user_id"],
            provider=AIProvider(metadata["provider"]),
            status=VirtualKeyStatus(metadata["status"]),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            portkey_virtual_key=metadata.get("portkey_virtual_key"),
            config_id=metadata.get("config_id")
        )

# Initialize global admin manager instance
admin_manager = AdminManager()
