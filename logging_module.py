"""
Logging Module
Handles comprehensive logging of API calls, system events, errors, and monitoring
"""
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import chromadb
from chromadb.config import Settings
import structlog
from config import AppConfig

class LogLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventType(Enum):
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    USER_AUTH = "user_auth"
    SUBSCRIPTION_CHANGE = "subscription_change"
    PAYMENT = "payment"
    ADMIN_ACTION = "admin_action"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT = "rate_limit"
    SECURITY = "security"

@dataclass
class LogEntry:
    log_id: str
    timestamp: datetime
    level: LogLevel
    event_type: EventType
    user_id: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]
    message: str
    details: Dict[str, Any]
    source: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

@dataclass
class APICallLog:
    call_id: str
    user_id: str
    virtual_key: str
    provider: str
    model: str
    endpoint: str
    request_tokens: int
    response_tokens: int
    latency_ms: float
    status_code: int
    cost_credits: float
    cost_usd: float
    timestamp: datetime
    request_metadata: Dict[str, Any]
    response_metadata: Dict[str, Any]
    error_message: Optional[str] = None

class LoggingManager:
    """Manages comprehensive logging and monitoring"""

    def __init__(self):
        """Initialize ChromaDB client and structured logger"""
        self.client = chromadb.PersistentClient(
            path=AppConfig.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(anonymized_telemetry=False)
        )
        self._setup_collections()
        self._setup_structured_logger()

    def _setup_collections(self):
        """Setup ChromaDB collections for logging data"""
        try:
            self.logs_collection = self.client.get_collection("system_logs")
        except:
            self.logs_collection = self.client.create_collection(
                name="system_logs",
                metadata={"description": "System event and error logs"}
            )

        try:
            self.api_calls_collection = self.client.get_collection("api_call_logs")
        except:
            self.api_calls_collection = self.client.create_collection(
                name="api_call_logs",
                metadata={"description": "API call tracking and monitoring"}
            )

        try:
            self.security_logs_collection = self.client.get_collection("security_logs")
        except:
            self.security_logs_collection = self.client.create_collection(
                name="security_logs",
                metadata={"description": "Security events and alerts"}
            )

    def _setup_structured_logger(self):
        """Setup structured logging with structlog"""
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(int(AppConfig.LOG_LEVEL)),
            cache_logger_on_first_use=True,
        )
        self.logger = structlog.get_logger()

    def log_event(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        details: Dict[str, Any] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        source: str = "system",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> LogEntry:
        """Log a system event"""

        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        log_entry = LogEntry(
            log_id=log_id,
            timestamp=timestamp,
            level=level,
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            message=message,
            details=details or {},
            source=source,
            ip_address=ip_address,
            user_agent=user_agent
        )

        # Store in ChromaDB
        collection = self.security_logs_collection if event_type == EventType.SECURITY else self.logs_collection

        collection.add(
            documents=[message],
            metadatas=[{
                "log_id": log_id,
                "timestamp": timestamp.isoformat(),
                "level": level.value,
                "event_type": event_type.value,
                "user_id": user_id,
                "session_id": session_id,
                "request_id": request_id,
                "message": message,
                "details": json.dumps(details or {}),
                "source": source,
                "ip_address": ip_address,
                "user_agent": user_agent
            }],
            ids=[log_id]
        )

        # Also log with structured logger
        self.logger.log(
            level.value,
            message,
            event_type=event_type.value,
            user_id=user_id,
            request_id=request_id,
            details=details
        )

        return log_entry

    def log_api_call(
        self,
        user_id: str,
        virtual_key: str,
        provider: str,
        model: str,
        endpoint: str,
        request_tokens: int,
        response_tokens: int,
        latency_ms: float,
        status_code: int,
        cost_credits: float,
        cost_usd: float,
        request_metadata: Dict[str, Any] = None,
        response_metadata: Dict[str, Any] = None,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> APICallLog:
        """Log an API call with detailed metrics"""

        call_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        api_call_log = APICallLog(
            call_id=call_id,
            user_id=user_id,
            virtual_key=virtual_key,
            provider=provider,
            model=model,
            endpoint=endpoint,
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
            status_code=status_code,
            cost_credits=cost_credits,
            cost_usd=cost_usd,
            timestamp=timestamp,
            request_metadata=request_metadata or {},
            response_metadata=response_metadata or {},
            error_message=error_message,
        )

        # Store in ChromaDB
        self.api_calls_collection.add(
            documents=[f"API call to {provider} {model} - {status_code}"],
            metadatas=[{
                "call_id": call_id,
                "user_id": user_id,
                "virtual_key": virtual_key,
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "request_tokens": request_tokens,
                "response_tokens": response_tokens,
                "latency_ms": latency_ms,
                "status_code": status_code,
                "cost_credits": cost_credits,
                "cost_usd": cost_usd,
                "timestamp": timestamp.isoformat(),
                "request_metadata": json.dumps(request_metadata or {}),
                "response_metadata": json.dumps(response_metadata or {}),
                "error_message": error_message,
                "request_id": request_id
            }],
            ids=[call_id]
        )

        # Log event
        self.log_event(
            level=LogLevel.INFO if status_code == 200 else LogLevel.ERROR,
            event_type=EventType.API_REQUEST,
            message=f"API call to {provider} {model}",
            details={
                "call_id": call_id,
                "provider": provider,
                "model": model,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "cost_credits": cost_credits
            },
            user_id=user_id,
            request_id=request_id,
            source="api_gateway"
        )

        return api_call_log

    def log_security_event(
        self,
        event_description: str,
        severity: LogLevel,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Dict[str, Any] = None
    ):
        """Log security-related events"""

        self.log_event(
            level=severity,
            event_type=EventType.SECURITY,
            message=event_description,
            details=details or {},
            user_id=user_id,
            ip_address=ip_address,
            source="security"
        )

    def log_rate_limit_event(
        self,
        user_id: str,
        endpoint: str,
        current_requests: int,
        limit: int,
        ip_address: Optional[str] = None
    ):
        """Log rate limiting events"""

        self.log_event(
            level=LogLevel.WARNING,
            event_type=EventType.RATE_LIMIT,
            message=f"Rate limit exceeded for user {user_id}",
            details={
                "endpoint": endpoint,
                "current_requests": current_requests,
                "limit": limit,
                "exceeded_by": current_requests - limit
            },
            user_id=user_id,
            ip_address=ip_address,
            source="rate_limiter"
        )

    def get_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Retrieve logs with filtering options"""

        where_clause = {}

        if level:
            where_clause["level"] = level.value
        if event_type:
            where_clause["event_type"] = event_type.value
        if user_id:
            where_clause["user_id"] = user_id

        collection = self.security_logs_collection if event_type == EventType.SECURITY else self.logs_collection

        logs = collection.query(
            query_texts=[""],
            n_results=limit,
            where=where_clause if where_clause else None
        )

        result = []
        if logs["metadatas"][0]:
            for metadata in logs["metadatas"][0]:
                # Filter by date if specified
                log_timestamp = datetime.fromisoformat(metadata["timestamp"])
                if start_date and log_timestamp < start_date:
                    continue
                if end_date and log_timestamp > end_date:
                    continue

                result.append(self._metadata_to_log_entry(metadata))

        return sorted(result, key=lambda x: x.timestamp, reverse=True)

    def get_api_call_logs(
        self,
        user_id: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[APICallLog]:
        """Retrieve API call logs with filtering"""

        where_clause = {}

        if user_id:
            where_clause["user_id"] = user_id
        if provider:
            where_clause["provider"] = provider
        if model:
            where_clause["model"] = model

        logs = self.api_calls_collection.query(
            query_texts=[""],
            n_results=limit,
            where=where_clause if where_clause else None
        )

        result = []
        if logs["metadatas"][0]:
            for metadata in logs["metadatas"][0]:
                # Filter by date if specified
                log_timestamp = datetime.fromisoformat(metadata["timestamp"])
                if start_date and log_timestamp < start_date:
                    continue
                if end_date and log_timestamp > end_date:
                    continue

                result.append(self._metadata_to_api_call_log(metadata))

        return sorted(result, key=lambda x: x.timestamp, reverse=True)

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""

        start_time = datetime.utcnow() - timedelta(hours=hours)

        error_logs = self.get_logs(
            start_date=start_time,
            level=LogLevel.ERROR,
            limit=1000
        )

        # Group errors by type and source
        error_counts = {}
        error_by_source = {}
        error_by_user = {}

        for log in error_logs:
            # Count by event type
            event_type = log.event_type.value
            if event_type not in error_counts:
                error_counts[event_type] = 0
            error_counts[event_type] += 1

            # Count by source
            source = log.source
            if source not in error_by_source:
                error_by_source[source] = 0
            error_by_source[source] += 1

            # Count by user
            if log.user_id:
                if log.user_id not in error_by_user:
                    error_by_user[log.user_id] = 0
                error_by_user[log.user_id] += 1

        return {
            "period_hours": hours,
            "total_errors": len(error_logs),
            "error_by_type": error_counts,
            "error_by_source": error_by_source,
            "error_by_user": error_by_user,
            "recent_errors": [asdict(log) for log in error_logs[:10]]
        }

    def get_api_performance_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get API performance metrics for the last N hours"""

        start_time = datetime.utcnow() - timedelta(hours=hours)

        api_logs = self.get_api_call_logs(
            start_date=start_time,
            limit=10000
        )

        if not api_logs:
            return {
                "period_hours": hours,
                "total_requests": 0,
                "average_latency_ms": 0,
                "success_rate": 0,
                "by_provider": {},
                "by_model": {}
            }

        total_requests = len(api_logs)
        total_latency = sum(log.latency_ms for log in api_logs)
        successful_requests = len([log for log in api_logs if log.status_code == 200])

        # Group by provider
        by_provider = {}
        for log in api_logs:
            provider = log.provider
            if provider not in by_provider:
                by_provider[provider] = {
                    "requests": 0,
                    "total_latency": 0,
                    "successful": 0,
                    "total_cost_credits": 0
                }

            by_provider[provider]["requests"] += 1
            by_provider[provider]["total_latency"] += log.latency_ms
            by_provider[provider]["total_cost_credits"] += log.cost_credits
            if log.status_code == 200:
                by_provider[provider]["successful"] += 1

        # Calculate averages for providers
        for provider_data in by_provider.values():
            if provider_data["requests"] > 0:
                provider_data["average_latency_ms"] = provider_data["total_latency"] / provider_data["requests"]
                provider_data["success_rate"] = provider_data["successful"] / provider_data["requests"]

        # Group by model
        by_model = {}
        for log in api_logs:
            model = log.model
            if model not in by_model:
                by_model[model] = {
                    "requests": 0,
                    "total_latency": 0,
                    "successful": 0,
                    "total_cost_credits": 0
                }

            by_model[model]["requests"] += 1
            by_model[model]["total_latency"] += log.latency_ms
            by_model[model]["total_cost_credits"] += log.cost_credits
            if log.status_code == 200:
                by_model[model]["successful"] += 1

        # Calculate averages for models
        for model_data in by_model.values():
            if model_data["requests"] > 0:
                model_data["average_latency_ms"] = model_data["total_latency"] / model_data["requests"]
                model_data["success_rate"] = model_data["successful"] / model_data["requests"]

        return {
            "period_hours": hours,
            "total_requests": total_requests,
            "average_latency_ms": round(total_latency / total_requests, 2) if total_requests > 0 else 0,
            "success_rate": round(successful_requests / total_requests, 4) if total_requests > 0 else 0,
            "by_provider": by_provider,
            "by_model": by_model
        }

    def export_logs(
        self,
        format_type: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[EventType] = None
    ) -> str:
        """Export logs in specified format"""

        logs = self.get_logs(
            start_date=start_date,
            end_date=end_date,
            event_type=event_type,
            limit=10000
        )

        if format_type == "json":
            return json.dumps([asdict(log) for log in logs], default=str, indent=2)
        elif format_type == "csv":
            import csv
            import io

            output = io.StringIO()
            writer = csv.writer(output)

            # Write header
            if logs:
                writer.writerow(asdict(logs[0]).keys())

                # Write data
                for log in logs:
                    writer.writerow(asdict(log).values())

            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _metadata_to_log_entry(self, metadata: Dict[str, Any]) -> LogEntry:
        """Convert ChromaDB metadata to LogEntry object"""
        return LogEntry(
            log_id=metadata["log_id"],
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            level=LogLevel(metadata["level"]),
            event_type=EventType(metadata["event_type"]),
            user_id=metadata.get("user_id"),
            session_id=metadata.get("session_id"),
            request_id=metadata.get("request_id"),
            message=metadata["message"],
            details=json.loads(metadata["details"]) if metadata.get("details") else {},
            source=metadata["source"],
            ip_address=metadata.get("ip_address"),
            user_agent=metadata.get("user_agent")
        )

    def _metadata_to_api_call_log(self, metadata: Dict[str, Any]) -> APICallLog:
        """Convert ChromaDB metadata to APICallLog object"""
        return APICallLog(
            call_id=metadata["call_id"],
            user_id=metadata["user_id"],
            virtual_key=metadata["virtual_key"],
            provider=metadata["provider"],
            model=metadata["model"],
            endpoint=metadata["endpoint"],
            request_tokens=metadata["request_tokens"],
            response_tokens=metadata["response_tokens"],
            latency_ms=metadata["latency_ms"],
            status_code=metadata["status_code"],
            cost_credits=metadata["cost_credits"],
            cost_usd=metadata["cost_usd"],
            timestamp=datetime.fromisoformat(metadata["timestamp"]),
            request_metadata=json.loads(metadata["request_metadata"]) if metadata.get("request_metadata") else {},
            response_metadata=json.loads(metadata["response_metadata"]) if metadata.get("response_metadata") else {},
            error_message=metadata.get("error_message")
        )

# Initialize global logging manager instance
logging_manager = LoggingManager()

# Convenience functions for common logging operations
def log_info(message: str, **kwargs):
    """Log info level event"""
    return logging_manager.log_event(LogLevel.INFO, EventType.API_REQUEST, message, kwargs)

def log_error(message: str, **kwargs):
    """Log error level event"""
    return logging_manager.log_event(LogLevel.ERROR, EventType.SYSTEM_ERROR, message, kwargs)

def log_security(message: str, **kwargs):
    """Log security event"""
    return logging_manager.log_security_event(message, LogLevel.WARNING, **kwargs)
