"""
User and authentication models
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Enum as SQLEnum
from datetime import datetime
from enum import Enum
from app.core.database import Base


class UserRole(str, Enum):
    """User roles for RBAC"""
    ADMIN = "admin"
    TRAFFIC_OPERATOR = "traffic_operator"
    EMERGENCY_DISPATCHER = "emergency_dispatcher"
    ANALYST = "analyst"
    VIEWER = "viewer"


class User(Base):
    """User accounts for system access"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(SQLEnum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    phone_number = Column(String(20), nullable=True)
    last_login = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RefreshToken(Base):
    """Refresh tokens for JWT authentication"""
    __tablename__ = "refresh_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    token = Column(String(500), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
