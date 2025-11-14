# apps/api/core/modern_database.py
"""
MODERNIZED Database Module - SQLAlchemy 2.0 Standards (November 2025)

Key changes from 1.x:
- DeclarativeBase instead of declarative_base()
- mapped_column() with type annotations
- Mapped[] type hints for all columns
- Modern async session patterns
- Proper relationship() usage
"""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager
import logging

from sqlalchemy import text, MetaData, event
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool, QueuePool

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy 2.0 metadata with naming convention
metadata = MetaData(
    naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    }
)


# Modern SQLAlchemy 2.0 Base class
class Base(DeclarativeBase):
    """
    Modern SQLAlchemy 2.0 declarative base

    Changes from 1.x:
    - Inherit from DeclarativeBase (not use declarative_base())
    - Use Mapped[] type annotations
    - Use mapped_column() instead of Column()
    """

    metadata = metadata

    # Disable implicit coercion (SQLAlchemy 2.0 best practice)
    __allow_unmapped__ = False


# Global engine instance
engine: Optional[AsyncEngine] = None


def create_engine_instance() -> AsyncEngine:
    """
    Create modern async engine with SQLAlchemy 2.0 best practices

    Key configuration:
    - pool_pre_ping: Verify connections before use
    - pool_recycle: Recycle connections after 3600s
    - echo: Log SQL in debug mode
    - future: Enable 2.0 style (already default in 2.0+)
    """
    global engine

    if engine is None:
        # Choose pool class based on environment
        pool_class = NullPool if settings.is_production else QueuePool

        engine = create_async_engine(
            settings.postgres_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=10 if not settings.is_production else 20,
            max_overflow=20 if not settings.is_production else 40,
            pool_recycle=3600,  # Recycle connections after 1 hour
            poolclass=pool_class,
            # SQLAlchemy 2.0 uses future=True by default, but explicit is clear
            connect_args={
                "server_settings": {"application_name": "voice-doc-intelligence"}
            },
        )

        # Set up engine event listeners for connection tracking
        @event.listens_for(engine.sync_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            logger.debug("Database connection established")

        @event.listens_for(engine.sync_engine, "close")
        def receive_close(dbapi_conn, connection_record):
            logger.debug("Database connection closed")

        logger.info("✅ Async database engine created")

    return engine


# Modern async session factory using async_sessionmaker
def create_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Create async session factory with modern SQLAlchemy 2.0 patterns

    Returns async_sessionmaker that creates AsyncSession instances
    """
    engine_instance = create_engine_instance()

    return async_sessionmaker(
        bind=engine_instance,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual control over flushes
        autocommit=False,  # Always use explicit commits
    )


# Session factory instance
AsyncSessionFactory = create_session_factory()


# Dependency for FastAPI routes
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions

    Modern async pattern with proper error handling:
    1. Create session from factory
    2. Yield for use in route
    3. Commit on success
    4. Rollback on error
    5. Always close
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}", exc_info=True)
            raise
        finally:
            await session.close()


# Context manager for manual usage (non-FastAPI code)
@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside of FastAPI

    Usage:
        async with get_db_context() as session:
            result = await session.execute(select(User))
            ...
    """
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Database initialization
async def init_db() -> None:
    """
    Initialize database schema

    Modern SQLAlchemy 2.0 pattern:
    - Use engine.begin() for automatic transaction
    - Use run_sync() to run sync metadata operations
    """
    try:
        engine_instance = create_engine_instance()

        async with engine_instance.begin() as conn:
            # Create schema if it doesn't exist
            await conn.execute(text("CREATE SCHEMA IF NOT EXISTS doc_intel"))

            # Create all tables using metadata
            await conn.run_sync(Base.metadata.create_all)

        logger.info("✅ Database tables initialized successfully")

    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}", exc_info=True)
        raise


# Database cleanup
async def close_db() -> None:
    """
    Gracefully close database connections

    Call this during application shutdown
    """
    global engine

    if engine is not None:
        await engine.dispose()
        engine = None
        logger.info("✅ Database connections closed")


# Health check
async def check_db_health() -> bool:
    """
    Check database health by executing a simple query

    Returns True if healthy, False otherwise
    """
    try:
        async with get_db_context() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
