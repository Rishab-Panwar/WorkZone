from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from sqlalchemy import text, MetaData, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.schema import CreateSchema
from .config import Config
from typing import AsyncGenerator, Generator
from contextlib import contextmanager
from pathlib import Path

class PublicBase(DeclarativeBase):
    metadata = MetaData(schema="public")

class TenantBase(DeclarativeBase):
    pass

from src.models.tenant import *

async_engine = create_async_engine(
    url=Config.ASYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
    connect_args={"prepared_statement_cache_size": 0},
)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine)

async def init_db() -> None:
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS public"))
        
        await conn.run_sync(PublicBase.metadata.create_all)

async def get_public_db() -> AsyncGenerator[AsyncSession, None]:
    async for sesion in get_schema("public"):
        yield sesion 

async def get_schema(tenant_id: str) -> AsyncGenerator[AsyncSession, None]:
    if not tenant_id.isidentifier():
        raise ValueError("Invalid tenant ID")
    
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(text(f"SET search_path TO {tenant_id}"))
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


sync_engine = create_engine(
    url=Config.SYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,
)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

def _tenant_alembic_head() -> str | None:
    """Latest revision of the tenant Alembic history (or None if unavailable)."""
    from alembic.config import Config as AlembicConfig
    from alembic.script import ScriptDirectory

    ini_path = Path(__file__).resolve().parents[2] / "alembic_tenant.ini"
    cfg = AlembicConfig(str(ini_path))
    cfg.set_main_option("script_location", str(ini_path.parent / "alembic_tenant"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def _stamp_tenant_alembic_version(tenant_id: str) -> None:
    """Record an app-created tenant schema as already at the latest revision, so
    the migrate step treats it as up to date instead of re-running the initial
    migration (which fails with 'type role already exists'). Best-effort: a
    failure here must never break onboarding, so it runs in its own transaction."""
    try:
        head = _tenant_alembic_head()
        if not head:
            return
        with sync_engine.begin() as conn:
            conn.execute(text(f"SET search_path TO {tenant_id}"))
            conn.execute(text(
                "CREATE TABLE IF NOT EXISTS alembic_version ("
                "version_num VARCHAR(32) NOT NULL, "
                "CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num))"
            ))
            conn.execute(text("DELETE FROM alembic_version"))
            conn.execute(text("INSERT INTO alembic_version (version_num) VALUES (:rev)"), {"rev": head})
    except Exception as e:
        from src.core.logger import logger
        logger.warning(f"Could not stamp alembic_version for tenant {tenant_id}: {e}")


def create_tenant_schema(tenant_id: str):
    with sync_engine.begin() as conn:
        conn.execute(CreateSchema(tenant_id, if_not_exists=True))
        conn.execute(text(f"SET search_path TO {tenant_id}"))
        TenantBase.metadata.create_all(bind=conn)
    # Separate transaction: stamping must not be able to roll back the schema.
    _stamp_tenant_alembic_version(tenant_id)


@contextmanager
def get_public_db_sync() -> Generator[Session, None, None]:
    with get_tenant_db_sync("public") as session:
        yield session

@contextmanager
def get_tenant_db_sync(tenant_id: str) -> Generator[Session, None, None]:
    if not tenant_id.isidentifier():
        raise ValueError("Invalid tenant ID")
    
    session = SyncSessionLocal()
    try:
        session.execute(text(f"SET search_path TO {tenant_id}"))
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
