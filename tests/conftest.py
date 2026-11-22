import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from catalist.store import Base

# Separate database: tests wipe their tables, so they must never touch the real one.
TEST_URL = make_url(
    os.environ.get(
        "TEST_DATABASE_URL",
        "postgresql+psycopg://catalist:catalist@localhost:5432/catalist_test",
    )
)


@pytest.fixture
def engine():
    admin = create_engine(
        TEST_URL.set(database="postgres"), isolation_level="AUTOCOMMIT"
    )
    with admin.connect() as conn:
        exists = conn.scalar(
            text("SELECT 1 FROM pg_database WHERE datname = :name"),
            {"name": TEST_URL.database},
        )
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{TEST_URL.database}"'))
    admin.dispose()

    engine = create_engine(TEST_URL)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()
