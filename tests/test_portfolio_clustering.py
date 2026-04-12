"""Tests for PortfolioCluster, PortfolioStore, and fascination key builder."""

from pathlib import Path
import pytest
import pytest_asyncio

from opencas.autonomy.portfolio import PortfolioCluster, PortfolioStore, build_fascination_key


def test_fascination_key_from_content() -> None:
    content = "The quick brown fox jumps over the lazy dog"
    key = build_fascination_key(content)
    # stop words removed, sorted, deduped, joined with +
    assert "quick" in key
    assert "brown" in key
    assert "fox" in key
    assert "the" not in key
    assert "+" in key


def test_fascination_key_from_tags() -> None:
    tags = ["AI", " rust", "ai "]
    key = build_fascination_key("some content", tags=tags)
    assert key == "ai+rust"


def test_fascination_key_limits_tokens() -> None:
    content = " ".join([f"word{i}" for i in range(20)])
    key = build_fascination_key(content)
    assert len(key.split("+")) <= 8


@pytest_asyncio.fixture
async def portfolio_store(tmp_path: Path) -> PortfolioStore:
    store = PortfolioStore(tmp_path / "portfolio.db")
    await store.connect()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_portfolio_save_and_get_by_key(portfolio_store: PortfolioStore) -> None:
    cluster = PortfolioCluster(fascination_key="ai+rust")
    await portfolio_store.save(cluster)

    fetched = await portfolio_store.get_by_key("ai+rust")
    assert fetched is not None
    assert fetched.fascination_key == "ai+rust"


@pytest.mark.asyncio
async def test_increment_counts(portfolio_store: PortfolioStore) -> None:
    cluster = PortfolioCluster(fascination_key="python+testing")
    await portfolio_store.save(cluster)

    ok = await portfolio_store.increment_counts("python+testing", sparks=2, initiatives=1, artifacts=1)
    assert ok is True

    fetched = await portfolio_store.get_by_key("python+testing")
    assert fetched.spark_count == 2
    assert fetched.initiative_count == 1
    assert fetched.artifact_count == 1


@pytest.mark.asyncio
async def test_list_all(portfolio_store: PortfolioStore) -> None:
    c1 = PortfolioCluster(fascination_key="a+b")
    c2 = PortfolioCluster(fascination_key="c+d")
    await portfolio_store.save(c1)
    await portfolio_store.save(c2)

    all_clusters = await portfolio_store.list_all()
    keys = {c.fascination_key for c in all_clusters}
    assert keys == {"a+b", "c+d"}


@pytest.mark.asyncio
async def test_increment_counts_missing_key(portfolio_store: PortfolioStore) -> None:
    ok = await portfolio_store.increment_counts("nonexistent", sparks=1)
    assert ok is False
