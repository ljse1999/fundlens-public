"""Unit tests for fundlens.cache.DiskCache."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fundlens.cache import DiskCache, _sanitise_key


@pytest.fixture
def cache(tmp_path):
    return DiskCache(cache_dir=tmp_path / "cache")


def test_df_roundtrip(cache):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    cache.put_df("navs/GB00B41YBW71/monthly", df)
    loaded = cache.get_df("navs/GB00B41YBW71/monthly", ttl_days=1)
    assert loaded is not None
    pd.testing.assert_frame_equal(df, loaded)


def test_json_roundtrip(cache):
    obj = {"foo": "bar", "n": 42, "list": [1, 2, 3]}
    cache.put_json("meta/GB00B41YBW71", obj)
    loaded = cache.get_json("meta/GB00B41YBW71", ttl_days=1)
    assert loaded == obj


def test_missing_key_returns_none(cache):
    assert cache.get_df("does/not/exist", ttl_days=1) is None
    assert cache.get_json("does/not/exist", ttl_days=1) is None


def test_ttl_expiry_df(cache):
    df = pd.DataFrame({"a": [1, 2, 3]})
    cache.put_df("navs/expiring", df)

    # Force the sidecar meta file to look old.
    meta_path = cache._meta_path("navs/expiring")
    old_ts = datetime.now(timezone.utc) - timedelta(days=10)
    meta_path.write_text(json.dumps({"written_at": old_ts.isoformat()}), encoding="utf-8")

    assert cache.get_df("navs/expiring", ttl_days=5) is None
    # Still fresh enough under a longer TTL.
    assert cache.get_df("navs/expiring", ttl_days=30) is not None


def test_ttl_expiry_json(cache):
    obj = {"x": 1}
    cache.put_json("meta/expiring", obj)

    meta_path = cache._meta_path("meta/expiring")
    old_ts = datetime.now(timezone.utc) - timedelta(days=10)
    meta_path.write_text(json.dumps({"written_at": old_ts.isoformat()}), encoding="utf-8")

    assert cache.get_json("meta/expiring", ttl_days=5) is None
    assert cache.get_json("meta/expiring", ttl_days=30) is not None


def test_key_sanitisation():
    assert _sanitise_key("navs/GB00B41YBW71/monthly") == "navs__GB00B41YBW71__monthly"
    assert "/" not in _sanitise_key("a/b/c")
    # Unsafe characters get replaced, not left as-is.
    weird = _sanitise_key("a b:c*d?e")
    assert all(ch not in weird for ch in " :*?")


def test_different_keys_do_not_collide(cache):
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    cache.put_df("navs/AAA/monthly", df1)
    cache.put_df("navs/BBB/monthly", df2)
    pd.testing.assert_frame_equal(cache.get_df("navs/AAA/monthly", ttl_days=1), df1)
    pd.testing.assert_frame_equal(cache.get_df("navs/BBB/monthly", ttl_days=1), df2)
