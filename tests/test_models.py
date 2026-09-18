"""Tests for src.models (your module)."""

from datetime import timezone

from src.models import Item, ItemStatus, ItemType, MatchRecord


def test_item_defaults():
    item = Item(
        item_type=ItemType.LOST,
        user_description="black umbrella",
        image_path="/tmp/x.png",
        description_json={"color": "black"},
        embedding=[0.1, 0.2, 0.3],
    )
    assert item.status == ItemStatus.PENDING
    assert item.id  # uuid generated
    # Must be timezone-aware (fixed from datetime.utcnow).
    assert item.created_at.tzinfo is not None
    assert item.created_at.utcoffset() == timezone.utc.utcoffset(None)


def test_match_record_defaults():
    m = MatchRecord(lost_item_id="l1", found_item_id="f1", score=0.9)
    assert m.reason == ""
    assert m.created_at.tzinfo is not None
