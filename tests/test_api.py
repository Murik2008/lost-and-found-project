"""Tests for src.api (your module) using fakes.

No real DB / AI needed: repo + service are injected, image storage is
monkeypatched. This is why api.py uses lazy imports for teammate modules.
"""

import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai.providers.base import ProviderError
from src.api import create_app
from src.models import Item, ItemStatus, ItemType

DATA_LOST = Path(__file__).parent.parent / "data" / "lost"


class FakeDesc:
    def to_dict(self):
        return {"color": "black", "category": "umbrella"}


class FakeRepo:
    def __init__(self):
        self.items: dict[str, Item] = {}

    def save(self, item: Item) -> Item:
        self.items[item.id] = item
        return item

    def get(self, item_id: str) -> Item | None:
        return self.items.get(item_id)

    def list(
        self,
        item_type: ItemType | None = None,
        status: ItemStatus | None = None,
    ) -> list[Item]:
        out = list(self.items.values())
        if item_type is not None:
            out = [i for i in out if i.item_type == item_type]
        if status is not None:
            out = [i for i in out if i.status == status]
        return out

    def update_item_status(self, item_id: str, status: ItemStatus) -> Item | None:
        item = self.items.get(item_id)
        if item is None:
            return None
        item.status = status
        return item

    def delete_item(self, item_id: str) -> bool:
        return self.items.pop(item_id, None) is not None


class FakeService:
    def __init__(self, vec=None):
        self.vec = vec or [0.1, 0.2, 0.3]

    async def adescribe_and_embed(self, image_path: str, user_text: str):
        return FakeDesc(), self.vec


class FailingService:
    async def adescribe_and_embed(self, image_path: str, user_text: str):
        raise ProviderError("boom")


@pytest.fixture()
def client(monkeypatch):
    repo = FakeRepo()
    service = FakeService()
    # Bypass real filesystem storage (teammate module).
    import src.api as api_mod

    monkeypatch.setattr(
        api_mod, "_store_image_bytes", lambda data, filename, item_type: "/tmp/fake.png"
    )
    app = create_app(repo=repo, service=service)
    return TestClient(app), repo, service


def _sample_png_bytes() -> tuple[bytes, str]:
    candidates = list(DATA_LOST.glob("*.png"))
    assert candidates, "data/lost/*.png samples missing"
    return candidates[0].read_bytes(), candidates[0].name


def test_health(client):
    c, _, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_register_lost_ok(client):
    c, repo, _ = client
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "black umbrella"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["item_type"] == "lost"
    assert body["user_description"] == "black umbrella"
    assert len(repo.items) == 1


def test_register_empty_text_422(client):
    c, _, _ = client
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "   "},
    )
    assert r.status_code == 422


def test_register_bad_image_400(client):
    c, _, _ = client
    r = c.post(
        "/items/lost",
        files={"image": ("evil.txt", b"not-an-image", "text/plain")},
        data={"user_description": "umbrella"},
    )
    assert r.status_code == 400


def test_register_ai_failure_502(monkeypatch):
    repo = FakeRepo()
    import src.api as api_mod

    monkeypatch.setattr(
        api_mod, "_store_image_bytes", lambda data, filename, item_type: "/tmp/fake.png"
    )
    app = create_app(repo=repo, service=FailingService())
    c = TestClient(app)
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "umbrella"},
    )
    assert r.status_code == 502


def test_list_filter_by_type_and_status(client):
    c, repo, _ = client
    data, name = _sample_png_bytes()
    c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "lost umbrella"},
    )
    c.post(
        "/items/found",
        files={"image": (name, data, "image/png")},
        data={"user_description": "found umbrella"},
    )
    # ?type= alias still works (no builtin shadowing in signature).
    assert len(c.get("/items?type=lost").json()) == 1
    assert len(c.get("/items?type=found").json()) == 1
    assert len(c.get("/items").json()) == 2
    # Both ?status= and ?item_status= accepted.
    assert len(c.get("/items?status=pending").json()) == 2
    assert len(c.get("/items?item_status=pending").json()) == 2


def test_get_item_404(client):
    c, _, _ = client
    assert c.get("/items/nope").status_code == 404


def test_matches_pipeline_missing_501():
    # No fakes for pipeline -> lazy import fails -> 501, not crash.
    repo = FakeRepo()
    service = FakeService()
    app = create_app(repo=repo, service=service)
    c = TestClient(app)
    # Need an item first so we get past repo checks; use direct save.
    item = Item(
        item_type=ItemType.LOST,
        user_description="umbrella",
        image_path="/tmp/fake.png",
        description_json={},
        embedding=[0.1, 0.2, 0.3],
    )
    repo.save(item)
    r = c.get(f"/items/{item.id}/matches")
    # pipeline.py is still empty in repo -> 501 expected.
    # If teammates implement it later, this will become 200.
    assert r.status_code in (200, 501)


def test_matches_ok_with_fake_pipeline(monkeypatch):
    repo = FakeRepo()
    service = FakeService()
    item = Item(
        item_type=ItemType.LOST,
        user_description="umbrella",
        image_path="/tmp/fake.png",
        description_json={},
        embedding=[0.1, 0.2, 0.3],
    )
    repo.save(item)

    fake_mod = types.ModuleType("src.concurrency.pipeline")

    async def fake_find(item_id, k, svc, rp):
        assert item_id == item.id
        return [("found-1", 0.95, "color match")]

    fake_mod.find_matches_for_item = fake_find
    monkeypatch.setitem(sys.modules, "src.concurrency.pipeline", fake_mod)

    app = create_app(repo=repo, service=service)
    c = TestClient(app)
    r = c.get(f"/items/{item.id}/matches?k=1")
    assert r.status_code == 200
    body = r.json()
    assert body["query_id"] == item.id
    assert body["matches"][0]["item_id"] == "found-1"


def test_update_status_ok(client):
    c, repo, _ = client
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "umbrella"},
    )
    item_id = r.json()["id"]
    r = c.patch(f"/items/{item_id}/status", json={"status": "matched"})
    assert r.status_code == 200
    assert r.json()["status"] == "matched"
    assert "owner_token" not in r.json()


def test_update_status_404(client):
    c, _, _ = client
    assert c.patch("/items/nope/status", json={"status": "closed"}).status_code == 404


def test_no_owner_token_leaked(client):
    c, _, _ = client
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "umbrella"},
    )
    assert "owner_token" not in r.json()
    for entry in c.get("/items").json():
        assert "owner_token" not in entry

def test_update_status_invalid(client):
    c, _, _ = client
    r = c.patch("/items/nope/status", json={"status": "bogus"})
    assert r.status_code == 422


def test_delete_ok(client):
    c, repo, _ = client
    data, name = _sample_png_bytes()
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "umbrella"},
    )
    item_id = r.json()["id"]
    r = c.delete(f"/items/{item_id}")
    assert r.status_code == 200
    assert r.json() == {"deleted": item_id}
    assert c.get(f"/items/{item_id}").status_code == 404


def test_delete_404(client):
    c, _, _ = client
    assert c.delete("/items/nope").status_code == 404


def test_image_ok():
    repo = FakeRepo()
    data, name = _sample_png_bytes()
    item = Item(
        item_type=ItemType.LOST,
        user_description="umbrella",
        image_path=str(Path(__file__).parent.parent / "data" / "lost" / name),
        description_json={},
        embedding=[0.1, 0.2, 0.3],
    )
    repo.save(item)
    app = create_app(repo=repo, service=FakeService())
    c = TestClient(app)
    r = c.get(f"/items/{item.id}/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == data


def test_image_404(client):
    c, _, _ = client
    assert c.get("/items/nope/image").status_code == 404


def test_preview_ok(client):
    c, _, _ = client
    data, name = _sample_png_bytes()
    r = c.post("/items/preview", files={"image": (name, data, "image/png")})
    assert r.status_code == 200
    assert r.json()["description"] == {"color": "black", "category": "umbrella"}


def test_preview_bad_image(client):
    c, _, _ = client
    r = c.post("/items/preview", files={"image": ("evil.txt", b"xx", "text/plain")})
    assert r.status_code == 400


def test_matches_min_score_filters(monkeypatch):
    repo = FakeRepo()
    service = FakeService()
    item = Item(
        item_type=ItemType.LOST,
        user_description="umbrella",
        image_path="/tmp/fake.png",
        description_json={},
        embedding=[0.1, 0.2, 0.3],
    )
    repo.save(item)

    fake_mod = types.ModuleType("src.concurrency.pipeline")

    async def fake_find(item_id, k, svc, rp):
        return [("a", 0.9, "good"), ("b", 0.3, "weak")]

    fake_mod.find_matches_for_item = fake_find
    monkeypatch.setitem(sys.modules, "src.concurrency.pipeline", fake_mod)

    app = create_app(repo=repo, service=service)
    c = TestClient(app)
    body = c.get(f"/items/{item.id}/matches?k=5").json()
    assert len(body["matches"]) == 2
    assert body["matches"][0]["image_url"] == "/items/a/image"
    body = c.get(f"/items/{item.id}/matches?k=5&min_score=0.5").json()
    assert [m["item_id"] for m in body["matches"]] == ["a"]
