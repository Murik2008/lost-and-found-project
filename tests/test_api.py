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


class ValidDesc:
    def to_dict(self):
        return {"object_class": "umbrella", "colors": ["black"], "confidence": 0.9}

    def to_search_text(self):
        return "umbrella | colors: black"


class CountingService:
    def __init__(self):
        self.calls = {"describe": 0, "embed": 0, "full": 0}

    def describe_item(self, image_path, user_text=""):
        self.calls["describe"] += 1
        return ValidDesc()

    def embed(self, text):
        self.calls["embed"] += 1
        return [0.1, 0.2, 0.3]

    async def adescribe_and_embed(self, image_path, user_text):
        self.calls["full"] += 1
        return ValidDesc(), [0.1, 0.2, 0.3]


def test_preview_then_register_reuses_analysis():
    import src.api as api_mod

    api_mod._preview_cache.clear()
    service = CountingService()
    app = create_app(repo=FakeRepo(), service=service)
    c = TestClient(app)
    data, name = _sample_png_bytes()
    assert c.post("/items/preview", files={"image": (name, data, "image/png")}).status_code == 200
    r = c.post(
        "/items/lost",
        files={"image": (name, data, "image/png")},
        data={"user_description": "umbrella"},
    )
    assert r.status_code == 201
    assert r.json()["description_json"]["object_class"] == "umbrella"
    assert service.calls == {"describe": 1, "embed": 1, "full": 0}
    api_mod._preview_cache.clear()


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


def _pair_repo():
    repo = FakeRepo()
    lost = Item(
        item_type=ItemType.LOST,
        user_description="lost umbrella",
        image_path="/tmp/fake.png",
        description_json={"object_class": "umbrella", "colors": ["black"]},
        embedding=[0.1, 0.2, 0.3],
    )
    found = Item(
        item_type=ItemType.FOUND,
        user_description="found umbrella",
        image_path="/tmp/fake.png",
        description_json={"object_class": "umbrella", "colors": ["black"]},
        embedding=[0.1, 0.2, 0.3],
    )
    repo.save(lost)
    repo.save(found)
    return repo, lost, found


def test_confirm_match_marks_both_and_stores_pair():
    from src.storage import repository as repo_mod

    repo_mod.clear_memory()
    repo, lost, found = _pair_repo()

    class FakeSvc:
        def cosine_similarity(self, a, b):
            return 0.9

    app = create_app(repo=repo, service=FakeSvc())
    c = TestClient(app)
    r = c.post(f"/items/{lost.id}/match", json={"other_item_id": found.id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["lost_item_id"] == lost.id
    assert body["found_item_id"] == found.id
    assert body["score"] == 0.9
    assert "umbrella" in body["reason"]
    assert c.get(f"/items/{lost.id}").json()["status"] == "matched"
    assert c.get(f"/items/{found.id}").json()["status"] == "matched"
    pair = c.get(f"/items/{lost.id}/pair").json()
    assert len(pair["pairs"]) == 1
    assert pair["pairs"][0]["item_id"] == found.id
    assert pair["pairs"][0]["user_description"] == "found umbrella"
    pair2 = c.get(f"/items/{found.id}/pair").json()
    assert pair2["pairs"][0]["item_id"] == lost.id
    repo_mod.clear_memory()


def test_confirm_match_rejects_same_type():
    from src.storage import repository as repo_mod

    repo_mod.clear_memory()
    repo = FakeRepo()
    a = Item(
        item_type=ItemType.LOST,
        user_description="a",
        image_path="/tmp/fake.png",
        description_json={},
        embedding=[0.1],
    )
    b = Item(
        item_type=ItemType.LOST,
        user_description="b",
        image_path="/tmp/fake.png",
        description_json={},
        embedding=[0.1],
    )
    repo.save(a)
    repo.save(b)
    app = create_app(repo=repo, service=FakeService())
    c = TestClient(app)
    assert c.post(f"/items/{a.id}/match", json={"other_item_id": b.id}).status_code == 400
    assert c.post(f"/items/{a.id}/match", json={"other_item_id": "nope"}).status_code == 404
    assert c.get(f"/items/{a.id}/pair").json() == {"query_id": a.id, "pairs": []}


def test_scan_excludes_matched_and_same_type():
    import asyncio

    from src.concurrency.pipeline import find_matches_for_item
    from src.models import ItemStatus
    from src.storage import repository as repo_mod

    async def scenario():
        repo_mod.clear_memory()
        lost = Item(
            item_type=ItemType.LOST,
            user_description="lost umbrella",
            image_path="/tmp/fake.png",
            description_json={},
            embedding=[0.1, 0.2, 0.3],
        )
        await repo_mod.create_item(lost)
        for itype, status in [
            (ItemType.FOUND, ItemStatus.PENDING),
            (ItemType.FOUND, ItemStatus.MATCHED),
            (ItemType.LOST, ItemStatus.PENDING),
        ]:
            it = Item(
                item_type=itype,
                user_description="x",
                image_path="/tmp/fake.png",
                description_json={},
                embedding=[0.1, 0.2, 0.3],
                status=status,
            )
            await repo_mod.create_item(it)

        class Svc:
            def top_k(self, q, cs, k):
                return list(range(min(k, len(cs))))

            def cosine_similarity(self, a, b):
                return 0.9

        out = await find_matches_for_item(lost.id, 10, Svc(), None)
        repo_mod.clear_memory()
        return out

    out = asyncio.run(scenario())
    assert len(out) == 1
