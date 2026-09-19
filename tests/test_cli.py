from pathlib import Path
import pytest

from src.cli import create_parser, main
from src.models import Item, ItemStatus, ItemType, MatchRecord


@pytest.fixture
def mock_repo(monkeypatch):
    """Mock storage for repository functions"""
    db: dict[str, Item] = {}
    matches_db: list[MatchRecord] = []

    async def fake_create_item(item: Item) -> Item:
        db[item.id] = item
        return item

    async def fake_get_item(item_id: str) -> Item | None:
        return db.get(item_id)

    async def fake_list_items(item_type=None, status=None, limit=50, offset=0) -> list[Item]:
        items = list(db.values())
        if item_type:
            items = [i for i in items if i.item_type == item_type]
        if status:
            items = [i for i in items if i.status == status]
        return items

    async def fake_update_item_status(item_id: str, status: ItemStatus) -> Item | None:
        item = db.get(item_id)
        if not item:
            return None
        item.status = status
        return item

    async def fake_get_matches_for_item(item_id: str, min_score=0.0) -> list[MatchRecord]:
        return [m for m in matches_db if m.lost_item_id == item_id or m.found_item_id == item_id]

    monkeypatch.setattr("src.storage.repository.create_item", fake_create_item)
    monkeypatch.setattr("src.storage.repository.get_item", fake_get_item)
    monkeypatch.setattr("src.storage.repository.list_items", fake_list_items)
    monkeypatch.setattr("src.storage.repository.update_item_status", fake_update_item_status)
    monkeypatch.setattr("src.storage.repository.get_matches_for_item", fake_get_matches_for_item)

    return {"db": db, "matches": matches_db}



def test_parser_subcommands():
    parser = create_parser()
    assert parser.prog == "lost-found"
    
    subparsers = next(a for a in parser._actions if a.dest == "command")
    cmds = set(subparsers.choices.keys())
    assert {"register-lost", "register-found", "list", "get", "update-status", "search-matches"}.issubset(cmds)



def test_register_lost_success(mock_repo, capsys):
    ret = main(["register-lost", "-d", "Black leather wallet", "-i", "photo.png"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "Success: Registered lost item with ID:" in out
    assert "Type: LOST" in out

    #Checking records in fake db
    assert len(mock_repo["db"]) == 1
    item = list(mock_repo["db"].values())[0]
    assert item.user_description == "Black leather wallet"
    assert item.item_type == ItemType.LOST
    assert item.status == ItemStatus.PENDING


def test_register_found_success(mock_repo, capsys):
    ret = main(["register-found", "-d", "House keys on blue lanyard"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "Success: Registered found item with ID:" in out

    item = list(mock_repo["db"].values())[0]
    assert item.item_type == ItemType.FOUND


def test_register_empty_description_error(mock_repo, capsys):
    ret = main(["register-lost", "-d", "   "])
    assert ret == 1

    err = capsys.readouterr().err
    assert "ERROR:" in err
    assert len(mock_repo["db"]) == 0


def test_register_bad_image_extension(mock_repo, tmp_path: Path, capsys):
    txt_file = tmp_path / "notes.txt"
    txt_file.write_text("not an image")

    ret = main(["register-lost", "-d", "Valid item", "-i", str(txt_file)])
    assert ret == 1

    err = capsys.readouterr().err
    assert "Unsupported image extension" in err
    assert len(mock_repo["db"]) == 0



def test_list_items(mock_repo, capsys):
    main(["register-lost", "-d", "Blue Backpack"])
    main(["register-found", "-d", "Red Umbrella"])
    capsys.readouterr()

    ret = main(["list"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "Found 2 item(s):" in out
    assert "Blue Backpack" in out
    assert "Red Umbrella" in out


def test_list_filter_by_type(mock_repo, capsys):
    main(["register-lost", "-d", "Lost Phone"])
    main(["register-found", "-d", "Found Keys"])
    capsys.readouterr()

    ret = main(["list", "--type", "lost"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "Found 1 item(s):" in out
    assert "Lost Phone" in out
    assert "Found Keys" not in out


def test_get_item_details(mock_repo, capsys):
    main(["register-lost", "-d", "Silver Ring"])
    item_id = list(mock_repo["db"].keys())[0]
    capsys.readouterr()

    ret = main(["get", "--item-id", item_id])
    assert ret == 0

    out = capsys.readouterr().out
    assert f"ID: {item_id}" in out
    assert "Silver Ring" in out


def test_get_nonexistent_item(mock_repo, capsys):
    ret = main(["get", "--item-id", "unknown_id_123"])
    assert ret == 1

    err = capsys.readouterr().err
    assert "ERROR: Item 'unknown_id_123' not found." in err



def test_update_status_success(mock_repo, capsys):
    main(["register-lost", "-d", "Black Watch"])
    item_id = list(mock_repo["db"].keys())[0]
    capsys.readouterr()

    ret = main(["update-status", "--item-id", item_id, "--status", "matched"])
    assert ret == 0

    out = capsys.readouterr().out
    assert "status to matched" in out
    assert mock_repo["db"][item_id].status == ItemStatus.MATCHED


def test_search_matches(mock_repo, capsys):
    main(["register-lost", "-d", "Lost AirPods"])
    lost_id = list(mock_repo["db"].keys())[0]
    main(["register-found", "-d", "Found AirPods"])
    found_id = list(mock_repo["db"].keys())[1]

    #mock match
    mock_repo["matches"].append(
        MatchRecord(lost_item_id=lost_id, found_item_id=found_id, score=0.92)
    )
    capsys.readouterr()

    ret = main(["search-matches", "--item-id", lost_id, "-k", "3"])
    assert ret == 0

    out = capsys.readouterr().out
    assert f"Top 1 match(es) for [LOST] item '{lost_id}':" in out
    assert found_id in out
    assert "Score: 0.92" in out