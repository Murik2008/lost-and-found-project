from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Sequence

from src.models import Item, ItemStatus, ItemType
from src.storage import repository

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_input(description: str, image_path: Optional[str]) -> Optional[str]:
    if not description or not description.strip():
        return "ERROR: Description cannot be empty or whitespace only."

    if image_path:
        ext = Path(image_path).suffix.lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            return (
                f"ERROR: Unsupported image extension '{ext}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}"
            )
    return None


async def handle_register(args: argparse.Namespace, item_type: ItemType) -> int:
    error_msg = validate_input(args.description, args.image)
    if error_msg:
        sys.stderr.write(f"{error_msg}\n")
        return 1

    item = Item(
        item_type=item_type,
        user_description=args.description.strip(),
        image_path=args.image or "",
        description_json={},
        embedding=[],
        status=ItemStatus.PENDING,
    )

    saved_item = await repository.create_item(item)

    sys.stdout.write(f"Success: Registered {item_type.value} item with ID: {saved_item.id}\n")
    sys.stdout.write(f"Type: {item_type.value.upper()}\n")
    return 0


async def handle_list(args: argparse.Namespace) -> int:
    item_type = ItemType(args.type.lower()) if getattr(args, "type", None) else None
    status = ItemStatus(args.status.lower()) if getattr(args, "status", None) else None

    items = await repository.list_items(item_type=item_type, status=status)

    sys.stdout.write(f"Found {len(items)} item(s):\n")
    for item in items:
        sys.stdout.write(
            f" - ID: {item.id} | [{item.item_type.value.upper()}] "
            f"{item.user_description} (Status: {item.status.value})\n"
        )
    return 0


async def handle_get(args: argparse.Namespace) -> int:
    item = await repository.get_item(args.item_id)
    if not item:
        sys.stderr.write(f"ERROR: Item '{args.item_id}' not found.\n")
        return 1

    sys.stdout.write(f"ID: {item.id}\n")
    sys.stdout.write(f"Description: {item.user_description}\n")
    sys.stdout.write(f"Type: {item.item_type.value}\n")
    sys.stdout.write(f"Status: {item.status.value}\n")
    return 0


async def handle_update_status(args: argparse.Namespace) -> int:
    new_status = ItemStatus(args.status.lower())
    updated_item = await repository.update_item_status(args.item_id, new_status)

    if not updated_item:
        sys.stderr.write(f"ERROR: Item '{args.item_id}' not found.\n")
        return 1

    sys.stdout.write(f"Success: Updated item {args.item_id} status to {new_status.value}\n")
    return 0


async def handle_search_matches(args: argparse.Namespace) -> int:
    target_item = await repository.get_item(args.item_id)
    if not target_item:
        sys.stderr.write(f"ERROR: Item '{args.item_id}' not found.\n")
        return 1

    matches = await repository.get_matches_for_item(args.item_id)
    matches = matches[: args.k]

    target_type_str = target_item.item_type.value.upper()
    sys.stdout.write(
        f"Top {len(matches)} match(es) for [{target_type_str}] item '{target_item.id}':\n"
    )
    for match in matches:
        cand_id = match.found_item_id if target_item.item_type == ItemType.LOST else match.lost_item_id
        sys.stdout.write(f" - Candidate ID: {cand_id} (Score: {match.score:.2f})\n")

    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lost-found",
        description="Command Line Interface for Lost and Found Project",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available subcommands")

    # register-lost
    p_lost = subparsers.add_parser("register-lost", help="Register a lost item")
    p_lost.add_argument("-d", "--description", type=str, required=True, help="Item description")
    p_lost.add_argument("-i", "--image", type=str, default=None, help="Path to image file")
    p_lost.set_defaults(func=lambda args: handle_register(args, ItemType.LOST))

    # register-found
    p_found = subparsers.add_parser("register-found", help="Register a found item")
    p_found.add_argument("-d", "--description", type=str, required=True, help="Item description")
    p_found.add_argument("-i", "--image", type=str, default=None, help="Path to image file")
    p_found.set_defaults(func=lambda args: handle_register(args, ItemType.FOUND))

    # search-matches
    p_search = subparsers.add_parser("search-matches", help="Search for item matches")
    p_search.add_argument("--item-id", type=str, required=True, help="Target item ID")
    p_search.add_argument("-k", type=int, default=5, help="Number of top matches")
    p_search.set_defaults(func=handle_search_matches)

    # list
    p_list = subparsers.add_parser("list", help="List registered items")
    p_list.add_argument("--type", type=str, choices=["lost", "found"], default=None, help="Filter by item type")
    p_list.add_argument("--status", type=str, choices=["pending", "matched", "closed"], default=None, help="Filter by status")
    p_list.set_defaults(func=handle_list)

    # get
    p_get = subparsers.add_parser("get", help="Get item details by ID")
    p_get.add_argument("--item-id", type=str, required=True, help="Target item ID")
    p_get.set_defaults(func=handle_get)

    # update-status
    p_update = subparsers.add_parser("update-status", help="Update item status")
    p_update.add_argument("--item-id", type=str, required=True, help="Target item ID")
    p_update.add_argument("--status", type=str, choices=["pending", "matched", "closed"], required=True, help="New status")
    p_update.set_defaults(func=handle_update_status)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1

    if hasattr(args, "func"):
        return asyncio.run(args.func(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())