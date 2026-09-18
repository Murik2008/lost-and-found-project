"""Generate recognizable sample images for the Topic 1 lost-and-found dataset.

The PNGs in lost/ and found/ are ALREADY GENERATED — you do not need to run
this script. It is provided as a reference and a way to regenerate or
extend the sample set.

These are illustrative drawings (NOT photographs) of common lost items.
They have recognizable shapes, colors, and proportions so visual inspection
isn't misleading, but they should not be confused with real photographs.

For real testing, replace these PNGs with actual photos. The filenames
hint at the content so the offline `_OfflineVLM` in `demo_ai.py` can
produce a plausible identification without any real CV.

Requires Pillow (`pip install pillow`). Run from the topic root:
    python data/_make_samples.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

IMG = 256  # canvas size


def _new(bg=(245, 245, 240)):
    img = Image.new("RGB", (IMG, IMG), bg)
    return img, ImageDraw.Draw(img)


# ---------------------------------------------------------------------------
# Item drawings. Each function takes a slight color/proportion variation
# parameter so the "found" duplicate looks similar but not pixel-identical
# to its "lost" counterpart.
# ---------------------------------------------------------------------------

def draw_umbrella(variant: int = 0):
    img, d = _new()
    canopy_color = [(35, 35, 40), (45, 45, 55)][variant]
    handle_color = (130, 90, 50)
    cx, cy = 128, 110
    # canopy (half-circle)
    d.pieslice([cx - 80, cy - 80, cx + 80, cy + 80], 180, 360, fill=canopy_color)
    # ribs
    for x in (cx - 60, cx - 30, cx, cx + 30, cx + 60):
        d.line([(cx, cy), (x, cy - (80 - abs(cx - x) * 0.5))], fill=(15, 15, 20), width=1)
    # rim scallops
    for x in (cx - 80, cx - 50, cx - 20, cx + 10, cx + 40, cx + 70):
        d.arc([x - 5, cy - 8, x + 25, cy + 12], 0, 180, fill=(15, 15, 20), width=2)
    # shaft
    d.line([(cx, cy), (cx, cy + 110)], fill=(40, 40, 45), width=4)
    # crook handle
    d.arc([cx - 25, cy + 95, cx + 5, cy + 130], 0, 180, fill=handle_color, width=6)
    return img


def draw_backpack(variant: int = 0):
    img, d = _new()
    body_color = [(30, 45, 95), (35, 55, 110)][variant]
    strap_color = (20, 30, 70)
    # main body (rounded rectangle)
    d.rounded_rectangle([60, 70, 196, 220], radius=20, fill=body_color)
    # top flap
    d.rounded_rectangle([70, 60, 186, 130], radius=15, fill=body_color, outline=(15, 25, 60), width=2)
    # straps
    d.rectangle([70, 60, 90, 220], fill=strap_color)
    d.rectangle([166, 60, 186, 220], fill=strap_color)
    # buckle
    d.rectangle([115, 115, 141, 130], outline=(200, 200, 200), width=2)
    # front pocket
    d.rounded_rectangle([85, 150, 171, 200], radius=8, outline=(15, 25, 60), width=2)
    # zipper line on pocket
    d.line([(95, 175), (161, 175)], fill=(180, 180, 180), width=1)
    # logo patch
    d.rectangle([108, 80, 148, 100], fill=(220, 220, 220))
    return img


def draw_phone(variant: int = 0):
    img, d = _new()
    body_color = [(20, 20, 25), (30, 30, 35)][variant]
    # phone body (tall rounded rect)
    d.rounded_rectangle([88, 40, 168, 220], radius=18, fill=body_color, outline=(60, 60, 65), width=2)
    # screen
    d.rounded_rectangle([95, 55, 161, 200], radius=8, fill=(15, 15, 20))
    # camera notch
    d.ellipse([122, 60, 134, 72], fill=(5, 5, 10))
    # crack on the screen (distinguishing mark)
    d.line([(100, 90), (140, 130), (130, 170)], fill=(200, 200, 210), width=1)
    d.line([(140, 130), (155, 145)], fill=(200, 200, 210), width=1)
    # home indicator
    d.rounded_rectangle([116, 205, 140, 211], radius=3, fill=(80, 80, 85))
    return img


def draw_wallet(variant: int = 0):
    img, d = _new()
    leather = [(95, 60, 35), (110, 70, 40)][variant]
    stitch = (200, 170, 100)
    # main body
    d.rounded_rectangle([45, 90, 211, 180], radius=8, fill=leather, outline=(60, 35, 15), width=2)
    # tri-fold line
    d.line([(128, 92), (128, 178)], fill=(60, 35, 15), width=1)
    # stitching
    for x in range(55, 205, 8):
        d.line([(x, 95), (x + 4, 95)], fill=stitch, width=1)
        d.line([(x, 175), (x + 4, 175)], fill=stitch, width=1)
    # subtle highlight on top
    d.line([(50, 95), (206, 95)], fill=(140, 100, 60), width=1)
    return img


def draw_keys(variant: int = 0):
    img, d = _new()
    metal = [(190, 190, 195), (170, 170, 175)][variant]
    keychain = (40, 80, 160)
    # ring
    d.ellipse([110, 50, 146, 86], outline=metal, width=4)
    # keychain fob
    d.rounded_rectangle([118, 86, 138, 130], radius=4, fill=keychain)
    # three keys fanning down
    for angle, ox in zip((-25, 0, 25), (-50, 0, 50)):
        # shaft
        cx_top, cy_top = 128, 130
        cx_bot, cy_bot = 128 + ox, 200
        d.line([(cx_top, cy_top), (cx_bot, cy_bot)], fill=metal, width=6)
        # head
        d.ellipse([cx_top - 8, cy_top - 8, cx_top + 8, cy_top + 8], fill=metal)
        # teeth (small notches near bottom)
        d.rectangle([cx_bot - 8, cy_bot - 4, cx_bot - 4, cy_bot], fill=metal)
        d.rectangle([cx_bot, cy_bot - 8, cx_bot + 4, cy_bot - 4], fill=metal)
    return img


def draw_scarf():
    img, d = _new()
    red = (180, 40, 50)
    # diagonal scarf shape
    d.polygon([(40, 80), (220, 60), (200, 200), (50, 220)], fill=red)
    # fringe
    for y in range(75, 215, 12):
        d.line([(40, y), (28, y + 5)], fill=red, width=2)
    # plaid pattern
    for x in range(50, 220, 25):
        d.line([(x, 70), (x - 5, 215)], fill=(140, 20, 30), width=1)
    for y in range(80, 210, 25):
        d.line([(40, y), (220, y - 8)], fill=(140, 20, 30), width=1)
    return img


def draw_book():
    img, d = _new()
    cover = (40, 110, 60)
    pages = (240, 235, 220)
    # main cover
    d.rounded_rectangle([55, 50, 200, 215], radius=4, fill=cover, outline=(20, 60, 35), width=2)
    # spine highlight
    d.rectangle([55, 50, 70, 215], fill=(30, 90, 50))
    # pages (right edge)
    d.rectangle([200, 55, 208, 210], fill=pages)
    for y in range(60, 210, 4):
        d.line([(200, y), (208, y)], fill=(200, 195, 180), width=1)
    # title bar
    d.rectangle([85, 80, 185, 100], fill=(220, 215, 200))
    d.line([(95, 90), (175, 90)], fill=(50, 50, 50), width=1)
    # author bar
    d.rectangle([95, 165, 175, 180], fill=(220, 215, 200))
    return img


SAMPLES = [
    # (subdir, filename, drawer, variant)
    ("lost",  "umbrella_black.png",       draw_umbrella, 0),
    ("lost",  "backpack_navy.png",        draw_backpack, 0),
    ("lost",  "phone_apple_black.png",    draw_phone,    0),
    ("lost",  "wallet_brown.png",         draw_wallet,   0),
    ("lost",  "keys_silver.png",          draw_keys,     0),

    ("found", "umbrella_black_2.png",     draw_umbrella, 1),
    ("found", "backpack_navy_2.png",      draw_backpack, 1),
    ("found", "phone_apple_black_2.png",  draw_phone,    1),
    ("found", "wallet_brown_2.png",       draw_wallet,   1),
    ("found", "keys_silver_2.png",        draw_keys,     1),
    ("found", "scarf_red.png",            draw_scarf,    None),  # noise
    ("found", "book_green.png",           draw_book,     None),  # noise
]


def main() -> None:
    root = Path(__file__).parent
    for sub, name, fn, variant in SAMPLES:
        out = root / sub / name
        out.parent.mkdir(parents=True, exist_ok=True)
        img = fn(variant) if variant is not None else fn()
        img.save(out, "PNG")
        print(f"wrote {out.relative_to(root.parent)}")


if __name__ == "__main__":
    main()
