"""2D dungeon map renderer with BFS grid placement and corridor drawing."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from src.escape_rooms.state import Room, WorldObject

CELL_MIN_W = 30
CELL_MIN_H = 7
H_GAP = 5
V_GAP = 3

OPPOSITES = {"north": "south", "south": "north", "east": "west", "west": "east"}
DELTA = {"east": (1, 0), "west": (-1, 0), "south": (0, 1), "north": (0, -1)}


def _place_rooms(rooms: list[Room]) -> dict[str, tuple[int, int]]:
    """BFS from first room, assign grid (col, row) to each room by adjacency."""
    id_to_room = {r.id: r for r in rooms}
    placed: dict[str, tuple[int, int]] = {}
    queue: deque[str] = deque()

    placed[rooms[0].id] = (0, 0)
    queue.append(rooms[0].id)

    while queue:
        current_id = queue.popleft()
        current_pos = placed[current_id]
        room = id_to_room.get(current_id)
        if room is None:
            continue
        for direction, neighbor_id in room.adjacency.items():
            if neighbor_id in placed:
                continue
            dc, dr = DELTA.get(direction, (0, 0))
            neighbor_pos = (current_pos[0] + dc, current_pos[1] + dr)
            placed[neighbor_id] = neighbor_pos
            if neighbor_id in id_to_room:
                queue.append(neighbor_id)

    if placed:
        max_row = max(r for _, r in placed.values())
    else:
        max_row = -1
    fallback_col = 0
    for room in rooms:
        if room.id not in placed:
            max_row += 1
            placed[room.id] = (fallback_col, max_row)
            fallback_col += 1

    min_col = min(c for c, _ in placed.values())
    min_row = min(r for _, r in placed.values())
    return {rid: (c - min_col, r - min_row) for rid, (c, r) in placed.items()}


def _cell_text_rows(
    room: Room,
    objects_here: list[WorldObject],
    party_marker: str = "",
    interacted_ids: set[str] | None = None,
    object_states: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Return (name_row, object_rows) as raw left-padded text, no borders.

    These are the strings whose lengths drive the cell's required width.
    """
    interacted_ids = interacted_ids or set()
    object_states = object_states or {}

    name_text = room.id.upper()
    if party_marker:
        name_text = f"{name_text} {party_marker}"

    obj_rows: list[str] = []
    for obj in objects_here:
        mark = "[x]" if obj.id in interacted_ids else "[ ]"
        state = object_states.get(obj.id, obj.state)
        obj_rows.append(f"  {mark} {obj.id} ({state})")
    if not obj_rows:
        obj_rows.append("  (empty)")
    return name_text, obj_rows


def _cell_required_inner(name_text: str, obj_rows: list[str]) -> int:
    """Inner width needed so every text row fits without truncation.

    Each row is rendered as " " + text, so it needs len(text) + 1 columns.
    """
    candidates = [len(name_text), len(" Objects:")] + [len(r) for r in obj_rows]
    return max(candidates) + 1


def _build_cell(
    name_text: str,
    obj_rows: list[str],
    cell_w: int,
    cell_h: int,
) -> list[str]:
    """Build a `cell_w` x `cell_h` box for one room from pre-computed text rows.

    Caller guarantees cell_w/cell_h fit all content (no truncation here)."""
    inner = cell_w - 2
    h, v = "─", "│"
    tl, tr, bl, br, lm, rm = "┌", "┐", "└", "┘", "├", "┤"

    lines: list[str] = [
        tl + h * inner + tr,
        v + f" {name_text}".ljust(inner) + v,
        lm + h * inner + rm,
        v + " Objects:".ljust(inner) + v,
    ]

    obj_lines = [v + f" {row}".ljust(inner) + v for row in obj_rows]

    while len(obj_lines) < cell_h - 5:
        obj_lines.append(v + " " * inner + v)

    lines.extend(obj_lines)
    lines.append(bl + h * inner + br)

    assert len(lines) == cell_h, f"Cell height mismatch: {len(lines)} != {cell_h}"
    return lines


def _make_canvas(width: int, height: int) -> list[list[str]]:
    return [[" "] * width for _ in range(height)]


def _blit(canvas: list[list[str]], lines: list[str], x: int, y: int) -> None:
    for dy, line in enumerate(lines):
        for dx, ch in enumerate(line):
            if 0 <= y + dy < len(canvas) and 0 <= x + dx < len(canvas[0]):
                canvas[y + dy][x + dx] = ch


def _draw_h_corridor(canvas: list[list[str]], x_start: int, x_end: int, y: int) -> None:
    for x in range(x_start, x_end):
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
            canvas[y][x] = "─"


def _draw_v_corridor(canvas: list[list[str]], x: int, y_start: int, y_end: int) -> None:
    for y in range(y_start, y_end):
        if 0 <= y < len(canvas) and 0 <= x < len(canvas[0]):
            canvas[y][x] = "│"


def render_room_layout(
    rooms: list[Room],
    objects: list[WorldObject] | None = None,
    party_room: str = "",
    party_label: str = "★",
    interacted_ids: set[str] | None = None,
    object_states: dict[str, str] | None = None,
) -> None:
    if not rooms:
        print("  (no rooms to display)")
        return

    objects = objects or []
    objects_by_room: dict[str, list[WorldObject]] = {r.id: [] for r in rooms}
    for obj in objects:
        if obj.location in objects_by_room:
            objects_by_room[obj.location].append(obj)

    grid = _place_rooms(rooms)
    id_to_room = {r.id: r for r in rooms}

    max_col = max(c for c, _ in grid.values())
    max_row = max(r for _, r in grid.values())

    # Pre-compute each room's text rows once, then size the shared cell to fit them.
    cell_text: dict[str, tuple[str, list[str]]] = {}
    for rid in grid:
        room = id_to_room.get(rid)
        if room is None:
            continue
        marker = party_label if rid == party_room else ""
        cell_text[rid] = _cell_text_rows(
            room,
            objects_by_room.get(rid, []),
            marker,
            interacted_ids=interacted_ids,
            object_states=object_states,
        )

    # All cells share one width/height — sized to the most demanding room, so the
    # grid stays aligned and corridors connect. Width fits all text (X axis), like
    # height already fits all objects (Y axis).
    required_inner = max(
        (_cell_required_inner(name, rows) for name, rows in cell_text.values()),
        default=0,
    )
    cell_w = max(CELL_MIN_W, required_inner + 2)

    max_objs = max((len(objs) for objs in objects_by_room.values()), default=0)
    cell_h = max(CELL_MIN_H, 5 + max(1, max_objs))

    canvas_w = (max_col + 1) * cell_w + max_col * H_GAP
    canvas_h = (max_row + 1) * cell_h + max_row * V_GAP
    canvas = _make_canvas(canvas_w, canvas_h)

    for rid, (col, row) in grid.items():
        if rid not in cell_text:
            continue
        name_text, obj_rows = cell_text[rid]
        cell_x = col * (cell_w + H_GAP)
        cell_y = row * (cell_h + V_GAP)
        _blit(
            canvas,
            _build_cell(name_text, obj_rows, cell_w, cell_h),
            cell_x,
            cell_y,
        )

    for rid, (col, row) in grid.items():
        room = id_to_room.get(rid)
        if room is None:
            continue
        cell_x = col * (cell_w + H_GAP)
        cell_y = row * (cell_h + V_GAP)

        if "east" in room.adjacency and room.adjacency["east"] in grid:
            neighbor_col, _ = grid[room.adjacency["east"]]
            if neighbor_col == col + 1:
                corridor_y = cell_y + 1
                x_start = cell_x + cell_w
                x_end = cell_x + cell_w + H_GAP
                _draw_h_corridor(canvas, x_start, x_end, corridor_y)

        if "south" in room.adjacency and room.adjacency["south"] in grid:
            _, neighbor_row = grid[room.adjacency["south"]]
            if neighbor_row == row + 1:
                corridor_x = cell_x + cell_w // 2
                y_start = cell_y + cell_h
                y_end = cell_y + cell_h + V_GAP
                _draw_v_corridor(canvas, corridor_x, y_start, y_end)

    for row_chars in canvas:
        print("".join(row_chars).rstrip())


# Room dimensions in tiles (interior, excluding walls).
ROOM_W = 10
ROOM_H = 10
TILE_SIZE = 16  # px, for SwiftUI to scale sprites

# Direction -> (door tileX, door tileY) on the room's interior edge.
# Doors sit centered on each wall: north/south center on X, east/west center on Y.
_DOOR_TILE: dict[str, tuple[int, int]] = {
    "north": (ROOM_W // 2, 0),
    "south": (ROOM_W // 2, ROOM_H - 1),
    "east":  (ROOM_W - 1, ROOM_H // 2),
    "west":  (0,           ROOM_H // 2),
}

# Keywords in object IDs -> sprite name.
# Checked in order; first match wins. Falls back to a generic bucket.
_SPRITE_RULES: list[tuple[tuple[str, ...], str]] = [
    # Doors
    (("door",),                          "door"),
    # Containers / storage
    (("chest", "cabinet", "box", "crate", "locker", "safe"), "chest"),
    (("drawer", "dresser", "wardrobe"),  "dresser"),
    (("bookshelf", "shelf", "bookcase"), "bookshelf"),
    # Furniture
    (("table", "desk"),                  "table"),
    (("chair",),                         "chair"),
    (("bed",),                           "bed"),
    (("sofa", "couch"),                  "sofa"),
    (("rug", "carpet", "mat"),           "rug"),
    # Interactive puzzle objects
    (("mirror",),                        "mirror"),
    (("clock",),                         "clock"),
    (("note", "letter", "journal", "diary", "scroll", "paper"), "note"),
    (("key",),                           "key"),
    (("lever", "switch", "button", "panel", "mechanism", "fuse"), "lever"),
    (("altar", "pedestal", "statue", "idol", "shrine"),           "altar"),
    (("lens", "glass", "crystal", "gem", "orb", "stone", "seal", "amulet", "relic"), "artifact"),
    (("machine", "device", "computer", "terminal", "generator"),  "machine"),
    (("barrel", "keg"),                  "barrel"),
    (("lamp", "lantern", "candle", "torch", "light", "fire"),     "light"),
    (("window", "curtain"),              "window"),
    (("painting", "picture", "portrait", "frame"),                "painting"),
    (("plant", "flower", "tree", "vine"), "plant"),
    # Atmospheric catch-alls
    (("floor", "blood", "stain", "mark", "symbol", "rune"),      "floor_detail"),
    (("wall", "crack", "brick"),         "wall_detail"),
    (("shadow", "figure", "whisper", "voice", "filler", "clutter", "debris", "sky", "storm", "cloud"), "item"),
]


def _sprite_for(obj_id: str) -> str:
    lower = obj_id.lower()
    for keywords, sprite in _SPRITE_RULES:
        if any(k in lower for k in keywords):
            return sprite
    return "item"  # generic fallback


# Placement category per sprite — drives which slot pool an object draws from.
# "wall": large furniture anchored to a wall tile (not a corner, not a door tile).
# "wall_decor": mounted on a wall tile, can share a wall with "wall" furniture.
# "floor": floor coverings, centered in the room.
# "small": interactive/portable items, placed in interior tiles near furniture.
# "detail": atmospheric overlays, lowest priority, fill whatever is left.
_PLACEMENT_CATEGORY: dict[str, str] = {
    "chest": "wall",
    "dresser": "wall",
    "bookshelf": "wall",
    "table": "wall",
    "bed": "wall",
    "sofa": "wall",
    "altar": "wall",
    "machine": "wall",
    "barrel": "wall",
    "mirror": "wall_decor",
    "clock": "wall_decor",
    "painting": "wall_decor",
    "window": "wall_decor",
    "rug": "floor",
    "chair": "small",
    "note": "small",
    "key": "small",
    "lever": "small",
    "artifact": "small",
    "light": "small",
    "plant": "small",
    "item": "small",
    "floor_detail": "detail",
    "wall_detail": "detail",
}


def _placement_category(sprite: str) -> str:
    return _PLACEMENT_CATEGORY.get(sprite, "small")


def _wall_tiles(w: int, h: int, door_tiles: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Interior wall-adjacent tiles (1-tile inset), excluding corners and door tiles."""
    tiles: list[tuple[int, int]] = []
    for x in range(2, w - 2):
        tiles.append((x, 1))
        tiles.append((x, h - 2))
    for y in range(2, h - 2):
        tiles.append((1, y))
        tiles.append((w - 2, y))
    return [t for t in tiles if t not in door_tiles]


def _floor_tiles(w: int, h: int) -> list[tuple[int, int]]:
    """Center tiles, away from walls."""
    tiles: list[tuple[int, int]] = []
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            tiles.append((x, y))
    return tiles


def _interior_tiles(w: int, h: int, door_tiles: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """All non-wall interior tiles, excluding door tiles."""
    tiles: list[tuple[int, int]] = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            tiles.append((x, y))
    return [t for t in tiles if t not in door_tiles]


def _adjacent_tiles(tile: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = tile
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _place_objects(
    objs: list[WorldObject],
    room_id: str,
    door_tiles: set[tuple[int, int]],
    w: int = ROOM_W,
    h: int = ROOM_H,
) -> dict[str, tuple[int, int]]:
    """Assign each object a (tileX, tileY), grouped by placement category.

    Large furniture anchors to wall tiles first; small/interactive items
    prefer a tile next to a placed furniture piece (so they read as "on" or
    "beside" it); leftover items and atmospheric details fill whatever floor
    tiles remain. Each room gets its own seeded shuffle so layouts vary
    between rooms but stay stable across re-renders of the same world.
    """
    rng = random.Random(room_id)

    wall_slots = _wall_tiles(w, h, door_tiles)
    floor_slots = _floor_tiles(w, h)
    interior_slots = _interior_tiles(w, h, door_tiles)
    rng.shuffle(wall_slots)
    rng.shuffle(floor_slots)
    rng.shuffle(interior_slots)

    occupied: set[tuple[int, int]] = set()
    placements: dict[str, tuple[int, int]] = {}
    furniture_tiles: list[tuple[int, int]] = []

    by_category: dict[str, list[WorldObject]] = {"wall": [], "wall_decor": [], "floor": [], "small": [], "detail": []}
    for obj in objs:
        by_category[_placement_category(_sprite_for(obj.id))].append(obj)

    def take(pool: list[tuple[int, int]]) -> tuple[int, int] | None:
        while pool:
            tile = pool.pop()
            if tile not in occupied:
                occupied.add(tile)
                return tile
        return None

    # Large furniture and wall decor anchor to the walls first.
    for obj in by_category["wall"] + by_category["wall_decor"]:
        tile = take(wall_slots) or take(interior_slots)
        if tile is None:
            continue
        placements[obj.id] = tile
        if _placement_category(_sprite_for(obj.id)) == "wall":
            furniture_tiles.append(tile)

    # Floor coverings (rugs) take center tiles.
    for obj in by_category["floor"]:
        tile = take(floor_slots) or take(interior_slots)
        if tile is None:
            continue
        placements[obj.id] = tile

    # Small/interactive items prefer a tile beside a furniture piece.
    rng.shuffle(furniture_tiles)
    for obj in by_category["small"]:
        tile = None
        for anchor in furniture_tiles:
            for candidate in _adjacent_tiles(anchor):
                if candidate in occupied or candidate not in interior_slots:
                    continue
                interior_slots.remove(candidate)
                occupied.add(candidate)
                tile = candidate
                break
            if tile:
                break
        if tile is None:
            tile = take(interior_slots) or take(wall_slots)
        if tile is None:
            continue
        placements[obj.id] = tile

    # Atmospheric details fill whatever is left, allowing overlap as a last resort.
    leftover = interior_slots + floor_slots + wall_slots
    for obj in by_category["detail"]:
        tile = take(leftover)
        if tile is None:
            tile = (w // 2, h // 2)
        placements[obj.id] = tile

    return placements


def render_world(
    rooms: list[Room],
    objects: list[WorldObject],
    current_room: str = "",
    inventory: list[str] | None = None,
    object_states: dict[str, str] | None = None,
    tick: int = 0,
) -> dict[str, Any]:
    """Return a SwiftUI tile-map render payload.

    Each room has pixel dimensions (widthTiles x heightTiles), a floor/wall
    tile hint, doors placed on the correct walls, and every object assigned a
    (tileX, tileY) position and a sprite key. Corridors are deduplicated
    undirected edges between rooms.
    """
    if not rooms:
        return {"grid": {"cols": 0, "rows": 0, "tileSize": TILE_SIZE}, "rooms": [], "corridors": [], "party": {}}

    inventory = inventory or []
    object_states = object_states or {}

    grid = _place_rooms(rooms)
    max_col = max(c for c, _ in grid.values())
    max_row = max(r for _, r in grid.values())

    room_ids = {r.id for r in rooms}
    objects_by_id = {o.id: o for o in objects}

    def _resolve_room(obj: WorldObject) -> str | None:
        """Walk an object's location chain until it reaches a room id.

        Objects can be nested inside other objects (location = another
        object's id); follow that chain to find the room they ultimately
        belong to. Guards against cycles with a visited set.
        """
        seen: set[str] = set()
        loc = obj.location
        while loc not in room_ids:
            if loc in seen or loc not in objects_by_id:
                return None
            seen.add(loc)
            loc = objects_by_id[loc].location
        return loc

    objects_by_room: dict[str, list[WorldObject]] = {r.id: [] for r in rooms}
    for obj in objects:
        room_id = _resolve_room(obj)
        if room_id is not None:
            objects_by_room[room_id].append(obj)

    room_list = []
    for room in rooms:
        col, row = grid[room.id]
        room_objs = objects_by_room.get(room.id, [])

        # Doors — one per adjacency direction, locked state derived from object states.
        doors = []
        for direction, neighbor_id in room.adjacency.items():
            tx, ty = _DOOR_TILE.get(direction, (ROOM_W // 2, ROOM_H // 2))
            # Find a door object in this room whose state we can read.
            door_obj = next(
                (o for o in room_objs if "door" in o.id.lower()), None
            )
            locked = (
                object_states.get(door_obj.id, door_obj.state) not in ("unlocked", "open")
                if door_obj else False
            )
            doors.append({
                "direction": direction,
                "toRoom": neighbor_id,
                "tileX": tx,
                "tileY": ty,
                "locked": locked,
            })

        # Place all of this room's objects naturally: furniture against walls,
        # small/interactive items near furniture, details filling what's left.
        # Objects whose id contains "door" are puzzle objects in their own
        # right (e.g. a locked cell door with its own code) and are rendered
        # like everything else; the synthetic nav-doors below are independent
        # tiles derived from room.adjacency.
        door_tiles = {(d["tileX"], d["tileY"]) for d in doors}
        placements = _place_objects(room_objs, room.id, door_tiles)
        rendered_objects = []
        for obj in room_objs:
            tx, ty = placements.get(obj.id, (ROOM_W // 2, ROOM_H // 2))
            rendered_objects.append({
                "id": obj.id,
                "sprite": _sprite_for(obj.id),
                "tileX": tx,
                "tileY": ty,
                "state": object_states.get(obj.id, obj.state),
                "interacted": obj.id in inventory,
                "takeable": obj.takeable,
                "interactable": obj.interactable,
            })

        room_list.append({
            "id": room.id,
            "label": room.id.upper(),
            "col": col,
            "row": row,
            "widthTiles": ROOM_W,
            "heightTiles": ROOM_H,
            "floorTile": "floor_wood",
            "wallTile": "wall_stone",
            "isCurrentRoom": room.id == current_room,
            "doors": doors,
            "objects": rendered_objects,
        })

    seen: set[frozenset[str]] = set()
    corridors = []
    for room in rooms:
        for direction, neighbor_id in room.adjacency.items():
            key = frozenset({room.id, neighbor_id})
            if key not in seen:
                corridors.append({"fromRoom": room.id, "toRoom": neighbor_id, "direction": direction})
                seen.add(key)

    return {
        "grid": {"cols": max_col + 1, "rows": max_row + 1, "tileSize": TILE_SIZE},
        "rooms": room_list,
        "corridors": corridors,
        "party": {
            "currentRoom": current_room,
            "inventory": inventory,
            "tick": tick,
        },
    }
