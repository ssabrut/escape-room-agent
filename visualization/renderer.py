"""2D dungeon map renderer with BFS grid placement and corridor drawing."""

from __future__ import annotations

from collections import deque

from state import Room, WorldObject

CELL_W = 30
CELL_H = 7
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


def _build_cell(
    room: Room,
    objects_here: list[WorldObject],
    party_marker: str = "",
    interacted_ids: set[str] | None = None,
    object_states: dict[str, str] | None = None,
) -> list[str]:
    """Build a fixed CELL_H-line representation of one room."""
    inner = CELL_W - 2
    h, v = "─", "│"
    tl, tr, bl, br, lm, rm = "┌", "┐", "└", "┘", "├", "┤"
    interacted_ids = interacted_ids or set()
    object_states = object_states or {}

    name_text = room.id.upper()
    if party_marker:
        name_text = f"{name_text} {party_marker}"
    name = name_text[: inner - 1]
    lines: list[str] = [
        tl + h * inner + tr,
        v + f" {name}".ljust(inner) + v,
        lm + h * inner + rm,
        v + " Objects:".ljust(inner) + v,
    ]

    obj_lines: list[str] = []
    for obj in objects_here[:2]:
        mark = "[x]" if obj.id in interacted_ids else "[ ]"
        state = object_states.get(obj.id, obj.state)
        label = f"  {mark} {obj.id} ({state})"[: inner - 1]
        obj_lines.append(v + f" {label}".ljust(inner) + v)

    if not obj_lines:
        obj_lines.append(v + "  (empty)".ljust(inner) + v)

    while len(obj_lines) < CELL_H - 5:
        obj_lines.append(v + " " * inner + v)

    lines.extend(obj_lines)
    lines.append(bl + h * inner + br)

    assert len(lines) == CELL_H, f"Cell height mismatch: {len(lines)} != {CELL_H}"
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

    canvas_w = (max_col + 1) * CELL_W + max_col * H_GAP
    canvas_h = (max_row + 1) * CELL_H + max_row * V_GAP
    canvas = _make_canvas(canvas_w, canvas_h)

    for rid, (col, row) in grid.items():
        room = id_to_room.get(rid)
        if room is None:
            continue
        cell_x = col * (CELL_W + H_GAP)
        cell_y = row * (CELL_H + V_GAP)
        marker = party_label if rid == party_room else ""
        _blit(
            canvas,
            _build_cell(
                room,
                objects_by_room.get(rid, []),
                marker,
                interacted_ids=interacted_ids,
                object_states=object_states,
            ),
            cell_x,
            cell_y,
        )

    for rid, (col, row) in grid.items():
        room = id_to_room.get(rid)
        if room is None:
            continue
        cell_x = col * (CELL_W + H_GAP)
        cell_y = row * (CELL_H + V_GAP)

        if "east" in room.adjacency and room.adjacency["east"] in grid:
            neighbor_col, _ = grid[room.adjacency["east"]]
            if neighbor_col == col + 1:
                corridor_y = cell_y + 1
                x_start = cell_x + CELL_W
                x_end = cell_x + CELL_W + H_GAP
                _draw_h_corridor(canvas, x_start, x_end, corridor_y)

        if "south" in room.adjacency and room.adjacency["south"] in grid:
            _, neighbor_row = grid[room.adjacency["south"]]
            if neighbor_row == row + 1:
                corridor_x = cell_x + CELL_W // 2
                y_start = cell_y + CELL_H
                y_end = cell_y + CELL_H + V_GAP
                _draw_v_corridor(canvas, corridor_x, y_start, y_end)

    for row_chars in canvas:
        print("".join(row_chars).rstrip())
