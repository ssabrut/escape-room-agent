"""ASCII 2D grid renderer for room layouts."""

from __future__ import annotations

from state import Room

CELL_WIDTH = 30  # total cell width including border chars
COLS_PER_ROW = 3


def _build_cell(room: Room, width: int = CELL_WIDTH, use_unicode: bool = True) -> list[str]:
    inner = width - 2

    if use_unicode:
        tl, tr, bl, br = "┌", "┐", "└", "┘"
        h, v, lm, rm, tm, bm = "─", "│", "├", "┤", "┬", "┴"
        divider = lm + h * inner + rm
    else:
        tl, tr, bl, br = "+", "+", "+", "+"
        h, v = "-", "|"
        divider = "+" + h * inner + "+"

    lines: list[str] = []
    lines.append(tl + h * inner + tr)
    name = room.name.upper()[: inner - 1]
    lines.append(v + f" {name}".ljust(inner) + v)
    lines.append(divider)
    lines.append(v + " Items:".ljust(inner) + v)

    if room.items:
        for item in room.items:
            label = f"  [*] {item.name}"[: inner - 1]
            lines.append(v + f" {label}".ljust(inner) + v)
    else:
        lines.append(v + "  (empty)".ljust(inner) + v)

    lines.append(bl + h * inner + br)
    return lines


def _pad_cell(cell: list[str], target_height: int, width: int, use_unicode: bool = True) -> list[str]:
    v = "│" if use_unicode else "|"
    inner = width - 2
    blank = v + " " * inner + v
    while len(cell) < target_height:
        cell.insert(-1, blank)
    return cell


def render_room_layout(rooms: list[Room], cols: int = COLS_PER_ROW, use_unicode: bool = True) -> None:
    for row_start in range(0, len(rooms), cols):
        row_rooms = rooms[row_start : row_start + cols]
        cells = [_build_cell(r, use_unicode=use_unicode) for r in row_rooms]
        max_h = max(len(c) for c in cells)
        cells = [_pad_cell(c, max_h, CELL_WIDTH, use_unicode) for c in cells]
        for line_idx in range(max_h):
            print("  ".join(cell[line_idx] for cell in cells))
        print()
