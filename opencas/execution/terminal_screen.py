"""Small terminal screen renderer for PTY snapshots."""

from __future__ import annotations

from typing import Any


def render_terminal_screen(text: str, *, rows: int = 24, cols: int = 80) -> dict[str, Any]:
    """Render common VT control sequences into a visible terminal buffer."""
    rows = max(1, int(rows or 24))
    cols = max(1, int(cols or 80))
    grid = [[" " for _ in range(cols)] for _ in range(rows)]
    row = 0
    col = 0
    saved_row = 0
    saved_col = 0
    i = 0

    def clamp_cursor() -> None:
        nonlocal row, col
        row = min(max(row, 0), rows - 1)
        col = min(max(col, 0), cols - 1)

    def scroll_if_needed() -> None:
        nonlocal row
        while row >= rows:
            grid.pop(0)
            grid.append([" " for _ in range(cols)])
            row -= 1

    def write_char(char: str) -> None:
        nonlocal row, col
        grid[row][col] = char
        col += 1
        if col >= cols:
            col = 0
            row += 1
            scroll_if_needed()

    def clear_screen(mode: int) -> None:
        if mode in (2, 3):
            for r in range(rows):
                grid[r] = [" " for _ in range(cols)]
        elif mode == 1:
            for r in range(0, row):
                grid[r] = [" " for _ in range(cols)]
            grid[row][: col + 1] = [" " for _ in range(col + 1)]
        else:
            grid[row][col:] = [" " for _ in range(cols - col)]
            for r in range(row + 1, rows):
                grid[r] = [" " for _ in range(cols)]

    def clear_line(mode: int) -> None:
        if mode == 2:
            grid[row] = [" " for _ in range(cols)]
        elif mode == 1:
            grid[row][: col + 1] = [" " for _ in range(col + 1)]
        else:
            grid[row][col:] = [" " for _ in range(cols - col)]

    def parse_csi_params(raw_params: str) -> list[int]:
        raw_params = raw_params.replace("?", "").replace(">", "").replace("<", "").replace("=", "")
        if not raw_params:
            return []
        params: list[int] = []
        for item in raw_params.split(";"):
            if item == "":
                params.append(0)
                continue
            try:
                params.append(int(item))
            except ValueError:
                params.append(0)
        return params

    def param(params: list[int], index: int, default: int) -> int:
        if index >= len(params) or params[index] == 0:
            return default
        return params[index]

    def apply_csi(raw_params: str, command: str) -> None:
        nonlocal row, col, saved_row, saved_col
        params = parse_csi_params(raw_params)
        if command in ("H", "f"):
            row = param(params, 0, 1) - 1
            col = param(params, 1, 1) - 1
            clamp_cursor()
        elif command == "J":
            clear_screen(param(params, 0, 0))
        elif command == "K":
            clear_line(param(params, 0, 0))
        elif command == "A":
            row -= param(params, 0, 1)
            clamp_cursor()
        elif command == "B":
            row += param(params, 0, 1)
            clamp_cursor()
        elif command == "C":
            col += param(params, 0, 1)
            clamp_cursor()
        elif command == "D":
            col -= param(params, 0, 1)
            clamp_cursor()
        elif command == "E":
            row += param(params, 0, 1)
            col = 0
            clamp_cursor()
        elif command == "F":
            row -= param(params, 0, 1)
            col = 0
            clamp_cursor()
        elif command in ("G", "`"):
            col = param(params, 0, 1) - 1
            clamp_cursor()
        elif command == "d":
            row = param(params, 0, 1) - 1
            clamp_cursor()
        elif command == "s":
            saved_row, saved_col = row, col
        elif command == "u":
            row, col = saved_row, saved_col
            clamp_cursor()

    length = len(text)
    while i < length:
        char = text[i]
        if char == "\x1b":
            if i + 1 >= length:
                break
            next_char = text[i + 1]
            if next_char == "[":
                final_index = i + 2
                while final_index < length and not (
                    "@" <= text[final_index] <= "~"
                ):
                    final_index += 1
                if final_index >= length:
                    break
                apply_csi(text[i + 2 : final_index], text[final_index])
                i = final_index + 1
                continue
            if next_char == "]":
                i = _skip_osc(text, i + 2)
                continue
            if next_char == "P":
                i = _skip_until_string_terminator(text, i + 2)
                continue
            if next_char in ("(", ")", "*", "+", "-", ".") and i + 2 < length:
                i += 3
                continue
            i += 2
            continue
        if char == "\r":
            col = 0
        elif char == "\n":
            row += 1
            col = 0
            scroll_if_needed()
        elif char == "\b":
            col = max(0, col - 1)
        elif char == "\t":
            target = min(cols, ((col // 8) + 1) * 8)
            while col < target:
                write_char(" ")
        elif char >= " " and char != "\x7f":
            write_char(char)
        i += 1

    lines = ["".join(line) for line in grid]
    trimmed = [line.rstrip() for line in lines]
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    return {
        "rows": rows,
        "cols": cols,
        "cursor": {"row": row + 1, "col": col + 1},
        "lines": lines,
        "text": "\n".join(trimmed),
        "non_empty_line_count": sum(1 for line in lines if line.strip()),
    }


def _skip_osc(text: str, index: int) -> int:
    length = len(text)
    while index < length:
        if text[index] == "\x07":
            return index + 1
        if text[index] == "\x1b" and index + 1 < length and text[index + 1] == "\\":
            return index + 2
        index += 1
    return length


def _skip_until_string_terminator(text: str, index: int) -> int:
    length = len(text)
    while index < length:
        if text[index] == "\x1b" and index + 1 < length and text[index + 1] == "\\":
            return index + 2
        index += 1
    return length
