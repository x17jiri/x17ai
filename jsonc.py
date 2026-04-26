import json
from pathlib import Path
from typing import Any


def _strip_json_comments(text: str) -> str:
	result: list[str] = []
	in_string = False
	escaped = False
	in_line_comment = False
	in_block_comment = False
	i = 0
	while i < len(text):
		ch = text[i]
		next_ch = text[i + 1] if i + 1 < len(text) else ""

		if in_line_comment:
			if ch == "\n":
				in_line_comment = False
				result.append(ch)
			i += 1
			continue

		if in_block_comment:
			if ch == "*" and next_ch == "/":
				in_block_comment = False
				i += 2
				continue
			if ch == "\n":
				result.append(ch)
			i += 1
			continue

		if in_string:
			result.append(ch)
			if escaped:
				escaped = False
			elif ch == "\\":
				escaped = True
			elif ch == '"':
				in_string = False
			i += 1
			continue

		if ch == '"':
			in_string = True
			result.append(ch)
			i += 1
			continue

		if ch == "/" and next_ch == "/":
			in_line_comment = True
			i += 2
			continue

		if ch == "/" and next_ch == "*":
			in_block_comment = True
			i += 2
			continue

		result.append(ch)
		i += 1

	return "".join(result)


def _strip_trailing_commas(text: str) -> str:
	result: list[str] = []
	in_string = False
	escaped = False
	i = 0
	while i < len(text):
		ch = text[i]

		if in_string:
			result.append(ch)
			if escaped:
				escaped = False
			elif ch == "\\":
				escaped = True
			elif ch == '"':
				in_string = False
			i += 1
			continue

		if ch == '"':
			in_string = True
			result.append(ch)
			i += 1
			continue

		if ch == ",":
			j = i + 1
			while j < len(text) and text[j].isspace():
				j += 1
			if j < len(text) and text[j] in "}]":
				i += 1
				continue

		result.append(ch)
		i += 1

	return "".join(result)


def loads_jsonc(text: str) -> Any:
	return json.loads(_strip_trailing_commas(_strip_json_comments(text)))


def load_jsonc(path: str | Path) -> Any:
	return loads_jsonc(Path(path).read_text(encoding="utf-8"))
