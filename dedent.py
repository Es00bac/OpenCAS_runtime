from pathlib import Path

TARGET = Path(__file__).resolve().parent / "opencas" / "execution" / "baa.py"
content = TARGET.read_text(encoding="utf-8")
# We know lines 148+ have bad indentation
lines = content.split('\n')
new_lines = []
in_run_bounded = False
for i, line in enumerate(lines):
    if line.startswith("    async def _run_bounded("):
        in_run_bounded = True
        new_lines.append(line)
        continue
    if in_run_bounded:
        if line.startswith("    def "):
            in_run_bounded = False
            new_lines.append(line)
            continue
        if line.startswith("            "):
            new_lines.append(line[4:])
        elif line.startswith("        ") and i == 148:
            new_lines.append(line)
        elif line.startswith("        ") and i == 149:
            new_lines.append(line)
        elif line.startswith("                ") and i == 150:
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

TARGET.write_text('\n'.join(new_lines), encoding="utf-8")
