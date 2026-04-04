
# ---------------- signature lists (kept separate) ----------------

# ================= int -> int =================

INT_INT_FUNCS = [
    lambda x: f'return {x} + 1',
    lambda x: f'return {x} - 1',
    lambda x: f'out = 1\nfor i in range(1, {x} + 1):\n    out *= i\nreturn out',
    lambda x: f'count = 0\nn = {x}\nwhile n:\n    count += n & 1\n    n >>= 1\nreturn count',
    lambda x: f'a, b = abs({x}), 42\nwhile b:\n    a, b = b, a % b\nreturn a',
    lambda x: f'return {x} * ({x} + 1) // 2',
    lambda x: f'return 1 if {x} % 2 == 0 else 0',
    lambda x: f'sign = -1 if {x} < 0 else 1\ns = str(abs({x}))\nreturn sign * int(s[::-1])',
    lambda x: f'bits = 0\nn = {x}\nwhile n > 0:\n    bits += 1\n    n >>= 1\nreturn bits',
    lambda x: f'return {x} % 7',
]

# ================= int -> str =================

INT_STR_FUNCS = [
    lambda x: f"return format({x}, 'b')",
    lambda x: f"return format({x}, 'x')",
    lambda x: f"return 'even' if {x} % 2 == 0 else 'odd'",
    lambda x: f"return 'positive' if {x} > 0 else 'negative' if {x} < 0 else 'zero'",
    lambda x: f"vals = [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]\nout = ''\nfor v, s in vals:\n    while {x} >= v:\n        out += s\n        {x} -= v\nreturn out",
    lambda x: f"suffix = 'th'\nif {x} % 100 < 11 or {x} % 100 > 13:\n    suffix = {{1:'st',2:'nd',3:'rd'}}.get({x} % 10, 'th')\nreturn str({x}) + suffix",
    lambda x: f'return str(len(str(abs({x}))))',
    lambda x: f"return format({x} & 0xff, '08b')",
    lambda x: f"if {x} < 2:\n    return 'not prime'\nfor i in range(2, int({x} ** 0.5) + 1):\n    if {x} % i == 0:\n        return 'composite'\nreturn 'prime'",
    lambda x: f"h = ({x} // 60) % 24\nm = {x} % 60\nreturn f'{{h:02d}}:{{m:02d}}'",
]

# ================= str -> str =================

STR_STR_FUNCS = [
    lambda x: f'return {x}.strip()',
    lambda x: f'return {x}.lower()',
    lambda x: f'return {x}.upper()',
    lambda x: f'return {x}[::-1]',
    lambda x: f"return {x}.replace(' ', '_')",
    lambda x: f"return ''.join(w.capitalize() for w in {x}.split())",
    lambda x: f'return {x} + {x}',
    lambda x: f'return {x}[:len({x})//2]',
    lambda x: f'return {x}.title()',
    lambda x: f"return ''.join(c for c in {x} if c.isalpha())",
]

# ================= str -> int =================

STR_INT_FUNCS = [
    lambda x: f'return len({x})',
    lambda x: f"return {x}.count('a')",
    lambda x: f'return sum(1 for c in {x} if c.isupper())',
    lambda x: f'return sum(1 for c in {x} if c.isdigit())',
    lambda x: f"return {x}.find('e')",
    lambda x: f"return sum(1 for c in {x} if c.lower() in 'aeiou')",
    lambda x: f'return len({x}.split())',
    lambda x: f'return sum(ord(c) for c in {x})',
    lambda x: f"return {x}.count('\\n')",
    lambda x: f'return 1 if {x}.isidentifier() else 0',
]

# ================= int -> bool =================

INT_BOOL_FUNCS = [
    lambda x: f'return {x} % 2 == 0',
    lambda x: f'return {x} > 0',
]

# ================= int -> list[int] =================

INT_LIST_INT_FUNCS = [
    lambda x: f'return [{x}]',
    lambda x: f'return [{x}, {x}]',
    lambda x: f'return list(range({x}))',
    lambda x: f'return [i * i for i in range({x})]',
    lambda x: f'return [{x} - 1, {x}, {x} + 1]',
]

# ================= int -> list[str] =================

INT_LIST_STR_FUNCS = [
    lambda x: f'return [str({x})]',
    lambda x: f"return [format({x}, 'b'), format({x}, 'x')]",
    lambda x: f'return list(str(abs({x})))',
    lambda x: f"return ['even' if {x} % 2 == 0 else 'odd', 'positive' if {x} > 0 else 'negative' if {x} < 0 else 'zero']",
    lambda x: f'return [str(i) for i in range(max(0, {x}))]',
]

# ================= int -> list[bool] =================

INT_LIST_BOOL_FUNCS = [
    lambda x: f'return [{x} % 2 == 0]',
    lambda x: f'return [{x} > 0, {x} < 0, {x} == 0]',
    lambda x: f'return [i % 2 == 0 for i in range(max(0, {x}))]',
    lambda x: f'n = {x}\nout = []\nwhile n > 0:\n    out.append((n & 1) == 1)\n    n >>= 1\nreturn out',
    lambda x: f'return [False] * max(0, {x})',
]


# ================= str -> bool =================

STR_BOOL_FUNCS = [
    lambda x: f'return len({x}) == 0',
    lambda x: f'return {x}.isalpha()',
]

# ================= bool -> int =================

BOOL_INT_FUNCS = [
    lambda x: f'return 1 if {x} else 0',
    lambda x: f'return 0 if {x} else 1',
]

# ================= bool -> bool =================

BOOL_BOOL_FUNCS = [
    lambda x: f'return False if {x} else True',
    lambda x: f'return True if {x} else False',
    lambda x: f'return {x}',
    lambda x: f'return not {x}',
]


# ================= bool -> str =================

BOOL_STR_FUNCS = [
    lambda x: f"return '' if {x} else 'not_empty'",
    lambda x: f"return 'not_empty' if {x} else ''",
    lambda x: f"return 'false' if {x} else 'true'",
    lambda x: f"return 'true' if {x} else 'false'",
    lambda x: f"return '0' if {x} else '1'",
    lambda x: f"return '1' if {x} else '0'",
]

# ---------------- boolean -> list[int] ----------------

BOOL_LIST_INT_FUNCS = [
    lambda x: f'return [1] if {x} else []',
    lambda x: f'return [] if {x} else [1]',
]

# ---------------- boolean -> list[str] ----------------

BOOL_LIST_STR_FUNCS = [
    lambda x: f"return [''] if {x} else []",
    lambda x: f"return [] if {x} else ['not_empty']",
    lambda x: f"return ['false'] if {x} else []",
    lambda x: f"return ['true'] if {x} else []",
    lambda x: f"return ['0'] if {x} else []",
    lambda x: f"return ['1'] if {x} else []",
]

# ---------------- list[int] -> int ----------------

LIST_INT_INT_FUNCS = [
    lambda x: f'return sum({x})',
    lambda x: f'out = 0\nfor v in {x}:\n    out += v * v\nreturn out',
    lambda x: f'return max({x}) if {x} else 0',
    lambda x: f'return min({x}) if {x} else 0',
    lambda x: f'return len({x})',
]

# ---------------- list[int] -> list[int] ----------------

LIST_INT_LIST_INT_FUNCS = [
    lambda x: f'return [v + 1 for v in {x}]',
    lambda x: f'return [v * 2 for v in {x}]',
    lambda x: f'return [v for v in {x} if v % 2 == 0]',
    lambda x: f'return {x}[::-1]',
    lambda x: f'return sorted({x})',
]

# ---------------- list[int] -> bool ----------------

LIST_INT_BOOL_FUNCS = [
    lambda x: f'return len({x}) == 0',
    lambda x: f'return all(v >= 0 for v in {x})',
]

# ---------------- list[str] -> int ----------------

LIST_STR_INT_FUNCS = [
    lambda x: f'return len({x})',
    lambda x: f'return sum(len(s) for s in {x})',
    lambda x: f'return sum(1 if s else 0 for s in {x})',
]


# ---------------- list[str] -> str ----------------

LIST_STR_STR_FUNCS = [
    lambda x: f"return ''.join({x})",
    lambda x: f"return ' '.join({x})",
    lambda x: f"return ''.join(s[0] for s in {x} if s)",
]


# ================= list[str] -> list[int] =================

LIST_STR_LIST_INT_FUNCS = [
    lambda x: f'return [len(s) for s in {x}]',
    lambda x: f'return [sum(c.isdigit() for c in s) for s in {x}]',
    lambda x: f'return [sum(c.isupper() for c in s) for s in {x}]',
    lambda x: f"return [s.count('a') for s in {x}]",
    lambda x: f'return [len(s.split()) for s in {x}]',
]

# ---------------- list[int] -> list[bool] ----------------

LIST_INT_LIST_BOOL_FUNCS = [
    lambda x: f'return [v % 2 == 0 for v in {x}]',
    lambda x: f'return [v > 0 for v in {x}]',
]

# ---------------- list[str] -> list[bool] ----------------

LIST_STR_LIST_BOOL_FUNCS = [
    lambda x: f'return [len(s) > 0 for s in {x}]',
    lambda x: f'return [s.isalpha() for s in {x} if s]',
]

# ---------------- list[bool] -> list[int] ----------------

LIST_BOOL_LIST_INT_FUNCS = [
    lambda x: f'return [1 if v else 0 for v in {x}]',
    lambda x: f'return [0 if v else 1 for v in {x}]',
]

# ---------------- list[bool] -> list[str] ----------------

LIST_BOOL_LIST_STR_FUNCS = [
    lambda x: f"return ['1' if v else '0' for v in {x}]",
    lambda x: f"return ['true' if v else 'false' for v in {x}]",
    lambda x: f"return ['' if v else 'not_empty' for v in {x}]",
]

# ================= str -> list[int] =================

STR_LIST_INT_FUNCS = [
    lambda x: f'return [len({x})]',
    lambda x: f'return [ord(c) for c in {x}]',
    lambda x: f'return [i for i, c in enumerate({x}) if c.isdigit()]',
    lambda x: f'return [len(w) for w in {x}.split()]',
    lambda x: f"return [{x}.count(c) for c in 'aeiou']",
]

# ================= str -> list[str] =================

STR_LIST_STR_FUNCS = [
    lambda x: f'return {x}.split()',
    lambda x: f'return list({x})',
    lambda x: f"return {x}.split(',') if ',' in {x} else [{x}]",
    lambda x: f'return [w.lower() for w in {x}.split()]',
    lambda x: f'return [{x}[:i] for i in range(len({x}) + 1)]',
]

# ================= str -> list[bool] =================

STR_LIST_BOOL_FUNCS = [
    lambda x: f'return [c.isalpha() for c in {x}]',
    lambda x: f'return [c.isdigit() for c in {x}]',
    lambda x: f'return [c.isupper() for c in {x}]',
    lambda x: f'return [len(w) > 0 for w in {x}.split()]',
    lambda x: f"return [c in 'aeiouAEIOU' for c in {x}]",
]

# ================= bool -> list[bool] =================

BOOL_LIST_BOOL_FUNCS = [
    lambda x: f'return [{x}]',
    lambda x: f'return [not {x}]',
    lambda x: f'return [{x}, not {x}]',
    lambda x: f'return [] if {x} else [False]',
    lambda x: f'return [True] if {x} else [False]',
]

# ================= list[int] -> str =================

LIST_INT_STR_FUNCS = [
    lambda x: f"return ','.join(str(v) for v in {x})",
    lambda x: f"return '' if not {x} else str(sum({x}))",
    lambda x: f"return '' if not {x} else str(max({x}))",
    lambda x: f"return ''.join(str(v % 10) for v in {x})",
    lambda x: f"return ' '.join(str(v) for v in {x})",
]

# ================= list[int] -> list[str] =================

LIST_INT_LIST_STR_FUNCS = [
    lambda x: f'return [str(v) for v in {x}]',
    lambda x: f"return [format(v, 'b') for v in {x}]",
    lambda x: f"return [format(v, 'x') for v in {x}]",
    lambda x: f"return ['even' if v % 2 == 0 else 'odd' for v in {x}]",
    lambda x: f'return [str(abs(v)) for v in {x}]',
]

# ================= list[str] -> bool =================

LIST_STR_BOOL_FUNCS = [
    lambda x: f'return len({x}) == 0',
    lambda x: f'return all(bool(s) for s in {x})',
    lambda x: f"return any('a' in s for s in {x})",
    lambda x: f'return any(s.isdigit() for s in {x} if s)',
    lambda x: f'return any(s.isupper() for s in {x} if s)',
]

# ================= list[str] -> list[str] =================

LIST_STR_LIST_STR_FUNCS = [
    lambda x: f'return [s.strip() for s in {x}]',
    lambda x: f'return [s.lower() for s in {x}]',
    lambda x: f'return [s.upper() for s in {x}]',
    lambda x: f'return [s[::-1] for s in {x}]',
    lambda x: f'return sorted({x})',
]

# ================= list[bool] -> int =================

LIST_BOOL_INT_FUNCS = [
    lambda x: f'return sum(1 for v in {x} if v)',
    lambda x: f'return len({x})',
    lambda x: f'return 0 if not {x} else (1 if all({x}) else 0)',
    lambda x: f'return 0 if not {x} else (1 if any({x}) else 0)',
    lambda x: f'return sum(int(v) for v in {x})',
]

# ================= list[bool] -> bool =================

LIST_BOOL_BOOL_FUNCS = [
    lambda x: f'return all({x})',
    lambda x: f'return any({x})',
    lambda x: f'return len({x}) == 0',
    lambda x: f'return False if not {x} else {x}[0]',
    lambda x: f'return False if not {x} else {x}[-1]',
]

# ================= list[bool] -> str =================

LIST_BOOL_STR_FUNCS = [
    lambda x: f"return ','.join('1' if v else '0' for v in {x})",
    lambda x: f"return ' '.join('true' if v else 'false' for v in {x})",
    lambda x: f"return '' if not {x} else ('true' if {x}[0] else 'false')",
    lambda x: f"return '0' if not {x} else str(len({x}))",
    lambda x: f'return str(sum(1 for v in {x} if v))',
]

# ================= list[bool] -> list[bool] =================

LIST_BOOL_LIST_BOOL_FUNCS = [
    lambda x: f'return [not v for v in {x}]',
    lambda x: f'return {x}[::-1]',
    lambda x: f'return [v for v in {x} if v]',
    lambda x: f'return [v for v in {x} if not v]',
    lambda x: f'return sorted({x})',
]

# ================= float -> float =================

FLOAT_FLOAT_FUNCS = [
    lambda x: f'return {x} + 0.5',
    lambda x: f'return {x} * 1.1',
    lambda x: f'return -{x}',
    lambda x: f'return abs({x})',
    lambda x: f'return {x} / 2.0',
]

# ================= float -> int =================

FLOAT_INT_FUNCS = [
    lambda x: f'return int({x})',
    lambda x: f'return int(abs({x}))',
    lambda x: f'return int(round({x}))',
]

# ================= float -> str =================

FLOAT_STR_FUNCS = [
    lambda x: f'return str({x})',
    lambda x: f'return f"{{{x}:.2f}}"',
]

# ================= float -> bool =================

FLOAT_BOOL_FUNCS = [
    lambda x: f'return {x} > 0.0',
    lambda x: f'return {x} == 0.0',
]

# ================= float -> list[int] =================

FLOAT_LIST_INT_FUNCS = [
    lambda x: f'return [int({x})]',
    lambda x: f'return [int({x}), int({x}) + 1]',
]

# ================= float -> list[str] =================

FLOAT_LIST_STR_FUNCS = [
    lambda x: f'return [str({x})]',
    lambda x: f'return [f"{{{x}:.1f}}", f"{{{x}:.3f}}"]',
]

# ================= float -> list[bool] =================

FLOAT_LIST_BOOL_FUNCS = [
    lambda x: f'return [{x} > 0.0]',
    lambda x: f'return [{x} == 0.0, {x} != 0.0]',
]

# ================= float -> list[float] =================

FLOAT_LIST_FLOAT_FUNCS = [
    lambda x: f'return [{x}]',
    lambda x: f'return [{x}, {x} / 2.0, {x} * 2.0]',
]

# ================= int -> float =================

INT_FLOAT_FUNCS = [
    lambda x: f'return float({x})',
    lambda x: f'return {x} / 2.0',
]

# ================= str -> float =================

STR_FLOAT_FUNCS = [
    lambda x: f'return float(len({x}))',
    lambda x: f'return float(sum(ord(c) for c in {x}))',
]

# ================= bool -> float =================

BOOL_FLOAT_FUNCS = [
    lambda x: f'return 1.0 if {x} else 0.0',
    lambda x: f'return 0.0 if {x} else 1.0',
]

# ================= list[int] -> float =================

LIST_INT_FLOAT_FUNCS = [
    lambda x: f'return 0.0 if not {x} else sum({x}) / len({x})',
    lambda x: f'return float(sum({x}))',
]

# ================= list[str] -> float =================

LIST_STR_FLOAT_FUNCS = [
    lambda x: f'return 0.0 if not {x} else sum(len(s) for s in {x}) / len({x})',
    lambda x: f'return float(sum(len(s) for s in {x}))',
]

# ================= list[bool] -> float =================

LIST_BOOL_FLOAT_FUNCS = [
    lambda x: f'return 0.0 if not {x} else sum(1 for v in {x} if v) / len({x})',
    lambda x: f'return float(sum(1 for v in {x} if v))',
]

# ================= list[float] -> float =================

LIST_FLOAT_FLOAT_FUNCS = [
    lambda x: f'return 0.0 if not {x} else sum({x}) / len({x})',
    lambda x: f'return float(sum({x}))',
    lambda x: f'return max({x}) if {x} else 0.0',
]

# ================= int -> list[float] =================

INT_LIST_FLOAT_FUNCS = [
    lambda x: f'return [float({x})]',
    lambda x: f'return [float(i) + 0.5 for i in range(max(0, {x}))]',
]

# ================= str -> list[float] =================

STR_LIST_FLOAT_FUNCS = [
    lambda x: f'return [float(len({x}))]',
    lambda x: f'return [float(ord(c)) / 10.0 for c in {x}]',
]

# ================= bool -> list[float] =================

BOOL_LIST_FLOAT_FUNCS = [
    lambda x: f'return [1.0 if {x} else 0.0]',
    lambda x: f'return [0.0, 1.0] if {x} else [0.0]',
]

# ================= list[int] -> list[float] =================

LIST_INT_LIST_FLOAT_FUNCS = [
    lambda x: f'return [float(v) for v in {x}]',
    lambda x: f'return [v / 2.0 for v in {x}]',
]

# ================= list[str] -> list[float] =================

LIST_STR_LIST_FLOAT_FUNCS = [
    lambda x: f'return [float(len(s)) for s in {x}]',
    lambda x: f'return [float(sum(c.isdigit() for c in s)) for s in {x}]',
]

# ================= list[bool] -> list[float] =================

LIST_BOOL_LIST_FLOAT_FUNCS = [
    lambda x: f'return [1.0 if v else 0.0 for v in {x}]',
    lambda x: f'return [0.0 if v else 1.0 for v in {x}]',
]

# ================= list[float] -> int =================

LIST_FLOAT_INT_FUNCS = [
    lambda x: f'return len({x})',
    lambda x: f'return int(sum({x}))',
    lambda x: f'return 0 if not {x} else int(max({x}))',
]

# ================= list[float] -> str =================

LIST_FLOAT_STR_FUNCS = [
    lambda x: f'return \',\'.join(f"{{v:.2f}}" for v in {x})',
    lambda x: f'return \'\' if not {x} else f"{{sum({x}):.2f}}"',
]

# ================= list[float] -> bool =================

LIST_FLOAT_BOOL_FUNCS = [
    lambda x: f'return len({x}) == 0',
    lambda x: f'return all(v > 0.0 for v in {x})',
]

# ================= list[float] -> list[int] =================

LIST_FLOAT_LIST_INT_FUNCS = [
    lambda x: f'return [int(v) for v in {x}]',
    lambda x: f'return [int(abs(v)) for v in {x}]',
]

# ================= list[float] -> list[str] =================

LIST_FLOAT_LIST_STR_FUNCS = [
    lambda x: f'return [str(v) for v in {x}]',
    lambda x: f'return [f"{{v:.1f}}" for v in {x}]',
]

# ================= list[float] -> list[bool] =================

LIST_FLOAT_LIST_BOOL_FUNCS = [
    lambda x: f'return [v > 0.0 for v in {x}]',
    lambda x: f'return [v == 0.0 for v in {x}]',
]

# ================= list[float] -> list[float] =================

LIST_FLOAT_LIST_FLOAT_FUNCS = [
    lambda x: f'return [v * 2.0 for v in {x}]',
    lambda x: f'return [v / 2.0 for v in {x}]',
    lambda x: f'return sorted({x})',
]