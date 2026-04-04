import random
from typing import Callable, Iterable, List, Optional, Tuple


def get_int_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return str(rng.randint(0, 100000))


def get_str_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    token = rng.choice(list(all_tokens))[1]
    return f"\"{token}\""


def get_bool_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return "True" if rng.choice([True, False]) else "False"


def get_float_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    return f"{rng.uniform(-1000.0, 1000.0):.6f}"


def get_list_int_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    length = rng.randint(0, 7)
    vals  = [get_int_literal(all_tokens, rng=rng) for _ in range(length)]
    return "[" + ", ".join(str(v) for v in vals) + "]"


def get_list_str_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    length = rng.randint(0, 7)
    vals = [get_str_literal(all_tokens, rng=rng) for _ in range(length)]
    return "[" + ", ".join(vals) + "]"


def get_list_bool_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    length = rng.randint(0, 7)
    vals = [get_bool_literal(all_tokens, rng=rng) for _ in range(length)]
    return "[" + ", ".join(vals) + "]"


def get_list_float_literal(all_tokens: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    length = rng.randint(0, 7)
    vals = [get_float_literal(all_tokens, rng=rng) for _ in range(length)]
    return "[" + ", ".join(vals) + "]"


LITERAL_GENERATORS: dict[str, Callable[..., str]] = {
    "int": get_int_literal,
    "str": get_str_literal,
    "bool": get_bool_literal,
    "float": get_float_literal,
    "list[int]": get_list_int_literal,
    "list[str]": get_list_str_literal,
    "list[bool]": get_list_bool_literal,
    "list[float]": get_list_float_literal,
}


def get_literal_of_type(typ: str, all_tkns_non_semantics: Iterable[Tuple[int, str]], rng: Optional[random.Random] = None) -> str:
    return LITERAL_GENERATORS[typ](all_tkns_non_semantics, rng=rng)
