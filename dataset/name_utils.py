import json
import keyword
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
TOKENS_DIR = HERE / "outV2" / "tkns"

_TOKENS_CACHE: Optional[List[Tuple[int, str]]] = None
_TOKENS_SEM_CACHE: Optional[List[Tuple[int, str]]] = None
_TOKENS_ALL_CACHE: Optional[List[str]] = None
_CACHE_MODEL: Optional[str] = None

MODEL_NAME = "bigcode/santacoder"
MAX_TOKEN_LEN = 12


def _model_name_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name)


def _tokens_paths(model_name: str) -> tuple[Path, Path]:
    slug = _model_name_slug(model_name)
    return (
        TOKENS_DIR / f"tokens_{slug}.json",
        TOKENS_DIR / f"tokens_sem_{slug}.json",
    )

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while", "record", "var", "sealed",
    "permits", "non-sealed", "sealed",
    "true", "false", "null",
}

# Common Java library/class identifiers that appear in your function bank bodies.
JAVA_BUILTINS = {
    "Math", "Integer", "Float", "String", "StringBuilder", "Character",
    "List", "ArrayList", "Arrays", "Collections",
}

JS_KEYWORDS = {
    "await", "break", "case", "catch", "class", "const", "continue", "debugger",
    "default", "delete", "do", "else", "enum", "export", "extends", "false",
    "finally", "for", "function", "if", "import", "in", "instanceof", "new",
    "null", "return", "super", "switch", "this", "throw", "true", "try",
    "typeof", "var", "void", "while", "with", "yield", "let", "async",
}

BANNED_SUBSTRINGS = {
    "int", "str", "list", "string", "integer", "float",
    "double", "char", "text", "num", "dict", "set", "tuple",
    "bool", "none", "true", "false", "array",
    "long", "short", "byte", "boolean", "object",
    "map", "hashmap", "hashset", "arraylist", "linkedlist",
}

BANNED_EXACT = set(keyword.kwlist) | JAVA_KEYWORDS | JS_KEYWORDS | BANNED_SUBSTRINGS


def _type_name_tokens() -> List[str]:
    try:
        from metadata import get_type_list
    except Exception:
        return ["int", "str", "bool", "list"]
    names = set()
    for t in get_type_list(False):
        if not t:
            continue
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", t):
            names.add(name)
    return sorted(names)


def _filter_to_type_names(tokens: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    type_names = set(_type_name_tokens())
    return [(tid, s) for tid, s in tokens if s in type_names]


def is_identifier_safe(s: str) -> bool:
    if not s:
        return False
    if s.strip() != s:
        return False
    if any(c.isspace() for c in s):
        return False
    if not IDENT_RE.match(s):
        return False
    if len(s) > MAX_TOKEN_LEN:
        return False
    return True


def is_semantically_neutral(s: str) -> bool:
    low = s.lower()
    if low in BANNED_EXACT:
        return False
    for bad in BANNED_SUBSTRINGS:
        if bad in low:
            return False
    return True


def is_single_token(tok, s: str) -> bool:
    return len(tok.encode(s, add_special_tokens=False)) == 1


def collect_identifier_tokens(model_name: str = MODEL_NAME, tok=None) -> List[Tuple[int, str]]:
    tokens_path, _ = _tokens_paths(model_name)
    if tokens_path.exists():
        with open(tokens_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [tuple(x) for x in data]

    if tok is None:
        raise RuntimeError("Tokenizer required to collect tokens when cache files are missing.")

    results: List[Tuple[int, str]] = []
    for tid in tqdm(range(tok.vocab_size), desc="Collect neutral tokens", unit="tok"):
        s = tok.decode([tid])

        if not is_identifier_safe(s):
            continue
        if not is_semantically_neutral(s):
            continue
        if not is_single_token(tok, s):
            continue

        results.append((tid, s))

    return results


def collect_semantic_tokens(model_name: str = MODEL_NAME, tok=None) -> List[Tuple[int, str]]:
    _, tokens_sem_path = _tokens_paths(model_name)
    if tokens_sem_path.exists():
        with open(tokens_sem_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _filter_to_type_names([tuple(x) for x in data])

    if tok is None:
        raise RuntimeError("Tokenizer required to collect tokens when cache files are missing.")

    results: List[Tuple[int, str]] = []
    for name in tqdm(_type_name_tokens(), desc="Collect semantic tokens", unit="tok"):
        tid_list = tok.encode(name, add_special_tokens=False)
        if len(tid_list) != 1:
            continue
        tid = tid_list[0]
        s = tok.decode([tid])
        if s != name:
            continue
        if not is_identifier_safe(s):
            continue
        if not is_single_token(tok, s):
            continue
        results.append((tid, s))

    return results


def _release_model(model) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    del model


def write_token_files(model_name: str = MODEL_NAME) -> None:
    tokens_path, tokens_sem_path = _tokens_paths(model_name)
    if tokens_path.exists() and tokens_sem_path.exists():
        print("Token files already exist - skipping.")
        return
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(model_name, trust_remote_code=True)
    tok = model.tokenizer
    neutral = collect_identifier_tokens(model_name=model_name, tok=tok)
    semantic = collect_semantic_tokens(model_name=model_name, tok=tok)
    _release_model(model)

    with open(tokens_path, "w", encoding="utf-8") as f:
        json.dump(neutral, f, ensure_ascii=False, indent=2)
    with open(tokens_sem_path, "w", encoding="utf-8") as f:
        json.dump(semantic, f, ensure_ascii=False, indent=2)


def get_identifier_tokens(model_name: str = MODEL_NAME) -> List[Tuple[int, str]]:
    global _TOKENS_CACHE, _CACHE_MODEL
    if _TOKENS_CACHE is None or _CACHE_MODEL != model_name:
        raise RuntimeError("name_utils not initialized; call init_name_utils() first.")
    return _TOKENS_CACHE


def get_semantic_tokens(model_name: str = MODEL_NAME) -> List[Tuple[int, str]]:
    global _TOKENS_SEM_CACHE, _CACHE_MODEL
    if _TOKENS_SEM_CACHE is None or _CACHE_MODEL != model_name:
        raise RuntimeError("name_utils not initialized; call init_name_utils() first.")
    return _TOKENS_SEM_CACHE


def init_name_utils(model_name: str = MODEL_NAME) -> None:
    print(f"Initializing name_utils for model {model_name}...")
    global _TOKENS_CACHE, _TOKENS_SEM_CACHE, _CACHE_MODEL
    if _CACHE_MODEL == model_name and _TOKENS_CACHE is not None and _TOKENS_SEM_CACHE is not None:
        print("name_utils already initialized.")
        return
    if not (_tokens_paths(model_name)[0].exists() and _tokens_paths(model_name)[1].exists()):
        print("Token files not found; writing them now...")
        write_token_files(model_name=model_name)
    _TOKENS_CACHE = collect_identifier_tokens(model_name=model_name)
    _TOKENS_SEM_CACHE = collect_semantic_tokens(model_name=model_name)
    _CACHE_MODEL = model_name
    print(f"name_utils initialized for model {model_name}.")


def random_neutral_identifier_name(
    model_name: str = MODEL_NAME,
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random
    tokens = get_identifier_tokens(model_name=model_name)
    return rng.choice(tokens)[1]


def random_semantic_identifier_name(
    model_name: str = MODEL_NAME,
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random
    tokens = get_semantic_tokens(model_name=model_name)
    return rng.choice(tokens)[1]


def random_semantic_disagree_identifier_name(
    type_t: str,
    model_name: str = MODEL_NAME,
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random
    tokens = get_semantic_tokens(model_name=model_name)
    candidates = [token for token in tokens if token[1] != type_t]
    if not candidates:
        raise RuntimeError("No semantic mismatch available; only one semantic token.")
    return rng.choice(candidates)[1]
