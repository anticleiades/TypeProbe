import ast

try:
    from dataset.name_utils import *
    import dataset.function_bank_python as python_bank
except ModuleNotFoundError:
    from name_utils import *
    import function_bank_python as python_bank

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable
import pyarrow as pa

prober_meta_null_type = "PROBER_NIHIL"

f1TargetClass = 0
f2TargetClass = 1
nullFunctionTargetClass = 2


def get_equiprobFunc_tgt_class(args) -> int:
    if args.nullClassTask0:
        return 3
    else:
        return 2

policyFixed = 0
policyRandomize = 1
policySemanticRandom = 2
policySemanticMismatch = 3
policySemanticAgree = 4

# Fixed, cross-language-safe identifier aliases for the *finite* type universe.
# Keep these <= 12 chars to match MAX_TOKEN_LEN in name_utils.

_TYPE_TO_IDENT = {
    "int": "int",
    "str": "str",
    "bool": "bool",
    "float": "float",
    "list[int]": "list_int",
    "list[str]": "list_str",
    "list[bool]": "list_bool",
    "list[float]": "list_float",
}

_IDENT_TO_TYPE = {v: k for k, v in _TYPE_TO_IDENT.items()}
_TYPE_LIST = list(_TYPE_TO_IDENT.keys())

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

_RESERVED_CACHE_PY: dict[tuple[int, int], set[str]] = {}
_RESERVED_CACHE_JAVA: dict[tuple[int, int], set[str]] = {}


def reserved_idents_in_py_body(func_list, idx: int, *, arg_name: str = "__ARG__") -> set[str]:
    """
    Return identifier names that occur anywhere in the Python function body (load/store),
    excluding the arg placeholder itself.
    """
    key = (id(func_list), int(idx))
    hit = _RESERVED_CACHE_PY.get(key)
    if hit is not None:
        return set(hit)

    if idx == -1:
        _RESERVED_CACHE_PY[key] = set()
        return set()

    body = func_list[idx](arg_name)
    src = "def _f(" + arg_name + "):\n" + _indent_block(body)

    try:
        tree = ast.parse(src)
    except SyntaxError:
        # If a body ever fails to parse, fall back to a conservative regex scan.
        names = set(_IDENT_RE.findall(body))
        names.discard(arg_name)
        _RESERVED_CACHE_PY[key] = names
        return set(names)

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)

    names.discard(arg_name)
    _RESERVED_CACHE_PY[key] = names
    return set(names)


def reserved_idents_in_java_body(func_list, idx: int, *, arg_name: str = "__ARG__") -> set[str]:
    """
    Conservative identifier scan for Java bodies:
    - extracts all identifier-ish tokens
    - filters keywords and common class names
    - excludes the arg placeholder itself
    """
    key = (id(func_list), int(idx))
    hit = _RESERVED_CACHE_JAVA.get(key)
    if hit is not None:
        return set(hit)

    if idx == -1:
        _RESERVED_CACHE_JAVA[key] = set()
        return set()

    body = func_list[idx](arg_name) if callable(func_list[idx]) else str(func_list[idx])
    toks = set(_IDENT_RE.findall(body))
    toks.discard(arg_name)

    filtered = {
        t for t in toks
        if t not in JAVA_BUILTINS and t.lower() not in JAVA_KEYWORDS
    }
    _RESERVED_CACHE_JAVA[key] = filtered
    return set(filtered)


def _random_type_ident(*, exclude_type: Optional[str] = None) -> str:
    """
    Sample an identifier alias from the finite type set.
    If exclude_type is given, sample uniformly from types != exclude_type.
    """
    if exclude_type is None:
        return _type_to_ident(random.choice(_TYPE_LIST))

    candidates = [t for t in _TYPE_LIST if t != exclude_type]
    if not candidates:
        raise RuntimeError("No mismatch candidates available (type set too small).")
    return _type_to_ident(random.choice(candidates))


def _type_to_ident(type_name: str) -> str:
    try:
        return _TYPE_TO_IDENT[type_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported type for identifier: {type_name!r}") from exc


def get_identifier(model_name, _type: str, policy: int, default_val=None, override_perturb=False) -> str:
    if override_perturb:
        return random_semantic_disagree_identifier_name(_type, model_name=model_name)
    if policy == policyFixed:
        if default_val is None:
            raise ValueError("default_val must be specified for policyFixed")
        return default_val
    if policy == policyRandomize:
        return random_neutral_identifier_name(model_name=model_name)
    if policy == policySemanticRandom:
        # Randomly pick from the finite type-alias list (not semantic tokens).
        return _random_type_ident()

    if policy == policySemanticMismatch:
        # Randomly pick an alias corresponding to a different type.
        return _random_type_ident(exclude_type=_type)

    if policy == policySemanticAgree:
        # Deterministically agree with the (finite) type by using its alias.
        return _type_to_ident(_type)
    raise ValueError(f"Unknown policy {policy}")


# Use Optional for compatibility with Python < 3.10.
# empty string represents no type annotation
typeListMinimal: List[str] = ["int", "str", "bool"]
typeListExtension: List[str] = ["list[int]", "list[str]", "list[bool]", "float", "list[float]"]


def get_type_list(is_minimal: bool) -> List[str]:
    return typeListMinimal if is_minimal else typeListMinimal + typeListExtension


def type_to_class(t: str, is_minimal: bool) -> int:
    types = get_type_list(is_minimal)
    if t not in types:
        raise ValueError(f"Unknown type '{t}'. Expected one of {types}.")
    return types.index(t)


def get_nihilTypeTgtClass(args) -> int:
    return len(get_type_list(args.minimal))


# ---------------- signature map ----------------

# Map from (input_type, output_type) to the corresponding function list.
Python_fncMap = {
    ("int", "int"): python_bank.INT_INT_FUNCS,
    ("int", "str"): python_bank.INT_STR_FUNCS,
    ("int", "bool"): python_bank.INT_BOOL_FUNCS,
    ("int", "list[int]"): python_bank.INT_LIST_INT_FUNCS,
    ("int", "list[str]"): python_bank.INT_LIST_STR_FUNCS,
    ("int", "list[bool]"): python_bank.INT_LIST_BOOL_FUNCS,

    ("str", "int"): python_bank.STR_INT_FUNCS,
    ("str", "str"): python_bank.STR_STR_FUNCS,
    ("str", "bool"): python_bank.STR_BOOL_FUNCS,
    ("str", "list[int]"): python_bank.STR_LIST_INT_FUNCS,
    ("str", "list[str]"): python_bank.STR_LIST_STR_FUNCS,
    ("str", "list[bool]"): python_bank.STR_LIST_BOOL_FUNCS,

    ("bool", "int"): python_bank.BOOL_INT_FUNCS,
    ("bool", "str"): python_bank.BOOL_STR_FUNCS,
    ("bool", "bool"): python_bank.BOOL_BOOL_FUNCS,
    ("bool", "list[int]"): python_bank.BOOL_LIST_INT_FUNCS,
    ("bool", "list[str]"): python_bank.BOOL_LIST_STR_FUNCS,
    ("bool", "list[bool]"): python_bank.BOOL_LIST_BOOL_FUNCS,

    ("list[int]", "int"): python_bank.LIST_INT_INT_FUNCS,
    ("list[int]", "str"): python_bank.LIST_INT_STR_FUNCS,
    ("list[int]", "bool"): python_bank.LIST_INT_BOOL_FUNCS,
    ("list[int]", "list[int]"): python_bank.LIST_INT_LIST_INT_FUNCS,
    ("list[int]", "list[str]"): python_bank.LIST_INT_LIST_STR_FUNCS,
    ("list[int]", "list[bool]"): python_bank.LIST_INT_LIST_BOOL_FUNCS,

    ("list[str]", "int"): python_bank.LIST_STR_INT_FUNCS,
    ("list[str]", "str"): python_bank.LIST_STR_STR_FUNCS,
    ("list[str]", "bool"): python_bank.LIST_STR_BOOL_FUNCS,
    ("list[str]", "list[int]"): python_bank.LIST_STR_LIST_INT_FUNCS,
    ("list[str]", "list[str]"): python_bank.LIST_STR_LIST_STR_FUNCS,
    ("list[str]", "list[bool]"): python_bank.LIST_STR_LIST_BOOL_FUNCS,

    ("list[bool]", "int"): python_bank.LIST_BOOL_INT_FUNCS,
    ("list[bool]", "str"): python_bank.LIST_BOOL_STR_FUNCS,
    ("list[bool]", "bool"): python_bank.LIST_BOOL_BOOL_FUNCS,
    ("list[bool]", "list[int]"): python_bank.LIST_BOOL_LIST_INT_FUNCS,
    ("list[bool]", "list[str]"): python_bank.LIST_BOOL_LIST_STR_FUNCS,
    ("list[bool]", "list[bool]"): python_bank.LIST_BOOL_LIST_BOOL_FUNCS,

    ("float", "int"): python_bank.FLOAT_INT_FUNCS,
    ("float", "str"): python_bank.FLOAT_STR_FUNCS,
    ("float", "bool"): python_bank.FLOAT_BOOL_FUNCS,
    ("float", "float"): python_bank.FLOAT_FLOAT_FUNCS,
    ("float", "list[int]"): python_bank.FLOAT_LIST_INT_FUNCS,
    ("float", "list[str]"): python_bank.FLOAT_LIST_STR_FUNCS,
    ("float", "list[bool]"): python_bank.FLOAT_LIST_BOOL_FUNCS,
    ("float", "list[float]"): python_bank.FLOAT_LIST_FLOAT_FUNCS,

    ("list[float]", "int"): python_bank.LIST_FLOAT_INT_FUNCS,
    ("list[float]", "str"): python_bank.LIST_FLOAT_STR_FUNCS,
    ("list[float]", "bool"): python_bank.LIST_FLOAT_BOOL_FUNCS,
    ("list[float]", "float"): python_bank.LIST_FLOAT_FLOAT_FUNCS,
    ("list[float]", "list[int]"): python_bank.LIST_FLOAT_LIST_INT_FUNCS,
    ("list[float]", "list[str]"): python_bank.LIST_FLOAT_LIST_STR_FUNCS,
    ("list[float]", "list[bool]"): python_bank.LIST_FLOAT_LIST_BOOL_FUNCS,
    ("list[float]", "list[float]"): python_bank.LIST_FLOAT_LIST_FLOAT_FUNCS,

    ("int", "float"): python_bank.INT_FLOAT_FUNCS,
    ("str", "float"): python_bank.STR_FLOAT_FUNCS,
    ("bool", "float"): python_bank.BOOL_FLOAT_FUNCS,
    ("list[int]", "float"): python_bank.LIST_INT_FLOAT_FUNCS,
    ("list[str]", "float"): python_bank.LIST_STR_FLOAT_FUNCS,
    ("list[bool]", "float"): python_bank.LIST_BOOL_FLOAT_FUNCS,

    ("int", "list[float]"): python_bank.INT_LIST_FLOAT_FUNCS,
    ("str", "list[float]"): python_bank.STR_LIST_FLOAT_FUNCS,
    ("bool", "list[float]"): python_bank.BOOL_LIST_FLOAT_FUNCS,
    ("list[int]", "list[float]"): python_bank.LIST_INT_LIST_FLOAT_FUNCS,
    ("list[str]", "list[float]"): python_bank.LIST_STR_LIST_FLOAT_FUNCS,
    ("list[bool]", "list[float]"): python_bank.LIST_BOOL_LIST_FLOAT_FUNCS,
}

add_colon = lambda s: f": {s}" if s else s
add_type_arrow = lambda s: f" -> {s}" if s else s

FIM_TOKENS_BY_MODEL = {
    "bigcode/santacoder": {
        "prefix": "<fim-prefix>",
        "suffix": "<fim-suffix>",
        "middle": "<fim-middle>",
    },
    "codellama/CodeLlama-7b-hf": {
        "prefix": "<PRE>",
        "suffix": "<SUF>",
        "middle": "<MID>",
    },
}


def _fim_tokens_for_model(model_name: str) -> dict:
    if model_name not in FIM_TOKENS_BY_MODEL:
        raise ValueError(
            f"Unknown model '{model_name}' for FIM tokens. "
            f"Add it to FIM_TOKENS_BY_MODEL."
        )
    return FIM_TOKENS_BY_MODEL[model_name]


def _indent_block(src: str) -> str:
    src = src.replace("\t", "    ")
    return "\n".join("    " + line for line in src.splitlines())


def get_fnc_body(list, idx, arg_name: str):
    if idx == -1:
        return "pass"
    body = list[idx](arg_name)
    return _indent_block(body)


def _java_type(type_name: str) -> str:
    if not type_name:
        return "Object"
    java_type_map = {
        "int": "int",
        "str": "String",
        "bool": "boolean",
        "float": "float",
        "list[int]": "List<Integer>",
        "list[str]": "List<String>",
        "list[bool]": "List<Boolean>",
        "list[float]": "List<Float>",
    }
    return java_type_map.get(type_name, "Object")


def _java_string_literal(value: str) -> str:
    return json.dumps(value)


def _java_float_literal(value: float) -> str:
    text = repr(float(value))
    if "e" not in text and "." not in text:
        text = f"{text}.0"
    return f"{text}f"


def _java_value_literal(value, type_name: str) -> str:
    if type_name == "int":
        return str(int(value))
    if type_name == "str":
        return _java_string_literal(str(value))
    if type_name == "bool":
        return "true" if bool(value) else "false"
    if type_name == "float":
        return _java_float_literal(float(value))
    if type_name.startswith("list["):
        if not isinstance(value, list):
            return "new ArrayList<>()"
        if type_name == "list[int]":
            items = ", ".join(str(int(v)) for v in value)
        elif type_name == "list[str]":
            items = ", ".join(_java_string_literal(str(v)) for v in value)
        elif type_name == "list[bool]":
            items = ", ".join("true" if bool(v) else "false" for v in value)
        elif type_name == "list[float]":
            items = ", ".join(_java_float_literal(float(v)) for v in value)
        else:
            items = ""
        return f"Arrays.asList({items})"
    return str(value)


def _java_literal_from_text(text: str, type_name: str) -> str:
    if not type_name:
        return text
    try:
        value = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    return _java_value_literal(value, type_name)


def _java_default_return(type_name: str) -> str:
    if not type_name:
        return "return null;"
    if type_name == "int":
        return "return 0;"
    if type_name == "str":
        return "return \"\";"
    if type_name == "bool":
        return "return false;"
    if type_name == "float":
        return "return 0.0f;"
    if type_name.startswith("list["):
        return "return new ArrayList<>();"
    return "return null;"


def _java_fnc_body(list, idx, arg_name: str, out_type: str) -> str:
    if idx == -1:
        return _indent_block(_java_default_return(out_type))
    body = list[idx](arg_name) if callable(list[idx]) else list[idx]
    return _indent_block(body)


def py_gen_example(func1_name: str, func2_name: str,
                   func1_body_idx: int, func2_body_idx: int,
                   func1_src_list: List[Callable[[str], str]], func2_src_list: List[Callable[[str], str]],
                   maybe_func1_out_type: str, maybe_func2_out_type: str,
                   maybe_func1_in_type: str, maybe_func2_in_type: str,
                   a_value: str, maybe_a_type: str, maybe_b_type: str,

                   func1_arg_name: str, func2_arg_name: str,
                   a_id_name: str, b_id_name: str,
                   model_name: str):
    fim = _fim_tokens_for_model(model_name)
    return f"""{fim['prefix']}def f_{func1_name}({func1_arg_name}{add_colon(maybe_func1_in_type)}){add_type_arrow(maybe_func1_out_type)}:
{get_fnc_body(func1_src_list, func1_body_idx, func1_arg_name)}

def f_{func2_name}({func2_arg_name}{add_colon(maybe_func2_in_type)}){add_type_arrow(maybe_func2_out_type)}:
{get_fnc_body(func2_src_list, func2_body_idx, func2_arg_name)}

{a_id_name}{add_colon(maybe_a_type)} = {a_value}

{b_id_name}{add_colon(maybe_b_type)} = f_{fim['suffix']}({a_id_name}){fim['middle']}
    """


def java_gen_example(func1_name: str, func2_name: str,
                     func1_body_idx: int, func2_body_idx: int,
                     func1_src_list: List[Callable[[str], str]], func2_src_list: List[Callable[[str], str]],
                     maybe_func1_out_type: str, maybe_func2_out_type: str,
                     maybe_func1_in_type: str, maybe_func2_in_type: str,
                     a_value: str, maybe_a_type: str, maybe_b_type: str,
                     func1_arg_name: str, func2_arg_name: str,
                     a_id_name: str, b_id_name: str,
                     model_name: str):
    fim = _fim_tokens_for_model(model_name)
    func1_out = _java_type(maybe_func1_out_type)
    func2_out = _java_type(maybe_func2_out_type)
    func1_in = _java_type(maybe_func1_in_type)
    func2_in = _java_type(maybe_func2_in_type)
    a_type = _java_type(maybe_a_type)
    b_type = _java_type(maybe_b_type)
    a_value_java = _java_literal_from_text(a_value, maybe_a_type)
    return f"""{fim['prefix']}import java.util.*;

class Main {{
    static {func1_out} f_{func1_name}({func1_in} {func1_arg_name}) {{
{_java_fnc_body(func1_src_list, func1_body_idx, func1_arg_name, maybe_func1_out_type)}
    }}

    static {func2_out} f_{func2_name}({func2_in} {func2_arg_name}) {{
{_java_fnc_body(func2_src_list, func2_body_idx, func2_arg_name, maybe_func2_out_type)}
    }}

    public static void main(String[] argv) {{
        {a_type} {a_id_name} = {a_value_java};
        {b_type} {b_id_name} = f_{fim['suffix']}({a_id_name}){fim['middle']};
    }}
}}
    """


def schemaV2() -> pa.Schema:
    return pa.schema([
        # -1 is "pass"
        pa.field("func1BodyIndex", pa.int8()),
        pa.field("func2BodyIndex", pa.int8()),

        pa.field("f1ArgID", pa.string()),
        pa.field("f2ArgID", pa.string()),
        pa.field("aID", pa.string()),
        pa.field("bID", pa.string()),

        pa.field("funcNamePolicy", pa.int8()),
        pa.field("funcArgPolicy", pa.int8()),
        pa.field("aIDPolicy", pa.int8()),
        pa.field("bIDPolicy", pa.int8()),

        pa.field("func1Name", pa.string()),
        pa.field("func2Name", pa.string()),

        pa.field("aVarExpectedType", pa.string()),
        pa.field("aVarValue", pa.string()),

        pa.field("func1InputType", pa.string()),
        pa.field("func2InputType", pa.string()),

        pa.field("func1OutType", pa.string()),
        pa.field("func2OutType", pa.string()),

        pa.field("func1InHasTypeTag", pa.bool_()),
        pa.field("func2InHasTypeTag", pa.bool_()),

        pa.field("func1OutHasTypeTag", pa.bool_()),
        pa.field("func2OutHasTypeTag", pa.bool_()),

        pa.field("aHasTypeTag", pa.bool_()),
        pa.field("bHasTypeTag", pa.bool_()),

        pa.field("b_expectedType", pa.string()),
        pa.field("expectedFunctionIDX", pa.int8()),  # 0 for func1, 1 for func2; it also encodes if the target is nearest
    ])


def schemaV2_dataset_fields(
        schema: Optional[Union[pa.Schema, pa.StructType]] = None
) -> Tuple[str, ...]:
    if schema is None:
        dataset_type = schemaV2()
    elif isinstance(schema, pa.StructType):
        dataset_type = schema
    elif isinstance(schema, pa.Schema):
        if "dataset" in schema.names:
            dataset_type = schema.field("dataset").type
        else:
            return tuple(schema.names)
    else:
        raise TypeError("schema must be a pyarrow.Schema or pyarrow.StructType")
    return tuple(dataset_type.names)


def schemaV2_dataset_flat(
        schema: Optional[Union[pa.Schema, pa.StructType]] = None
) -> pa.Schema:
    if schema is None:
        dataset_type = schemaV2()
    elif isinstance(schema, pa.StructType):
        dataset_type = schema
    elif isinstance(schema, pa.Schema):
        if "dataset" in schema.names:
            dataset_type = schema.field("dataset").type
        else:
            return schema
    else:
        raise TypeError("schema must be a pyarrow.Schema or pyarrow.StructType")
    return pa.schema(list(dataset_type))


def schemaV2_meta() -> pa.Schema:
    return pa.schema([
        ("nClassesTask0", pa.int64()),
        ("nClassesTask1", pa.int64()),
        ("nClassesTask2", pa.int64()),
    ])


@dataclass
class ParquetStreamConfigV2:
    out_path: Path
    schema: pa.Schema
    batch_size: int = 10_000
    compression: str = "zstd"
    use_dictionary: bool = True

    # If True, dedupe within this one run (in-memory set).
    dedupe: bool = False
    dedupe_keys: Optional[Tuple[str, ...]] = None

    # If True, delete existing out_path first (safer during iteration)
    overwrite: bool = True

    def __post_init__(self) -> None:
        if self.dedupe_keys is None:
            self.dedupe_keys = schemaV2_dataset_fields(self.schema)


if __name__ == "__main__":
    print("Some examples.")
    s = py_gen_example(
        func1_name="inc",
        func2_name="to_hex",
        func1_body_idx=0,  # INT_INT_FUNCS[0](arg) -> "return arg + 1"
        func2_body_idx=1,  # INT_STR_FUNCS[1](arg) -> "return format(arg, 'x')"
        func1_src_list=python_bank.INT_INT_FUNCS,
        func2_src_list=python_bank.INT_STR_FUNCS,
        maybe_func1_out_type="int",
        maybe_func2_out_type="str",
        maybe_func1_in_type="int",
        maybe_func2_in_type="int",
        a_value="41",
        maybe_a_type="int",
        maybe_b_type="str",
        func1_arg_name="x",
        func2_arg_name="y",
        a_id_name="a",
        b_id_name="b",
        model_name="bigcode/santacoder",
    )

    print(s)
    import function_bank_java as java_bank

    s = java_gen_example(
        func1_name="inc",
        func2_name="to_hex",
        func1_body_idx=0,  # e.g., first INT->INT Java body
        func2_body_idx=1,  # e.g., second INT->STR Java body
        func1_src_list=java_bank.INT_INT_FUNCS,
        func2_src_list=java_bank.INT_STR_FUNCS,
        maybe_func1_out_type="int",
        maybe_func2_out_type="str",  # maps to Java "String"
        maybe_func1_in_type="int",
        maybe_func2_in_type="list[int]",
        a_value="[41]",
        maybe_a_type="list[str]",
        maybe_b_type="list[bool]",  # maps to Java "String"
        func1_arg_name="x",
        func2_arg_name="y",
        a_id_name="a",
        b_id_name="b",
        model_name="bigcode/santacoder",
    )

    print(s)
