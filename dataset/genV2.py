import argparse
import time
from itertools import product
from typing import List, Optional, Tuple, Dict, Iterator, Any, Iterable, Union

#workaround
try:
    from dataset.metadata import *
    from dataset.name_utils import *
    from dataset.value_utils import *
    from dataset.parquet_stream import ParquetStreamConfig
    from dataset.strict_normalize import normalize_record_strict, ParquetStreamWriterV2
except ModuleNotFoundError:
    from metadata import *
    from name_utils import *
    from value_utils import *
    from parquet_stream import ParquetStreamConfig
    from strict_normalize import normalize_record_strict, ParquetStreamWriterV2

import function_bank_java as java_bank

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outV2"

depth = 5  # f1(Input/Output) + f2(Input/Output) + atype


def fmt_s(secs: float) -> str:
    if secs < 60:
        return f"{secs:.1f}s"
    if secs < 3600:
        return f"{secs / 60:.1f}m"
    return f"{secs / 3600:.2f}h"


_TYPE_TO_TOKEN = {
    "int": "INT",
    "str": "STR",
    "bool": "BOOL",
    "float": "FLOAT",
    "list[int]": "LIST_INT",
    "list[str]": "LIST_STR",
    "list[bool]": "LIST_BOOL",
    "list[float]": "LIST_FLOAT",
}


def _java_func_list(in_type: str, out_type: str):
    name = f"{_TYPE_TO_TOKEN[in_type]}_{_TYPE_TO_TOKEN[out_type]}_FUNCS"
    return getattr(java_bank, name)


def _avoid_forbidden(base: str, forbidden: set[str]) -> str:
    """
    If `base` is forbidden, return a deterministic fresh variant: base_2, base_3, ...
    """
    if base not in forbidden:
        return base
    i = 2
    while f"{base}_{i}" in forbidden:
        i += 1
    return f"{base}_{i}"


def _make_unique_idents(names: Dict[str, str]) -> Dict[str, str]:
    """
    Ensure all identifiers in `names` are unique by adding deterministic suffixes.
    This is required for Java compilation (no redeclaration in same scope) and also
    avoids accidental shadowing/weirdness in Python examples.
    """
    used: set[str] = set()
    out: Dict[str, str] = {}
    for key, base in names.items():
        cand = base
        if cand in used:
            i = 2
            while f"{base}_{i}" in used:
                i += 1
            cand = f"{base}_{i}"
        used.add(cand)
        out[key] = cand
    return out


def genV2(args, all_tkns_non_semantics: Iterable[Tuple[int, str]]) -> Iterator[Dict[str, Any]]:
    overridePerturbateOnlyTarget = args.perturbateOnlyTarget
    functionNamePolicy = args.functionNamePolicy
    hyper_loop = product(get_type_list(args.minimal), repeat=depth)

    if functionNamePolicy == policyFixed:
        raise ValueError("policyFixed is not supported for function names.")

    for f1In, f1Out, f2In, f2Out, aType in hyper_loop:
        f1List = Python_fncMap[(f1In, f1Out)]
        f2List = Python_fncMap[(f2In, f2Out)]

        f1List_java = _java_func_list(f1In, f1Out)
        f2List_java = _java_func_list(f2In, f2Out)

        for f1IDX, f2IDX in product(range(len(f1List)), range(len(f2List))):
            f1 = f1List[f1IDX]
            f2 = f2List[f2IDX]

            it = None
            if args.allTypeTagChoices:
                it = product([True, False], repeat=6)
            elif args.onlyTypeTag:
                it = [(True, True, True, True, True, True)]
            elif args.noTypeTag:
                it = [(False, False, False, False, False, False)]
            elif args.mixedTypeTag:
                it = [(False, False, False, False, False, False),
                      (True, True, True, True, True, True)]
            else:
                raise ValueError("invalid combination of --allTypeTagChoices, --onlyTypeTag, --noTypeTag, and --mixedTypeTag")

            for choices in it:
                f1InHasTag, f2InHasTag, f1OutHasTag, f2OutHasTag, aHasTypeTag, bHasTypeTag = choices
                expectedFunctionIDX = None

                if aType == f1In:
                    expectedFunctionIDX = f1TargetClass

                if aType == f2In:
                    if expectedFunctionIDX is None:
                        expectedFunctionIDX = f2TargetClass
                    else:
                        if args.equiprobClassTask0:
                            expectedFunctionIDX = get_equiprobFunc_tgt_class(args)
                        else:
                            # TODO: equally split 0 and 1
                            continue

                if expectedFunctionIDX is None:
                    if args.nullClassTask0:
                        expectedFunctionIDX = nullFunctionTargetClass
                    else:
                        # skipping cases in which the model cannot pick any correct function
                        continue

                bExpectedType = None
                if expectedFunctionIDX == f1TargetClass:
                    bExpectedType = f1Out
                elif expectedFunctionIDX == f2TargetClass:
                    bExpectedType = f2Out

                if bExpectedType is None:
                    if args.nullClassTask2:
                        bExpectedType = prober_meta_null_type
                    else:
                        continue

                def mustPerturb(idx):
                    if not overridePerturbateOnlyTarget:
                        return False
                    return idx == expectedFunctionIDX

                f1ArgID_str = get_identifier(model_name=args.model, _type=f1In, policy=args.identifierPolicyFncArg, default_val="x")
                f2ArgID_str = get_identifier(model_name=args.model, _type=f2In, policy=args.identifierPolicyFncArg, default_val="x")
                aID_str = get_identifier(model_name=args.model, _type=aType, policy=args.identifierPolicyA, default_val="a")
                bID_str = get_identifier(model_name=args.model, _type=bExpectedType, policy=args.identifierPolicyB, default_val="b")

                if args.identifierPolicyFncArg > policyRandomize:
                    f1ArgID_str = "var_" + f1ArgID_str
                    f2ArgID_str = "var_" + f2ArgID_str

                if args.identifierPolicyA > policyRandomize:
                    aID_str = "var_" + aID_str
                    bID_str = "var_" + bID_str

                # ---- freshness against identifiers used INSIDE the chosen function bodies ----
                forbidden_f1 = reserved_idents_in_py_body(f1List, f1IDX) | reserved_idents_in_java_body(f1List_java, f1IDX)
                forbidden_f2 = reserved_idents_in_py_body(f2List, f2IDX) | reserved_idents_in_java_body(f2List_java, f2IDX)

                f1ArgID_str = _avoid_forbidden(f1ArgID_str, forbidden_f1)
                f2ArgID_str = _avoid_forbidden(f2ArgID_str, forbidden_f2)

                uniq = _make_unique_idents(
                    {
                        "f1ArgID": f1ArgID_str,
                        "f2ArgID": f2ArgID_str,
                        "aID": aID_str,
                        "bID": bID_str,
                    }
                )
                f1ArgID_str = uniq["f1ArgID"]
                f2ArgID_str = uniq["f2ArgID"]
                aID_str = uniq["aID"]
                bID_str = uniq["bID"]

                yield {
                    "func1BodyIndex": f1IDX,
                    "func2BodyIndex": f2IDX,

                    "f1ArgID": f1ArgID_str,
                    "f2ArgID": f2ArgID_str,
                    "aID": aID_str,
                    "bID": bID_str,

                    "funcNamePolicy": functionNamePolicy,
                    "funcArgPolicy": args.identifierPolicyFncArg,
                    "aIDPolicy": args.identifierPolicyA,
                    "bIDPolicy": args.identifierPolicyB,

                    "func1Name": get_identifier(model_name=args.model, _type=f1Out, policy=functionNamePolicy, override_perturb=mustPerturb(0), default_val="a"),
                    "func2Name": get_identifier(model_name=args.model, _type=f2Out, policy=functionNamePolicy, override_perturb=mustPerturb(1), default_val="b"),

                    "aVarExpectedType": aType,
                    "aVarValue": f"{get_literal_of_type(aType, all_tkns_non_semantics)}",

                    "func1InputType": f1In,
                    "func2InputType": f2In,

                    "func1OutType": f1Out,
                    "func2OutType": f2Out,

                    "func1InHasTypeTag": f1InHasTag,
                    "func2InHasTypeTag": f2InHasTag,

                    "func1OutHasTypeTag": f1OutHasTag,
                    "func2OutHasTypeTag": f2OutHasTag,

                    "aHasTypeTag": aHasTypeTag,
                    "bHasTypeTag": bHasTypeTag,

                    "b_expectedType": bExpectedType,
                    "expectedFunctionIDX": expectedFunctionIDX,
                }


def write_records_to_parquet(
        records: Iterable[Dict[str, Any]],
        out_path: Union[str, Path],
        *,
        schema: pa.Schema,
        batch_size: int = 10_000,
        compression: str = "zstd",
        use_dictionary: bool = True,
        dedupe: bool = False,
        dedupe_keys: Optional[Tuple[str, ...]] = None,
        overwrite: bool = True,
) -> Tuple[int, int]:
    if dedupe_keys is None:
        dedupe_keys = tuple(schema.names)
    cfg = ParquetStreamConfig(
        out_path=Path(out_path),
        schema=schema,
        batch_size=batch_size,
        compression=compression,
        use_dictionary=use_dictionary,
        dedupe=dedupe,
        dedupe_keys=dedupe_keys,
        overwrite=overwrite,
    )
    w = ParquetStreamWriterV2(cfg)
    try:
        for r in records:
            w.add(normalize_record_strict(r, schema))
    finally:
        w.close()
    return w.written, w.skipped_dupes


def main():
    ap = argparse.ArgumentParser(
        description="""
Policies:
  - 0: Chooses a fixed identifier name (x, y, f_a, f_b).
  - 1: Chooses a random identifier.
  - 2: Chooses a random identifier from semantically meaningful tokens (e.g., int, string, str, list, list_str, number).
  - 3: Chooses a random adversarial identifier. Example: String var_int = "x".
  - 4: Always uses agreeing identifiers. Example: String var_string = "test".
    """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--complete", action="store_true", help="Generate all examples, including those that are not semantically valid (overrides all)")
    ap.add_argument("--model", type=str, default="bigcode/santacoder", help="Model name.")
    ap.add_argument("--allTypeTagChoices", default=False, action="store_true")
    ap.add_argument("--noTypeTag", default=False, action="store_true")
    ap.add_argument("--onlyTypeTag", default=False, action="store_true")
    ap.add_argument("--mixedTypeTag", default=False, action="store_true")
    ap.add_argument("--identifierPolicyFncArg", default=1, type=int, help="Policy for function arguments.")
    ap.add_argument("--identifierPolicyA", default=1, type=int, help="Policy for variable a.")
    ap.add_argument("--identifierPolicyB", default=1, type=int, help="Policy for variable b.")
    ap.add_argument("--functionNamePolicy", default=1, type=int, help="Policy for function names.")
    ap.add_argument("--perturbateOnlyTarget", default=False, action="store_true", help="Only perturbate the target function. (overrides --functionNamePolicy)")
    ap.add_argument(
        "--identifierPolicyAll",
        default=None,
        type=int,
        help=(
            "Override *all* identifier policies at once (function names, function args, a, b). "
            "If set, it overrides --identifierPolicyFncArg/--identifierPolicyA/--identifierPolicyB "
            "and also --functionNamePolicy (unless --perturbateOnlyTarget is set)."
        ),
    )
    ap.add_argument("--onlyTknExtract", default=False, action="store_true", help="Only extract tokens from the code, do not generate examples.")
    ap.add_argument("--nullClassTask0", default=False, action="store_true", help="Enables examples in which the model cannot pick any correct function.")
    ap.add_argument("--nullClassTask2", default=False, action="store_true", help="Enables examples in which the model cannot deduce the correct type of b.")
    ap.add_argument("--equiprobClassTask0", default=False, action="store_true", help="Add another class – equiprob")
    ap.add_argument("--minimal", default=False, action="store_true", help="Restricts training set to int, str, bool.")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
    args = ap.parse_args()
    random.seed(args.seed)

    if args.identifierPolicyAll is not None:
        args.identifierPolicyFncArg = args.identifierPolicyAll
        args.identifierPolicyA = args.identifierPolicyAll
        args.identifierPolicyB = args.identifierPolicyAll
        if not args.perturbateOnlyTarget:
            args.functionNamePolicy = args.identifierPolicyAll

    if args.complete:
        print("complete mode")
        args.equiprobClassTask0 = True
        args.minimal = False
        args.nullClassTask0 = True
        args.nullClassTask2 = True

    init_name_utils(args.model)
    if args.onlyTknExtract:
        return

    all_tkns: Iterable[Tuple[int, str]] = get_identifier_tokens(args.model)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _typetag_slug(a: argparse.Namespace) -> str:
        if a.noTypeTag:
            return "untagged"
        return "tagged"

    # Encode (functionNamePolicy + typetag) into the output filename.
    # If --perturbateOnlyTarget is used, it overrides the semantics of functionNamePolicy, so encode that explicitly.
    if args.perturbateOnlyTarget:
        fn_slug = "perturbateOnlyTarget"
    else:
        if args.functionNamePolicy == policyRandomize:
            fn_slug = ""
        elif args.functionNamePolicy == policySemanticMismatch:
            fn_slug = "adv_"
        else:
            fn_slug = f"fnPol{args.functionNamePolicy}_"

    tt_slug = _typetag_slug(args)

    out_path = OUT_DIR / f"{fn_slug}{tt_slug}.parquet"

    nClassesTask0 = 2
    nClassesTask1 = len(get_type_list(args.minimal))
    nClassesTask2 = nClassesTask1

    if args.nullClassTask0:
        nClassesTask0 += 1
    if args.equiprobClassTask0:
        nClassesTask0 += 1
    if args.nullClassTask2:
        nClassesTask2 += 1

    meta_schema = schemaV2_meta()
    dataset_schema = schemaV2_dataset_flat()

    print(f"[datasetV2] writing -> {out_path}")
    meta_path = OUT_DIR / f"{fn_slug}{tt_slug}.meta.parquet"
    print(f"[datasetV2] writing meta -> {meta_path}")

    write_records_to_parquet(
        [{"nClassesTask0": nClassesTask0, "nClassesTask1": nClassesTask1, "nClassesTask2": nClassesTask2}],
        meta_path,
        schema=meta_schema,
        batch_size=1,
        compression="zstd",
        dedupe=False,
    )

    t0 = time.perf_counter()
    records = genV2(args, all_tkns)
    if not args.no_progress:
        records = tqdm(records, desc="datasetV2", unit="rows", smoothing=0.05)
    written, dupes = write_records_to_parquet(
        records,
        out_path,
        schema=dataset_schema,
        batch_size=100_000,
        compression="zstd",
        dedupe=False,
    )
    t1 = time.perf_counter()
    print(f"[datasetV2] wrote {written} rows in {fmt_s(t1 - t0)}")


if __name__ == "__main__":
    main()
