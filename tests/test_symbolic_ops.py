import pytest

from attestation import symbolic_ops as ops


def test_simplify_returns_result_latex_and_parsed_input():
    out = ops.op_simplify({"expr": "(x**2 - 1)/(x - 1)"})
    assert out["result"] == "x + 1"
    assert out["latex"] == "x + 1"
    assert out["parsed_input"] == "(x**2 - 1)/(x - 1)"


def test_differentiate_auto_detects_single_symbol():
    out = ops.op_differentiate({"expr": "x**3", "symbol": None, "order": 1})
    assert out["result"] == "3*x**2"


def test_differentiate_respects_order():
    out = ops.op_differentiate({"expr": "x**3", "symbol": None, "order": 2})
    assert out["result"] == "6*x"


def test_solve_auto_detects_and_returns_roots():
    out = ops.op_solve({"expr": "x**2 - 4", "symbol": None})
    assert set(out["result"].strip("[]").split(", ")) == {"-2", "2"}


def test_solve_explicit_symbol_overrides_autodetect():
    """x*y - 1 is ambiguous; naming y must solve for y, not x."""
    out = ops.op_solve({"expr": "x*y - 1", "symbol": "y"})
    assert out["result"] == "[1/x]"


def test_ambiguous_symbol_raises_naming_candidates():
    with pytest.raises(ValueError) as exc:
        ops.op_solve({"expr": "x*y - 1", "symbol": None})
    assert "x" in str(exc.value) and "y" in str(exc.value)


def test_differentiate_rejects_symbol_absent_from_expression():
    """A typo'd symbol must error, not silently return 0."""
    with pytest.raises(ValueError) as exc:
        ops.op_differentiate({"expr": "x**2", "symbol": "y", "order": 1})
    assert "y" in str(exc.value) and "x" in str(exc.value)


def test_solve_rejects_symbol_absent_from_expression():
    """A typo'd symbol must error, not silently return []."""
    with pytest.raises(ValueError) as exc:
        ops.op_solve({"expr": "x**2 - 4", "symbol": "z"})
    assert "z" in str(exc.value) and "x" in str(exc.value)


def test_integrate_explicit_symbol_on_constant_expression_still_works():
    """A constant expression has no free symbols, so any explicit symbol is
    permissive -- there is nothing to be absent from."""
    out = ops.op_integrate({"expr": "1", "symbol": "x", "bounds": None})
    assert out["result"] == "x"


def test_integrate_indefinite_and_definite():
    indef = ops.op_integrate({"expr": "x**2", "symbol": None, "bounds": None})
    assert indef["result"] == "x**3/3"
    definite = ops.op_integrate({"expr": "x**2", "symbol": None, "bounds": [0, 1]})
    assert definite["result"] == "1/3"
    assert definite["numeric"] == pytest.approx(1 / 3)


def test_verify_proves_a_true_identity():
    out = ops.op_verify({"lhs": "sin(x)**2 + cos(x)**2", "rhs": "1"})
    assert out["verdict"] == "equal"


def test_verify_reports_unequal_for_a_clear_mismatch():
    out = ops.op_verify({"lhs": "x + 1", "rhs": "x + 2"})
    assert out["verdict"] == "unequal"


def test_verify_never_claims_unequal_without_numeric_evidence():
    """simplify is incomplete, so 'could not prove' must never render as
    'unequal' -- an agent reading that would report a false disproof."""
    out = ops.op_verify({"lhs": "sqrt(x**2)", "rhs": "x"})
    assert out["verdict"] in ("equal", "unproven", "unequal")
    if out["verdict"] == "unproven":
        assert "not a disproof" in out["message"].lower()
    if out["verdict"] == "unequal":
        # an "unequal" verdict must always cite concrete evidence
        assert any(w in out["message"].lower() for w in ("counterexample", "reduces to"))


def test_evaluate_substitutes_and_returns_float():
    out = ops.op_evaluate({"expr": "x**2 + 1", "subs": {"x": 3}, "units": None})
    assert out["numeric"] == pytest.approx(10.0)


def test_evaluate_converts_units():
    out = ops.op_evaluate({"expr": "5", "subs": None, "units": "meter/second -> kilometer/hour"})
    assert out["numeric"] == pytest.approx(18.0)


def test_derivation_traces_integral_steps():
    out = ops.op_derivation({"expr": "x*sin(x)", "operation": "integrate", "symbol": None})
    assert out["steps"], "expected at least one step"
    assert any("Parts" in s["rule"] for s in out["steps"])


def test_derivation_differentiate_is_labeled_not_faked():
    out = ops.op_derivation({"expr": "x**3", "operation": "differentiate", "symbol": None})
    assert out["result"] == "3*x**2"
    assert "not" in out["note"].lower()


@pytest.mark.parametrize(
    "name",
    ["__builtins__", "definitions", "convert_to", "Quantity", "meter.__class__"],
)
def test_unit_expr_rejects_non_unit_and_attribute_access(name):
    """Regression guard: _unit_expr must reject any module attribute that is
    not a Quantity, and must never allow attribute access through the text it
    parses. See the code review that caught this: getattr-by-presence and a
    dot in the charset let 'meter.__class__' and '__builtins__' through to
    convert_to, which then called sympify on a live Python object."""
    with pytest.raises(ValueError):
        ops._unit_expr(name)


def test_evaluate_rejects_unit_conversion_to_builtins():
    """op_evaluate must surface a ValueError, not reach convert_to/sympify,
    when a caller aims a unit target at a non-unit module attribute."""
    with pytest.raises(ValueError):
        ops.op_evaluate({"expr": "1", "subs": None, "units": "meter -> __builtins__"})


def test_evaluate_converts_units_feet():
    out = ops.op_evaluate({"expr": "5", "subs": None, "units": "meter -> foot"})
    assert out["numeric"] == pytest.approx(16.4, abs=0.01)


def test_unit_expr_rejects_unknown_but_plausible_unit():
    """erg is a real unit name but genuinely absent from sympy.physics.units."""
    with pytest.raises(ValueError):
        ops._unit_expr("erg")


def test_unit_expr_handles_exponents():
    """standard_transformations alone starves the '**' path of the numeric
    constructors it needs (parse_expr('meter**2') raised NameError: name
    'Integer' is not defined) even though 'meter*meter' worked fine. An agent
    converting an area or acceleration got a confusing internal error instead
    of a ValueError."""
    assert str(ops._unit_expr("meter**2")) == "meter**2"


def test_evaluate_converts_units_with_exponents():
    out = ops.op_evaluate({"expr": "5", "subs": None, "units": "meter**2 -> foot**2"})
    assert out["numeric"] == pytest.approx(53.8195, abs=0.01)


def test_numeric_is_null_when_symbols_remain_free():
    """A symbolic result must not report a number.

    The units path needs a coefficient -- 18 in `18*kilometer/hour` -- and the
    fallback that extracts it was reached for EVERY non-numeric result, not
    only unit ones. So `x**2 + 1` reported numeric=1.0 and `2*x` reported 2.0:
    the coefficient of an arbitrary arg, presented as the value. An agent
    quoting that to a researcher is confidently wrong, which is worse than a
    tool that refuses.
    """
    for expr in ("x**2+1", "2*x", "sin(x)", "a*b + c"):
        out = ops.op_evaluate({"expr": expr})
        assert out["numeric"] is None, f"{expr} reported numeric={out['numeric']}"
        assert "free" in out.get("message", "").lower(), (
            f"{expr} should say which symbols are unresolved"
        )


def test_partial_substitution_still_reports_no_number():
    out = ops.op_evaluate({"expr": "x+y", "subs": {"x": 1}})
    assert out["result"] == "y + 1"
    assert out["numeric"] is None
    assert "y" in out["message"]


def test_full_substitution_does_report_a_number():
    out = ops.op_evaluate({"expr": "x**2+1", "subs": {"x": 3}})
    assert out["numeric"] == 10.0


def test_units_conversion_keeps_its_coefficient():
    """The case the fallback exists for must keep working."""
    out = ops.op_evaluate({"expr": "5", "units": "meter/second -> kilometer/hour"})
    assert out["numeric"] == pytest.approx(18.0)
