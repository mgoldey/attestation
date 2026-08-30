from attestation import mcp_server


def test_sym_differentiate_returns_structured_result():
    out = mcp_server._sym_differentiate_impl("x**3")
    assert out["ok"] is True
    assert out["result"] == "3*x**2"
    assert out["latex"]


def test_sym_solve_reports_ambiguity_without_crashing():
    out = mcp_server._sym_solve_impl("x*y - 1")
    assert out["ok"] is False
    assert "ambiguous" in out["message"]


def test_malicious_expression_is_refused_not_executed():
    out = mcp_server._sym_simplify_impl("__import__('os').getcwd()")
    assert out["ok"] is False
    assert "attestation" not in str(out), "filesystem path leaked -- code executed"


def test_quoted_string_rce_is_dead_end_to_end(tmp_path):
    """A quoted string literal is a single benign ast.Constant -- no Attribute
    node, no Name, no underscore -- so it used to sail through _screen and
    parse_expr handed back a live Python str. op_simplify then called
    sp.simplify() on that str, which sympifies it with SymPy's UNRESTRICTED
    default namespace and executes it -- verified end-to-end: this exact
    payload previously returned ok=True and created the marker file.

    Uses tmp_path so nothing is written outside pytest's own sandbox.
    """
    marker = tmp_path / "marker"
    quote = chr(39)
    payload = quote + f'__import__("os").system("touch {marker}")' + quote

    out = mcp_server._sym_simplify_impl(payload)

    assert out["ok"] is False
    assert not marker.exists(), "RCE executed -- marker file was created"


def test_runaway_computation_is_cancelled_not_hung():
    """expand((x+1)**200000) would exhaust memory; the subprocess must be killed."""
    out = mcp_server._sym_simplify_impl("(x+1)**200000", timeout=3)
    assert out["ok"] is False
    assert "cancelled" in out["message"] or "memory" in out["message"]


def test_sym_verify_surfaces_the_verdict():
    out = mcp_server._sym_verify_impl("sin(x)**2 + cos(x)**2", "1")
    assert out["ok"] is True
    assert out["verdict"] == "equal"


def test_sym_derivation_flags_untraced_differentiate():
    """`traced` says, in the response itself, whether `steps` is a genuine
    rule-by-rule trace or prose noting SymPy has no step engine for it --
    a fact the docstring stated but the payload did not."""
    from attestation.mcp.symbolic import _sym_derivation

    assert _sym_derivation("x**2", operation="differentiate")["traced"] is False
    assert _sym_derivation("x**2", operation="integrate")["traced"] is True


def test_all_seven_tools_are_served():
    import asyncio

    names = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    expected = {
        "sym.simplify",
        "sym.solve",
        "sym.differentiate",
        "sym.integrate",
        "sym.derivation",
        "sym.verify",
        "sym.evaluate",
    }
    assert expected <= names
