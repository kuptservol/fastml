def test_near_zero(a, tol=1e-3): assert a.abs() < tol, f"Near zero: {a}"
