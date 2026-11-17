"""Small sparse LP builder on top of scipy.optimize.linprog(method="highs").

Variables are created in named blocks (numpy index arrays), constraints are added row by row
from (var_index, coefficient) lists, and the solve returns primal values, row duals and reduced costs.

Dual sign convention (fixed here once and tested in tests/test_duals.py):
    linprog minimises c'x.  For a row  a'x <= b  the reported marginal y = d(obj)/d(b) <= 0;
    for a row  a'x == b  the marginal y = d(obj)/d(b) has either sign.
    Reduced cost of column j:  c_j - sum_i A_ij y_i.
    We report the *scarcity rent* of a <= row as  pi = -y >= 0.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


class LP:
    def __init__(self):
        self.n = 0
        self.c, self.lb, self.ub = [], [], []
        self.blocks = {}
        self._ub_rows, self._eq_rows = [], []        # each row: (idx array, coef array, rhs, name)
        self._bulk = {"ub": [], "eq": []}            # bulk blocks: (n_rows, row_idx, col_idx, val, rhs)
        self._nrows = {"ub": 0, "eq": 0}

    # ---------------- variables ----------------
    def var(self, name, shape, cost=0.0, lb=0.0, ub=np.inf):
        size = int(np.prod(shape))
        idx = np.arange(self.n, self.n + size).reshape(shape)
        self.n += size
        self.c.append(np.broadcast_to(np.asarray(cost, float), shape).ravel().copy())
        self.lb.append(np.broadcast_to(np.asarray(lb, float), shape).ravel().copy())
        self.ub.append(np.broadcast_to(np.asarray(ub, float), shape).ravel().copy())
        self.blocks[name] = idx
        return idx

    # ---------------- constraints ----------------
    def le(self, idx, coef, rhs, name=None):
        self._ub_rows.append((np.asarray(idx, int).ravel(), np.asarray(coef, float).ravel(), float(rhs), name))
        r = self._nrows["ub"]; self._nrows["ub"] += 1
        return r

    def eq(self, idx, coef, rhs, name=None):
        self._eq_rows.append((np.asarray(idx, int).ravel(), np.asarray(coef, float).ravel(), float(rhs), name))
        r = self._nrows["eq"]; self._nrows["eq"] += 1
        return r

    def bulk(self, kind, n_rows, rows, cols, vals, rhs):
        """Add n_rows constraints at once. rows are 0..n_rows-1 (relative); returns absolute row indices."""
        start = self._nrows[kind]
        self._bulk[kind].append((start, n_rows, np.asarray(rows, int), np.asarray(cols, int), np.asarray(vals, float),
                                 np.broadcast_to(np.asarray(rhs, float), (n_rows,)).copy()))
        self._nrows[kind] += n_rows
        return np.arange(start, start + n_rows)

    def _mat(self, kind, n):
        rows = self._ub_rows if kind == "ub" else self._eq_rows
        m = self._nrows[kind]
        if m == 0:
            return None, None
        # single rows were numbered in the order they were added, interleaved with bulk blocks
        r, cidx, v = [], [], []
        b = np.zeros(m)
        single_ids = [i for i in range(m)]
        taken = np.zeros(m, bool)
        for (start, nr, rr, cc, vv, rhs) in self._bulk[kind]:
            r.append(rr + start); cidx.append(cc); v.append(vv); b[start:start + nr] = rhs; taken[start:start + nr] = True
        free = np.flatnonzero(~taken)
        assert len(free) == len(rows), (len(free), len(rows))
        for i, (ix, cf, rhs, _) in zip(free, rows):
            cf = np.broadcast_to(cf, ix.shape)
            r.append(np.full(ix.size, i)); cidx.append(ix); v.append(cf); b[i] = rhs
        A = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cidx))), shape=(m, n))
        A.sum_duplicates()
        return A, b

    def build(self):
        self.cvec = np.concatenate(self.c); self.lbv = np.concatenate(self.lb); self.ubv = np.concatenate(self.ub)
        self.A_ub, self.b_ub = self._mat("ub", self.n)
        self.A_eq, self.b_eq = self._mat("eq", self.n)
        return self

    def solve(self, b_ub=None, b_eq=None, method="highs-ipm", time_limit=None):
        if not hasattr(self, "cvec"):
            self.build()
        res = linprog(self.cvec, A_ub=self.A_ub, b_ub=self.b_ub if b_ub is None else b_ub,
                      A_eq=self.A_eq, b_eq=self.b_eq if b_eq is None else b_eq,
                      bounds=np.column_stack([self.lbv, self.ubv]), method=method,
                      options={} if time_limit is None else {"time_limit": time_limit})
        if res.status != 0:
            raise RuntimeError(f"LP not solved: {res.message}")
        sol = Solution(res, self)
        return sol


class Solution:
    def __init__(self, res, lp: LP):
        self.res, self.lp = res, lp
        self.x = res.x
        self.obj = res.fun
        self.y_ub = res.ineqlin.marginals if lp.A_ub is not None else np.zeros(0)
        self.y_eq = res.eqlin.marginals if lp.A_eq is not None else np.zeros(0)

    def val(self, name):
        return self.x[self.lp.blocks[name]]

    def rent_ub(self, row):
        """Scarcity rent (>= 0) of a <= row."""
        return -self.y_ub[row]

    def reduced_costs(self):
        rc = self.lp.cvec.copy()
        if self.lp.A_ub is not None:
            rc -= self.lp.A_ub.T @ self.y_ub
        if self.lp.A_eq is not None:
            rc -= self.lp.A_eq.T @ self.y_eq
        return rc
