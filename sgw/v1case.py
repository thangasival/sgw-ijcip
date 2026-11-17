"""Reconstruction of the V1 synthetic 12-port case (V1 manuscript Tables 3-5).

The V1 input CSVs (arcs.csv, destination_shares.csv) were not available, so two inputs are reconstructed
and documented here:
  * route distances: approximate sea distances (nm) with Atlantic/Gulf-to-Pacific routes via a Panama
    Canal waypoint, transit = ceil(nm / (14 kn x 24 h));
  * destination shares of each commodity's demand and destination unloading capacity within V1's
    stated 35-65 kt/day range.
All other values are taken from V1 Tables 3-5.  Numbers are therefore expected to differ from V1 Table 6;
the regression test checks the regime pattern (which scenarios select the waiver, and whether backlog is
eliminated), not exact magnitudes.
"""
import numpy as np
from .network import Network, Scenario

DESTS = ["HNL", "SJU", "ANC", "TPA", "JAX"]
COMM = ["Energy", "Food", "Fertilizer", "Medical", "Defense"]
CLS = {"Energy": "liquid", "Food": "dry", "Fertilizer": "dry", "Medical": "dry", "Defense": "dry"}
DEMAND = {"Energy": 60, "Food": 30, "Fertilizer": 20, "Medical": 5, "Defense": 8}         # kt/day (V1 Table 3)
PRIORITY = {"Energy": 0.80, "Food": 0.70, "Fertilizer": 0.40, "Medical": 1.00, "Defense": 0.90}
SOURCES = {"Energy": ["HOU", "MSY", "PHL"], "Food": ["HOU", "MSY", "OAK", "SEA"], "Fertilizer": ["HOU", "MSY", "TPA"],
           "Medical": ["NYC", "PHL", "LAX", "OAK"], "Defense": ["NYC", "JAX", "SEA", "OAK"]}
SUPPLY = {"Energy": 45, "Food": 20, "Fertilizer": 18, "Medical": 6, "Defense": 9}         # kt/day per source
SHARES = {"HNL": 0.20, "SJU": 0.25, "ANC": 0.15, "TPA": 0.20, "JAX": 0.20}                # reconstructed
H = {"HNL": 45, "SJU": 50, "ANC": 35, "TPA": 65, "JAX": 55}                                # kt/day, reconstructed

# approximate sea distances (nm); Panama waypoint for cross-basin routes (reconstructed)
_PAN = {"HOU": 1500, "MSY": 1400, "TPA": 1250, "JAX": 1550, "NYC": 2000, "PHL": 2000,
        "LAX": 2900, "OAK": 3250, "SEA": 4000}
_PAC = {"HNL": 4700, "ANC": 4900}            # Panama -> Pacific destination
_DIRECT = {  # same-basin
    ("HOU", "TPA"): 870, ("HOU", "JAX"): 1250, ("HOU", "SJU"): 1900,
    ("MSY", "TPA"): 500, ("MSY", "JAX"): 1000, ("MSY", "SJU"): 1650,
    ("TPA", "JAX"): 600, ("TPA", "SJU"): 1250, ("JAX", "TPA"): 600, ("JAX", "SJU"): 1100,
    ("NYC", "TPA"): 1200, ("NYC", "JAX"): 800, ("NYC", "SJU"): 1400,
    ("PHL", "TPA"): 1150, ("PHL", "JAX"): 750, ("PHL", "SJU"): 1350,
    ("LAX", "HNL"): 2230, ("LAX", "ANC"): 2000, ("OAK", "HNL"): 2090, ("OAK", "ANC"): 1850,
    ("SEA", "HNL"): 2400, ("SEA", "ANC"): 1450,
}
_ATL = {"TPA", "JAX", "SJU"}


def distances():
    d = {}
    srcs = sorted({s for v in SOURCES.values() for s in v})
    for s in srcs:
        for j in DESTS:
            if s == j:
                continue
            if (s, j) in _DIRECT:
                d[(s, j)] = _DIRECT[(s, j)]
            elif j in _PAC:                              # Atlantic/Gulf source to Pacific destination
                d[(s, j)] = _PAN[s] + _PAC[j]
            else:                                        # Pacific source to Atlantic/Gulf destination
                d[(s, j)] = _PAN[s] + {"SJU": 1000, "TPA": 1250, "JAX": 1550}[j]
    return d


def network():
    sup = {(s, k): SUPPLY[k] for k in COMM for s in SOURCES[k]}
    return Network(dests=DESTS, commodities=COMM, cls=CLS, sources=SOURCES, dist=distances(), supply=sup, priority=PRIORITY)


# V1 Table 5: base-capacity loss, demand change, port loss, window, lag
SCENARIOS = {
    "Normal":     dict(loss=0.00, dem=0.00, port=0.00, win=(0, 0), lag=2),
    "Moderate":   dict(loss=0.18, dem=0.12, port=0.00, win=(1, 7), lag=2),
    "Severe":     dict(loss=0.72, dem=0.45, port=0.00, win=(1, 12), lag=3),
    "Congestion": dict(loss=0.62, dem=0.30, port=0.42, win=(1, 11), lag=3),
    "Energy":     dict(loss=0.74, dem=0.55, port=0.00, win=(1, 12), lag=2, only="Energy"),
    "Defense":    dict(loss=0.50, dem=4.00, port=0.00, win=(1, 7), lag=1, only="Defense"),
}
CAP = {("liquid", "base"): 1500, ("liquid", "surge"): 250, ("liquid", "waiver"): 800,
       ("dry", "base"): 1900, ("dry", "surge"): 350, ("dry", "waiver"): 1000}
MULT = {"base": 1.0, "surge": 1.9, "waiver": 1.3}


def scenario(name, T=40, release_days=12, loss=None, activation=2.0, waiver_scale=1.0, lag=None):
    s = dict(SCENARIOS[name])
    if loss is not None:
        s["loss"] = loss
    w0, w1 = s["win"]
    inwin = np.zeros(T, bool)
    if w1 > 0:
        inwin[w0 - 1:w1] = True
    q = np.zeros((len(DESTS), len(COMM), T))
    for ki, k in enumerate(COMM):
        for ji, j in enumerate(DESTS):
            for t in range(release_days):
                m = 1.0
                if inwin[t] and (s.get("only") in (None, k)):
                    m += s["dem"]
                q[ji, ki, t] = DEMAND[k] * SHARES[j] * m
    cap = {}
    for (c, v), val in CAP.items():
        arr = np.full(T, float(val))
        if v == "base":
            hit = (s.get("only") is None) or (CLS[s["only"]] == c and s["only"] == "Energy")
            if s.get("only") == "Defense":
                hit = True
            if hit:
                arr[inwin] *= (1 - s["loss"])
        if v == "waiver":
            arr *= waiver_scale
        cap[(c, v)] = arr
    Hm = np.array([[H[j]] * T for j in DESTS], float)
    Hm[:, inwin] *= (1 - s["port"])
    return Scenario(T=T, q=q, cap=cap, H=Hm, mult=MULT, activation=activation,
                    lag=s["lag"] if lag is None else lag, window=s["win"])
