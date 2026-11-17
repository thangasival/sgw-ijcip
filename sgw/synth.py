"""Env-S: labelled synthetic stress-test environment for P4 (E2b).  It keeps the Env-R network, demand and arrivals
and changes only what P4 is about: the repeat-voyage share (vessel lifetimes scaled), throughput scarcity (r_share)
and signal informativeness (policy.inform).  Results from Env-S are never pooled with Env-R results."""
from __future__ import annotations
from . import model as SM


def env_s(t, prm, life_scale):
    from .experiments import get_env
    base = get_env(t, prm)[0]
    VG = base.VG.copy()
    VG["lifetime"] = VG.lifetime * life_scale
    env = SM.Env(west_mbbl=t.get("west_mbbl", 28.3), red_share=t.get("red_share"), VG=VG)
    env.label = f"Env-S (life x{life_scale})"
    return env


def repeat_share(env):
    """Share of voyages that are a vessel's second or later voyage, implied by lifetimes and the mean voyage cycle."""
    import numpy as np
    from . import data as SD
    cyc = np.mean([2 * SD.TRANSIT[j] + 2 for j in SD.DESTS])
    n = (env.VG.lifetime / cyc).clip(lower=1.0)
    return float(1 - len(n) / n.sum())
