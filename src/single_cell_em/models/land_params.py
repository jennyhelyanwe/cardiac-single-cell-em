"""Land (2017) contraction parameters, exported as a single source of truth."""
from dataclasses import dataclass, replace, asdict
import hashlib, json


@dataclass(frozen=True)
class LandParams:
    perm50: float = 0.35
    TRPN_n: float = 2.0
    koff: float = 0.1
    dr: float = 0.25
    wfrac: float = 0.5
    TOT_A: float = 25.0
    ktm_unblock: float = 0.021
    beta_1: float = -2.4
    beta_0: float = 2.3
    gamma_xb: float = 0.0085
    gamma_wu: float = 0.615
    phi: float = 2.23
    nperm: float = 2.036
    ca50: float = 0.805
    Tref: float = 120.0
    nu: float = 7.0
    mu: float = 3.0
    lambda_min: float = 0.87
    lambda_max: float = 1.2

    # derived
    @property
    def k_ws(self):  return 0.004 * self.mu
    @property
    def k_uw(self):  return 0.026 * self.nu
    @property
    def cds(self):   return self.phi * self.k_ws * (1 - self.dr) * self.wfrac / self.dr
    @property
    def cdw(self):   return (self.phi * self.k_uw * (1 - self.dr) * (1 - self.wfrac)
                             / ((1 - self.dr) * self.wfrac))
    @property
    def k_wu(self):  return self.k_uw * (1.0 / self.wfrac - 1) - self.k_ws
    @property
    def k_su(self):  return self.k_ws * (1.0 / self.dr - 1) * self.wfrac
    @property
    def A(self):     return (0.25 * self.TOT_A) / ((1 - self.dr) * self.wfrac + self.dr) * (self.dr / 0.25)

    @property
    def tau_zetas(self): return 1.0 / self.cds
    @property
    def tau_zetaw(self): return 1.0 / self.cdw
    @property
    def dt_stable_euler(self):
        """Forward-Euler stability limit set by the fastest distortion variable."""
        return 2.0 * min(self.tau_zetas, self.tau_zetaw)

    def hash(self, n=10):
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:n]


LAND2017 = LandParams()

# Fill these in once you have Xin's file. Keep his numbers here, nowhere else.
XIN_RETUNED = replace(LAND2017)   # e.g. replace(LAND2017, Tref=..., ca50=..., ...)
MARGARA_RETUNED = replace(LAND2017)
REGISTRY = {"land2017": LAND2017, "xin_retuned": XIN_RETUNED} #, "margara_retuned": MARGARA_RETUNED }