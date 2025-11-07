from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

try:
    from pandas_datareader import data as pdr
except Exception:  # pragma: no cover
    pdr = None

from scipy.optimize import least_squares
from sklearn.decomposition import PCA

# Config & utilities

@dataclass
class Config:
    # Data window
    start: str = "2005-01-01"
    end: str = None
    # FRED tickers (percent) → tenor years
    fred_tickers: Dict[str, float] = None
    # Forecast pillars (fixed‑maturity quotes we forecast)
    pillars_years: Tuple[float, ...] = (0.25, 2.0, 5.0, 10.0, 30.0)
    # PCA pillars (exclude very short end by default)
    pca_pillars_years: Tuple[float, ...] = (2.0, 5.0, 10.0, 30.0)
    pca_standardize: bool = True  # correlation PCA
    # Backtest horizon and warmup
    forecast_horizon_bdays: int = 21
    min_history_bdays: int = 252 * 2
    # Rolling OU estimation
    estimation_window_bdays: int = 252 * 3
    ou_source: str = "f0ns"  # "tenor" or "f0ns"
    ou_tenor_for_fit: float = 0.25
    # Parameter boxes
    a_bounds: Tuple[float, float] = (0.01, 1.0)
    sigma_bounds: Tuple[float, float] = (0.001, 0.05)
    # Repro
    seed: int = 42

    def __post_init__(self):
        if self.end is None:
            self.end = pd.Timestamp.today().strftime("%Y-%m-%d")
        if self.fred_tickers is None:
            self.fred_tickers = {
                "DGS1MO": 1.0 / 12.0,
                "DGS2": 2.0,
                "DGS5": 5.0,
                "DGS10": 10.0,
                "DGS30": 30.0,
            }


def tenor_label(T: float) -> str:
    """Standard column label (e.g., 2y, 2.5y, 0.0833333y)."""
    return f"{float(T):.6g}y"


def yearfrac_bdays(n_bdays: int) -> float:
    return n_bdays / 252.0


def _safe_exp(x: float) -> float:
    if x > 50:
        return math.exp(50)
    if x < -50:
        return math.exp(-50)
    return math.exp(x)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoid rule with graceful fallback (avoids NumPy deprecation)."""
    try:
        return float(np.trapezoid(y, x))  # NumPy alias
    except AttributeError:
        try:
            from scipy.integrate import trapezoid as trap
            return float(trap(y, x))
        except Exception:
            y = np.asarray(y, float); x = np.asarray(x, float)
            return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))

# Data

class DataLoader:
    """Fetch US Treasury constant maturities from FRED via pandas‑datareader."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch(self) -> pd.DataFrame:
        if pdr is None:
            raise RuntimeError("pandas_datareader not available; install it or provide a CSV loader.")
        syms = list(self.cfg.fred_tickers.keys())
        df = pdr.DataReader(syms, "fred", start=self.cfg.start, end=self.cfg.end)
        df = df.ffill().dropna(how="all")
        df = df / 100.0  # percent → decimals
        # Align to business days and ffill
        bdays = pd.date_range(df.index.min(), df.index.max(), freq=BDay())
        df = df.reindex(bdays).ffill()
        df.index.name = "date"
        rename = {k: tenor_label(self.cfg.fred_tickers[k]) for k in syms}
        return df.rename(columns=rename).sort_index()


# Nelson–Siegel curve

class NelsonSiegel:
    def __init__(self, tenors: np.ndarray, yields: np.ndarray):
        self.tenors = np.asarray(tenors, float)
        self.yields = np.asarray(yields, float)
        self.params_: Optional[np.ndarray] = None

    @staticmethod
    def _ns(T: np.ndarray, b0: float, b1: float, b2: float, tau: float) -> np.ndarray:
        x = T / max(tau, 1e-6)
        with np.errstate(all="ignore"):
            term1 = (1.0 - np.exp(-x)) / np.where(x == 0.0, 1.0, x)
            term2 = term1 - np.exp(-x)
        return b0 + b1 * term1 + b2 * term2

    def fit(self, init: Tuple[float, float, float, float] = None) -> np.ndarray:
        T, y = self.tenors, self.yields
        if init is None:
            init = (y[-1], y[0] - y[-1], 0.0, 1.5)

        def resid(p):
            b0, b1, b2, tau = p
            tau = max(tau, 1e-4)
            return self._ns(T, b0, b1, b2, tau) - y

        bounds = ([-0.05, -2.0, -2.0, 1e-3], [0.20, 2.0, 2.0, 20.0])
        sol = least_squares(resid, x0=np.array(init, float), bounds=bounds, method="trf")
        self.params_ = sol.x
        return self.params_

    def y(self, T: np.ndarray | float) -> np.ndarray:
        assert self.params_ is not None, "Fit NS first."
        b0, b1, b2, tau = self.params_
        T = np.asarray(T, float)
        return self._ns(T, b0, b1, b2, tau)

    def P0(self, T: np.ndarray | float) -> np.ndarray:
        yT = self.y(T)
        return np.exp(-yT * np.asarray(T, float))

    def f0(self, T: np.ndarray | float) -> np.ndarray:
        T = np.asarray(T, float)
        eps = 1e-5
        g = lambda u: self.y(u) * u
        return (g(T + eps) - g(T - eps)) / (2 * eps)

    def df0(self, T: np.ndarray | float) -> np.ndarray:
        T = np.asarray(T, float)
        eps = 1e-4
        return (self.f0(T + eps) - self.f0(T - eps)) / (2 * eps)


# Hull–White 1F 

@dataclass
class HWParams:
    a: float
    sigma: float


class HullWhite1F:
    def __init__(self, curve: NelsonSiegel, params: HWParams):
        self.curve = curve
        self.p = params

    def B(self, t: float, T: float) -> float:
        a = self.p.a
        dt = max(T - t, 0.0)
        return (1.0 - math.exp(-a * dt)) / a

    def A(self, t: float, T: float) -> float:
        a, sigma = self.p.a, self.p.sigma
        if T < t:
            raise ValueError("T must be ≥ t")
        B = self.B(t, T)
        P0T = float(self.curve.P0(T))
        P0t = float(self.curve.P0(t)) if t > 0 else 1.0
        f0t = float(self.curve.f0(t)) if t > 0 else float(self.curve.f0(0.0))
        term_var = (sigma ** 2) / (4.0 * a ** 3) * (1.0 - math.exp(-a * (T - t))) ** 2 * (1.0 - math.exp(-2.0 * a * t))
        lnA = math.log(P0T / P0t) + B * f0t - term_var
        return _safe_exp(lnA)

    def theta(self, t: float) -> float:
        a, sigma = self.p.a, self.p.sigma
        return float(self.curve.df0(t) + a * self.curve.f0(t) + (sigma ** 2) / (2.0 * a) * (1.0 - math.exp(-2.0 * a * t)))

    def r_mean_var(self, r_t: float, t: float, delta: float, n_steps: int = 64) -> Tuple[float, float]:
        a, sigma = self.p.a, self.p.sigma
        s = np.linspace(0.0, delta, n_steps)
        w = np.exp(-a * (delta - s))
        theta_vals = np.array([self.theta(t + float(si)) for si in s])
        integral = _trapz(w * theta_vals, s)
        m = r_t * math.exp(-a * delta) + integral
        v = (sigma ** 2) / (2.0 * a) * (1.0 - math.exp(-2.0 * a * delta))
        return float(m), float(v)

    def zcb_price(self, r_t: float, t: float, T: float) -> float:
        B = self.B(t, T)
        A = self.A(t, T)
        return float(A * math.exp(-B * r_t))

    def expected_zcb_price(self, r_t: float, t: float, T: float, delta: float) -> float:
        tp = t + delta
        if T <= tp:
            raise ValueError("T must exceed t+Δ")
        B = self.B(tp, T)
        A = self.A(tp, T)
        m, v = self.r_mean_var(r_t, t, delta)
        return float(A * math.exp(-B * m + 0.5 * (B ** 2) * v))

    def yield_from_price(self, P: float, t: float, T: float) -> float:
        return -math.log(P) / max(T - t, 1e-12)

    def forecast_yield_fixed_mty(self, r_t: float, delta_years: float, target_mty_years: float) -> float:
        T_model = target_mty_years + delta_years
        P = self.expected_zcb_price(r_t, t=0.0, T=T_model, delta=delta_years)
        return float(self.yield_from_price(P, t=delta_years, T=T_model))


# Estimation & PCA

def _estimate_ou(sr: pd.Series, delta_years: float, a_bounds: Tuple[float, float], sigma_bounds: Tuple[float, float]) -> HWParams:
    sr = sr.dropna()
    if len(sr) < 100:
        return HWParams(a_bounds[0], sigma_bounds[0])
    r0 = sr.shift(1).dropna(); r1 = sr.loc[r0.index]
    X = np.vstack([np.ones_like(r0.values), r0.values]).T
    c_hat, phi_hat = np.linalg.lstsq(X, r1.values, rcond=None)[0]
    phi_hat = float(np.clip(phi_hat, 1e-6, 0.9999))
    a_hat = -math.log(phi_hat) / delta_years
    eps = r1.values - (c_hat + phi_hat * r0.values)
    var_eps = float(np.var(eps, ddof=1))
    sigma_sq = var_eps * 2.0 * a_hat / (1.0 - math.exp(-2.0 * a_hat * delta_years))
    sigma_hat = math.sqrt(max(sigma_sq, 1e-12))
    a_hat = float(np.clip(a_hat, a_bounds[0], a_bounds[1]))
    sigma_hat = float(np.clip(sigma_hat, sigma_bounds[0], sigma_bounds[1]))
    return HWParams(a_hat, sigma_hat)


def pca_bps(yield_panel: pd.DataFrame, pillars: Tuple[float, ...], standardize: bool = True) -> Tuple[PCA, pd.DataFrame]:
    cols = {c: float(c.rstrip("y")) for c in yield_panel.columns}
    have = set(round(v, 6) for v in cols.values())
    targets = tuple(float(x) for x in pillars)

    # Interpolate per day via NS if pillar missing
    if not set(targets).issubset(have):
        rows = []
        for dt, row in yield_panel.iterrows():
            ts = np.array(list(cols.values()), float)
            ys = row.values.astype(float)
            m = ~np.isnan(ys)
            if m.sum() < 3:
                rows.append(pd.Series([np.nan] * len(targets), index=targets, name=dt))
            else:
                ns = NelsonSiegel(ts[m], ys[m]); ns.fit()
                rows.append(pd.Series(ns.y(np.array(targets)), index=targets, name=dt))
        Y = pd.DataFrame(rows).sort_index()
    else:
        keep = [tenor_label(p) for p in targets]
        Y = yield_panel[keep].copy(); Y.columns = list(targets)

    dY = Y.diff().dropna() * 1e4  # bp
    if standardize:
        dY = (dY - dY.mean()) / dY.std(ddof=0)
        dY = dY.dropna(how="any")
    pca = PCA(n_components=min(5, len(targets)))
    pca.fit(dY.values)
    loadings = pd.DataFrame(pca.components_, columns=[str(p) for p in targets])
    loadings.index = [f"PC{i+1}" for i in range(loadings.shape[0])]
    return pca, loadings

class BacktestResult: # (rolling, no look‑ahead) with AR(1) Δy baseline
    def __init__(self, errors_bp: pd.DataFrame, naive_bp: pd.DataFrame, ar_bp: pd.DataFrame):
        self.errors_bp = errors_bp
        self.naive_errors_bp = naive_bp
        self.ar1_errors_bp = ar_bp
        self.mae_bp = errors_bp.abs().mean()
        self.naive_mae_bp = naive_bp.abs().mean()
        self.ar1_mae_bp = ar_bp.abs().mean()


class BacktestEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _fit_curve(self, tenors: np.ndarray, yields: np.ndarray) -> NelsonSiegel:
        ns = NelsonSiegel(tenors, yields); ns.fit(); return ns

    def _pillar_series(self, panel: pd.DataFrame, T: float, start_idx: int, end_idx: int) -> pd.Series:
        """Build daily y(·,T) via NS interpolation on [start_idx:end_idx)."""
        tenors = np.array([float(c.rstrip("y")) for c in panel.columns], float)
        vals: List[float] = []
        idx = panel.index[start_idx:end_idx]
        for _, row in panel.iloc[start_idx:end_idx].iterrows():
            yv = row.values.astype(float)
            m = ~np.isnan(yv)
            if m.sum() < 3:
                vals.append(np.nan)
            else:
                ns = NelsonSiegel(tenors[m], yv[m]); ns.fit()
                vals.append(float(ns.y(T)))
        return pd.Series(vals, index=idx, name=tenor_label(T))

    def _precompute_f0_series(self, panel: pd.DataFrame) -> pd.Series:
        tenors = np.array([float(c.rstrip("y")) for c in panel.columns], float)
        out = []
        for _, row in panel.iterrows():
            yv = row.values.astype(float)
            m = ~np.isnan(yv)
            if m.sum() < 3:
                out.append(np.nan)
            else:
                ns = NelsonSiegel(tenors[m], yv[m]); ns.fit()
                out.append(float(ns.f0(0.0)))
        return pd.Series(out, index=panel.index, name="f0ns")

    def _ar1_forecast_dy(self, hist_series_bp: pd.Series) -> float:
        s = hist_series_bp.dropna()
        if len(s) < 30:
            return float(s.iloc[-1]) if len(s) else 0.0
        y0 = s.shift(1).dropna(); y1 = s.loc[y0.index]
        X = np.vstack([np.ones_like(y0.values), y0.values]).T
        c, phi = np.linalg.lstsq(X, y1.values, rcond=None)[0]
        return float(c + phi * s.iloc[-1])

    def run(self, yields_df: pd.DataFrame) -> BacktestResult:
        cfg = self.cfg
        tenors = np.array([float(c.rstrip("y")) for c in yields_df.columns], float)
        horizon_days = cfg.forecast_horizon_bdays
        delta_years = yearfrac_bdays(horizon_days)

        f0_series = None
        if cfg.ou_source == "f0ns":
            print("    [prep] Building f0(0) short‑rate series via daily NS fits…")
            f0_series = self._precompute_f0_series(yields_df)

        # Containers
        err: Dict[float, List[float]] = {p: [] for p in cfg.pillars_years}
        err_naive: Dict[float, List[float]] = {p: [] for p in cfg.pillars_years}
        err_ar: Dict[float, List[float]] = {p: [] for p in cfg.pillars_years}
        out_dates: List[pd.Timestamp] = []

        dates = yields_df.index
        i = cfg.min_history_bdays
        while i + horizon_days < len(dates):
            t0, t1 = dates[i], dates[i + horizon_days]
            hist_start = max(0, i - cfg.estimation_window_bdays)
            window_idx = slice(hist_start, i)

            # OU source on rolling window
            if cfg.ou_source == "tenor":
                short_col = tenor_label(cfg.ou_tenor_for_fit)
                if short_col not in yields_df.columns:
                    # fallback to nearest available column, else interpolate
                    try:
                        short_col = sorted(yields_df.columns, key=lambda x: float(x.rstrip("y")))[0]
                    except Exception:
                        short_col = None
                if short_col is not None and short_col in yields_df.columns:
                    sr = yields_df[short_col].iloc[window_idx]
                else:
                    sr = self._pillar_series(yields_df, cfg.ou_tenor_for_fit, hist_start, i)
            else:
                sr = f0_series.iloc[window_idx]

            params = _estimate_ou(sr, delta_years=yearfrac_bdays(1), a_bounds=cfg.a_bounds, sigma_bounds=cfg.sigma_bounds)

            # Curve at t0
            y0 = yields_df.iloc[i]
            m0 = ~np.isnan(y0.values)
            ns0 = self._fit_curve(tenors[m0], y0.values[m0])
            r0 = float(ns0.f0(0.0))
            hw = HullWhite1F(ns0, params)

            # Observed curve at t1 (for realized Δy)
            y1 = yields_df.iloc[i + horizon_days]
            m1 = ~np.isnan(y1.values)
            ns1 = self._fit_curve(tenors[m1], y1.values[m1])

            for T in cfg.pillars_years:
                y_prev = float(ns0.y(T))
                y_obs = float(ns1.y(T))
                y_hat = hw.forecast_yield_fixed_mty(r_t=r0, delta_years=delta_years, target_mty_years=T)
                dy_obs_bp = (y_obs - y_prev) * 1e4
                dy_hat_bp = (y_hat - y_prev) * 1e4
                e_bp = dy_hat_bp - dy_obs_bp
                if np.isfinite(e_bp) and abs(e_bp) <= 5000:
                    err[T].append(e_bp)
                err_naive[T].append(0.0 - dy_obs_bp)  # RW baseline
                dy_hist = (self._pillar_series(yields_df, T, hist_start, i).diff().dropna() * 1e4)
                ar_pred = self._ar1_forecast_dy(dy_hist)
                err_ar[T].append(ar_pred - dy_obs_bp)

            out_dates.append(t1)
            i += 1

        err_df = pd.DataFrame({tenor_label(T): v for T, v in err.items()}, index=pd.to_datetime(out_dates)).sort_index()
        naive_df = pd.DataFrame({tenor_label(T): v for T, v in err_naive.items()}, index=err_df.index).sort_index()
        ar_df = pd.DataFrame({tenor_label(T): v for T, v in err_ar.items()}, index=err_df.index).sort_index()
        return BacktestResult(err_df, naive_df, ar_df)


def run_selftests() -> None:
    np.random.seed(0)
    # σ→0 consistency: model price equals curve price
    ten = np.array([0.25, 2.0, 5.0, 10.0, 30.0]); yv = np.full_like(ten, 0.03, float)
    ns = NelsonSiegel(ten, yv); ns.fit()
    hw = HullWhite1F(ns, HWParams(a=0.1, sigma=1e-8))
    r0 = float(ns.f0(0.0)); T = 5.0
    assert abs(hw.zcb_price(r0, 0.0, T) - float(ns.P0(T))) < 1e-6
    # Jensen correction reduces to price at Δ=0
    assert abs(hw.expected_zcb_price(r0, 0.0, 7.0, 0.0) - hw.zcb_price(r0, 0.0, 7.0)) < 1e-12
    # Monotonicity of ZCB prices
    assert check_monotonic_prices(hw, r0, 0.0, 30.0)
    print("[selftest] OK — math sanity checks passed.")


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    cfg = Config(); np.random.seed(cfg.seed)

    run_selftests()

    print("[+] Loading FRED Treasury data…")
    data = DataLoader(cfg).fetch()
    print(f"    Loaded {data.shape[0]} business days, columns={list(data.columns)}")

    # PCA on chosen pillars
    print(f"[+] PCA on Δy (bp) — pillars={cfg.pca_pillars_years}, standardize={cfg.pca_standardize}")
    pca, loadings = pca_bps(data, cfg.pca_pillars_years, standardize=cfg.pca_standardize)
    evr = pca.explained_variance_ratio_
    print("    Variance explained:")
    for i, v in enumerate(evr[: min(5, len(evr))], 1):
        print(f"      PC{i}: {v:.2%}")
    print("    Loadings (rows=PCs, cols=pillars):\n", loadings.round(3))

    # Rolling backtest
    print("[+] Rolling 1M‑ahead Δy forecast (no look‑ahead):")
    bt = BacktestEngine(cfg).run(data)
    print("    MAE (model) by pillar (bp):\n", bt.mae_bp.round(2))
    print("    MAE (naive RW) by pillar (bp):\n", bt.naive_mae_bp.round(2))
    print("    MAE (AR1 Δy) by pillar (bp):\n", bt.ar1_mae_bp.round(2))
    edge_vs_naive = (bt.naive_mae_bp - bt.mae_bp).mean()
    edge_vs_ar1 = (bt.ar1_mae_bp - bt.mae_bp).mean()
    print(f"    Mean MAE improvement vs naive (bp): {float(edge_vs_naive):.2f}")
    print(f"    Mean MAE improvement vs AR(1) (bp): {float(edge_vs_ar1):.2f}")

    last = data.iloc[-1]
    ten = np.array([float(c.rstrip('y')) for c in data.columns], float)
    ns = NelsonSiegel(ten, last.values); ns.fit()
    # quick OU estimate on last window for a sanity HW object
    start = max(0, len(data) - cfg.estimation_window_bdays)
    if cfg.ou_source == "tenor":
        col = tenor_label(cfg.ou_tenor_for_fit) if tenor_label(cfg.ou_tenor_for_fit) in data.columns else data.columns[0]
        sr = data[col].iloc[start:]
    else:
        f0s = BacktestEngine(cfg)._precompute_f0_series(data)
        sr = f0s.iloc[start:]
    params = _estimate_ou(sr, delta_years=yearfrac_bdays(1), a_bounds=cfg.a_bounds, sigma_bounds=cfg.sigma_bounds)
    ok = check_monotonic_prices(HullWhite1F(ns, params), float(ns.f0(0.0)))
    print(f"[+] Monotonic ZCB prices today: {'OK' if ok else 'FAIL'}")



if __name__ == "__main__":
    main()
