import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from typing import Tuple, Optional, Dict, List, Callable
import warnings
import logging
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Try to import pandas_datareader for real data
try:
    from pandas_datareader import data as pdr
    DATAREADER_AVAILABLE = True
except ImportError:
    DATAREADER_AVAILABLE = False
    print("pandas_datareader not available. Install with: pip install pandas_datareader")

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YieldCurveData:
    """Fallback synthetic data generator."""
    @staticmethod
    def generate_synthetic_data(n_days: int = 1000, start_date: str = '2020-01-01') -> pd.DataFrame:
        """Generate synthetic yield curve data for testing."""
        logger.info("Generating synthetic yield curve data")
        np.random.seed(42)
        
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        maturities = ['3M', '6M', '1Y', '2Y', '5Y', '10Y', '30Y']
        base_yields = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
        
        data = []
        for i, date in enumerate(dates):
            daily_yields = []
            for j, base in enumerate(base_yields):
                # Add random walk + mean reversion
                noise = np.random.normal(0, 0.001)
                trend = 0.0001 * np.sin(2 * np.pi * i / 252)  # Annual cycle
                yield_val = max(0.001, base + trend + noise)
                daily_yields.append(yield_val)
            data.append(daily_yields)
        
        df = pd.DataFrame(data, index=dates, columns=maturities)
        return df

class RealYieldCurveData:
    """Enhanced data loader for real yield curve data from FRED and other sources."""
    
    @staticmethod
    def load_fred_data(start_date: str = '2015-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
        """Load real US Treasury yield data from FRED."""
        if not DATAREADER_AVAILABLE:
            logger.warning("pandas_datareader not available, using synthetic data")
            return YieldCurveData.generate_synthetic_data()
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        # FRED tickers for US Treasury yields
        fred_tickers = {
            '1M': 'DGS1MO',
            '3M': 'DGS3MO', 
            '6M': 'DGS6MO',
            '1Y': 'DGS1',
            '2Y': 'DGS2',
            '5Y': 'DGS5',
            '10Y': 'DGS10',
            '30Y': 'DGS30'
        }
        
        try:
            df_list = []
            for label, ticker in fred_tickers.items():
                try:
                    ts = pdr.DataReader(ticker, 'fred', start_date, end_date)
                    ts.columns = [f"{label}"]
                    df_list.append(ts)
                except Exception as e:
                    logger.warning(f"Failed to load {ticker}: {e}")
            
            if df_list:
                yield_data = pd.concat(df_list, axis=1)
                # Convert to decimal (FRED data is in %)
                yield_data = yield_data / 100.0
                # Forward fill missing values and drop rows with too many NaNs
                yield_data = yield_data.fillna(method='ffill').dropna(thresh=len(yield_data.columns)//2)
                logger.info(f"Loaded FRED data: {len(yield_data)} observations from {yield_data.index[0]} to {yield_data.index[-1]}")
                return yield_data
            else:
                raise Exception("No FRED data loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading FRED data: {e}")
            logger.info("Falling back to synthetic data")
            return YieldCurveData.generate_synthetic_data()

class NelsonSiegelFitter:
    @staticmethod
    def nelson_siegel_yield(params: np.ndarray, maturities: List[float]) -> np.ndarray:
        beta0, beta1, beta2, lambda1 = params
        T = np.array(maturities, dtype=float)
        # Avoid division by zero
        T = np.maximum(T, 1e-6)
        term1 = (1 - np.exp(-T / lambda1)) / (T / lambda1)
        term2 = term1 - np.exp(-T / lambda1)
        return beta0 + beta1 * term1 + beta2 * term2

    def fit_nelson_siegel(
        self, maturities: List[float], yields: List[float]
    ) -> Tuple[np.ndarray, float]:
        """
        Fit the four-parameter Nelson-Siegel model to (maturities, yields).
        Returns (params_array, sum_of_squared_errors).
        """
        # Convert yields list to a NumPy array so .mean() works
        y = np.array(yields, dtype=float)
        T = np.array(maturities, dtype=float)

        def objective(params: np.ndarray) -> float:
            model_yields = self.nelson_siegel_yield(params, T)
            return np.sum((model_yields - y) ** 2)

        # Initial guess: [β0, β1, β2, λ]
        initial_guess = [np.mean(y), y[0] - y[-1], 0.0, 1.0]
        bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.1, 10.0)]

        result = opt.minimize(
            objective,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )

        if not result.success:
            raise RuntimeError(f"Nelson-Siegel fit did not converge: {result.message}")

        fitted_params = result.x
        fitting_error = result.fun  # sum of squared errors

        return fitted_params, fitting_error
    def get_zero_curve_func(self, params: np.ndarray) -> Callable[[float], float]:
        """
        Given fitted NS parameters, return a function zero_curve(T) that
        computes the zero‐coupon yield for any maturity T (in years).
        """
        def zero_curve(T: float) -> float:
            beta0, beta1, beta2, lambda1 = params
            # Avoid division by zero
            t = max(T, 1e-6)
            term1 = (1 - np.exp(-t / lambda1)) / (t / lambda1)
            term2 = term1 - np.exp(-t / lambda1)
            return beta0 + beta1 * term1 + beta2 * term2
        return zero_curve

    def get_forward_curve_func(self, zero_curve_func: Callable[[float], float]) -> Callable[[float], float]:
        """
        Given a zero_curve function, return its instantaneous forward curve f(0,T)
        via f(0,T) = zero_curve(T) + T * d/dT zero_curve(T).
        We'll approximate the derivative numerically.
        """
        def forward_curve(T: float) -> float:
            h = 1e-5
            z_plus = zero_curve_func(T + h)
            z_minus = zero_curve_func(T - h)
            dz_dT = (z_plus - z_minus) / (2 * h)
            return zero_curve_func(T) + T * dz_dT
        return forward_curve



class ForwardCurveCalculator:
    """Calculate forward curves from zero-coupon curves."""
    
    @staticmethod
    def forward_rate_from_zeros(zero_curve_func: Callable, T: float, dt: float = 1e-4) -> float:
        """Calculate instantaneous forward rate f(0,T) from zero curve."""
        try:
            r_T_plus = zero_curve_func(T + dt)
            r_T_minus = zero_curve_func(max(T - dt, dt))
            
            # f(0,T) = r(T) + T * dr/dT
            dr_dT = (r_T_plus - r_T_minus) / (2 * dt)
            return zero_curve_func(T) + T * dr_dT
        except:
            return zero_curve_func(T)
    
    @staticmethod
    def create_forward_curve(zero_curve_func: Callable) -> Callable:
        """Create forward curve function from zero curve function."""
        def forward_func(T):
            return ForwardCurveCalculator.forward_rate_from_zeros(zero_curve_func, T)
        return forward_func

class CapFloorPricer:
    """Cap and Floor pricing for volatility calibration."""
    
    @staticmethod
    def caplet_price_hw(forward_rate: float, strike: float, vol: float, maturity: float, 
                       discount_factor: float) -> float:
        """Price a caplet under Hull-White (Black's formula approximation)."""
        if maturity <= 0 or vol <= 0:
            return 0.0
        
        d1 = (np.log(forward_rate / strike) + 0.5 * vol**2 * maturity) / (vol * np.sqrt(maturity))
        d2 = d1 - vol * np.sqrt(maturity)
        
        return discount_factor * (forward_rate * norm.cdf(d1) - strike * norm.cdf(d2))
    
    @staticmethod
    def cap_price_hw(forward_curve_func: Callable, zero_curve_func: Callable,
                    hw_vol: float, cap_maturity: float, strike: float, 
                    payment_freq: float = 0.25) -> float:
        """Price a cap as a strip of caplets."""
        payment_dates = np.arange(payment_freq, cap_maturity + payment_freq/2, payment_freq)
        cap_value = 0.0
        
        for T in payment_dates:
            forward_rate = forward_curve_func(T)
            discount_factor = np.exp(-zero_curve_func(T) * T)
            # Simplified HW caplet volatility (would need more complex formula in practice)
            caplet_vol = hw_vol * np.sqrt(T)
            cap_value += CapFloorPricer.caplet_price_hw(forward_rate, strike, caplet_vol, T, discount_factor)
        
        return cap_value

# -------------------------------------------------------------------
# In enhanced_ir_analyzer_draft2.py, replace the existing
# EnhancedHullWhiteModel.calibrate_to_market_data(...) with this version:
# -------------------------------------------------------------------
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import scipy.optimize as opt

class EnhancedHullWhiteModel:
    def __init__(self, a: float = 0.1, sigma: float = 0.01):
        self.a = a
        self.sigma = sigma
        self.theta_func: Optional[Callable[[float], float]] = None
        # Nelson-Siegel fitter instance
        self.ns_fitter = NelsonSiegelFitter()
        # Calibration history list (caller will append with date)
        self.calibration_history: List[Dict[str, Any]] = []

    def set_parameters(self, a: float, sigma: float) -> None:
        self.a = a
        self.sigma = sigma

    def build_theta_grid(self, forward_curve_func: Callable[[float], float]) -> None:
        """
        Given the forward curve f(0,t), build a theta(t) function on [0, T_max].
        """
        a, sigma = self.a, self.sigma

        def theta(t: float) -> float:
            # θ(t) = ∂f(0,t)/∂t + a f(0,t) + (σ^2 / (2 a)) (1 - e^{-2 a t})
            h = 1e-5
            df_dt = (forward_curve_func(t + h) - forward_curve_func(t - h)) / (2 * h)
            return df_dt + a * forward_curve_func(t) + (sigma ** 2 / (2 * a)) * (1 - np.exp(-2 * a * t))

        self.theta_func = theta

    def bond_price_analytical(self, r0: float, T: float) -> float:
        """
        Analytical bond price under one-factor HW, using the piecewise theta(t) computed above.
        """
        a, sigma = self.a, self.sigma

        # Compute B(T):
        if abs(a) < 1e-10:
            B_T = T
        else:
            B_T = (1 - np.exp(-a * T)) / a

        # Numerically integrate θ(u) from 0 to T:
        n_steps = 200
        t_grid = np.linspace(0, T, n_steps)
        theta_vals = np.array([self.theta_func(t) for t in t_grid])
        theta_integral = np.trapz(theta_vals, t_grid)

        # Compute A(T) using the standard HW formula
        if abs(a) < 1e-10:
            # In the a->0 limit
            A_T = theta_integral * T - 0.5 * sigma**2 * (T**3) / 3
        else:
            # Approximate ∫0^T θ(u) B(T-u) du via trapezoid
            B_vals = np.array([(1 - np.exp(-a * (T - u))) / a for u in t_grid])
            A_integral = np.trapz(theta_vals * B_vals, t_grid)
            A_T = A_integral - (sigma ** 2 / (4 * a)) * (B_T**2)

        return np.exp(A_T - B_T * r0)


    def simulate_paths_enhanced(
        self,
        r0: float,
        T: float,
        n_steps: int,
        n_paths: int
    ) -> np.ndarray:
        """
        Simulate one-factor HW short-rate paths by Euler discretization.
        """
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)

        r_paths = np.zeros((n_paths, n_steps + 1), dtype=float)
        r_paths[:, 0] = r0
        time_grid = np.linspace(0, T, n_steps + 1)

        for i in range(n_steps):
            t = time_grid[i]
            if self.theta_func is None:
                raise RuntimeError("build_theta_grid(...) must be called before simulate_paths_enhanced()")
            theta_t = self.theta_func(t)

            dW = np.random.normal(0.0, sqrt_dt, size=n_paths)
            r_prev = r_paths[:, i]
            r_paths[:, i + 1] = r_prev + (theta_t - self.a * r_prev) * dt + self.sigma * dW

        return r_paths

    def calibrate_to_market_data(
        self,
        maturities: List[float],
        market_yields: List[float],
        r0: float,
        vol_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate (a, sigma) to match (maturities, market_yields) plus optional cap_vol targets.
        Returns a dict: {"a":..., "sigma":..., "rmse":..., "success": True/False}.
        """

        def objective(x: np.ndarray) -> float:
            a_test = float(abs(x[0]))
            sigma_test = float(abs(x[1]))

            # 1) Fit Nelson-Siegel to current market yields
            ns_params, ns_error = self.ns_fitter.fit_nelson_siegel(
                maturities, market_yields
            )
            zero_curve_func = self.ns_fitter.get_zero_curve_func(ns_params)
            forward_curve_func = self.ns_fitter.get_forward_curve_func(zero_curve_func)

            # 2) Build θ(t) with (a_test, sigma_test)
            self.set_parameters(a_test, sigma_test)
            self.build_theta_grid(forward_curve_func)

            # 3) Compute model yields for each maturity
            model_yields = []
            for T_i in maturities:
                P_i = self.bond_price_analytical(r0, T_i)
                y_i = -np.log(P_i) / T_i
                model_yields.append(y_i)

            # 4) Compute yield-squared-error
            arr_model = np.array(model_yields)
            arr_mkt = np.array(market_yields, dtype=float)
            yield_error = np.sum((arr_model - arr_mkt) ** 2)

            # 5) If vol_data has 'cap_vols', add penalty on (sigma*sqrt(T) - marketVol)^2
            vol_error = 0.0
            if vol_data is not None and 'cap_vols' in vol_data:
                for cap_T, mkt_vol in vol_data['cap_vols'].items():
                    model_cap_vol = sigma_test * np.sqrt(cap_T)
                    vol_error += (model_cap_vol - mkt_vol) ** 2
                vol_error *= 50.0  # weight to push sigma upward

            return yield_error + vol_error

        # Initial guess & bounds
        x0 = np.array([0.1, 0.01])
        bounds = [(0.001, 2.0), (0.001, 0.5)]

        result = opt.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12}
        )

        calib: Dict[str, Any] = {
            'success': False,
            'a': None,
            'sigma': None,
            'rmse': None
        }

        if result.success:
            a_calib = abs(result.x[0])
            sigma_calib = abs(result.x[1])

            # Recompute RMSE on yields only
            ns_params_opt, _ = self.ns_fitter.fit_nelson_siegel(maturities, market_yields)
            zero_curve_opt = self.ns_fitter.get_zero_curve_func(ns_params_opt)
            forward_curve_opt = self.ns_fitter.get_forward_curve_func(zero_curve_opt)
            self.set_parameters(a_calib, sigma_calib)
            self.build_theta_grid(forward_curve_opt)

            model_yields_opt = []
            for T_i in maturities:
                P_i_opt = self.bond_price_analytical(r0, T_i)
                y_i_opt = -np.log(P_i_opt) / T_i
                model_yields_opt.append(y_i_opt)

            arr_model_opt = np.array(model_yields_opt)
            arr_mkt = np.array(market_yields, dtype=float)
            rmse_opt = np.sqrt(np.mean((arr_model_opt - arr_mkt) ** 2))

            calib.update({
                'success': True,
                'a': a_calib,
                'sigma': sigma_calib,
                'rmse': rmse_opt
            })
        else:
            calib['error'] = result.message

        # The caller will append calibration date to calib
        return calib


class TwoFactorHullWhiteModel:
    """Two-factor Hull-White (G2++) model implementation."""
    
    def __init__(self, a1: float = 0.1, a2: float = 0.3, sigma1: float = 0.01, 
                 sigma2: float = 0.015, rho: float = -0.3):
        self.a1 = a1
        self.a2 = a2
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.rho = rho
        self.theta1_grid = None
        self.theta2_grid = None
        self.time_grid = None
    
    def bond_price_analytical_2f(self, x1_0: float, x2_0: float, T: float) -> float:
        """Two-factor bond price (simplified version)."""
        if T <= 0:
            return 1.0
        
        # B coefficients
        if abs(self.a1) < 1e-10:
            B1_T = T
        else:
            B1_T = (1 - np.exp(-self.a1 * T)) / self.a1
            
        if abs(self.a2) < 1e-10:
            B2_T = T
        else:
            B2_T = (1 - np.exp(-self.a2 * T)) / self.a2
        
        # Simplified A calculation (would need theta integration in practice)
        A_T = 0.0  # Placeholder - would integrate theta terms
        
        return np.exp(A_T - B1_T * x1_0 - B2_T * x2_0)
    
    def simulate_paths_2f(self, x1_0: float, x2_0: float, T: float, 
                         n_steps: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate two-factor paths."""
        dt = T / n_steps
        sqrt_dt = np.sqrt(dt)
        
        x1_paths = np.zeros((n_paths, n_steps + 1))
        x2_paths = np.zeros((n_paths, n_steps + 1))
        x1_paths[:, 0] = x1_0
        x2_paths[:, 0] = x2_0
        
        # Correlated Brownian motions
        dW1 = np.random.normal(0, sqrt_dt, (n_paths, n_steps))
        dW2_indep = np.random.normal(0, sqrt_dt, (n_paths, n_steps))
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * dW2_indep
        
        for i in range(n_steps):
            # Simplified - would use theta grids in practice
            theta1, theta2 = 0.01, 0.01
            
            x1_paths[:, i + 1] = x1_paths[:, i] + \
                                (theta1 - self.a1 * x1_paths[:, i]) * dt + \
                                self.sigma1 * dW1[:, i]
            
            x2_paths[:, i + 1] = x2_paths[:, i] + \
                                (theta2 - self.a2 * x2_paths[:, i]) * dt + \
                                self.sigma2 * dW2[:, i]
        
        return x1_paths, x2_paths

class PCAAnalyzer:
    """Principal Component Analysis for yield curve modeling."""
    
    def __init__(self):
        self.pca = None
        self.scaler = None
        self.components = None
        self.explained_variance_ratio = None
    
    def fit_pca(self, yield_changes: pd.DataFrame, n_components: int = 3) -> Dict:
        """Fixed PCA fitting with robust data cleaning."""
        try:
            # Step 1: Remove infinite and NaN values
            data_clean = yield_changes.replace([np.inf, -np.inf], np.nan)
            data_clean = data_clean.dropna()
            
            if len(data_clean) < 10:
                return {'error': 'Insufficient clean data for PCA'}
            
            # Step 2: Remove extreme outliers using IQR method
            for col in data_clean.columns:
                Q1 = data_clean[col].quantile(0.01)
                Q3 = data_clean[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data_clean = data_clean[(data_clean[col] >= lower_bound) & (data_clean[col] <= upper_bound)]
            
            if len(data_clean) < 10:
                return {'error': 'Insufficient data after outlier removal'}
            
            # Step 3: Additional safety checks
            # Clip extreme values to reasonable bounds for yield changes (-50% to +50%)
            data_clean = data_clean.clip(-0.5, 0.5)
            
            # Check for remaining infinite or very large values
            if np.any(np.isinf(data_clean.values)) or np.any(np.abs(data_clean.values) > 1e10):
                return {'error': 'Data contains infinite or extremely large values'}
            
            # Step 4: Standardize the data with additional safety
            self.scaler = StandardScaler()
            try:
                data_scaled = self.scaler.fit_transform(data_clean)
                
                # Final check on scaled data
                if np.any(np.isnan(data_scaled)) or np.any(np.isinf(data_scaled)):
                    return {'error': 'Scaled data contains NaN or infinite values'}
                    
            except Exception as e:
                return {'error': f'Scaling failed: {str(e)}'}
            
            # Step 5: Fit PCA with error handling
            self.pca = PCA(n_components=min(n_components, data_scaled.shape[1], data_scaled.shape[0]-1))
            try:
                self.components = self.pca.fit_transform(data_scaled)
                self.explained_variance_ratio = self.pca.explained_variance_ratio_
            except Exception as e:
                return {'error': f'PCA fitting failed: {str(e)}'}
            
            # Step 6: Interpret components
            actual_components = len(self.explained_variance_ratio)
            component_names = ['Level', 'Slope', 'Curvature'][:actual_components]
            if actual_components > 3:
                component_names.extend([f'PC{i+1}' for i in range(3, actual_components)])
            
            results = {
                'explained_variance_ratio': self.explained_variance_ratio,
                'components': pd.DataFrame(
                    self.pca.components_,
                    columns=data_clean.columns,
                    index=component_names
                ),
                'factor_loadings': pd.DataFrame(
                    self.components,
                    columns=component_names,
                    index=data_clean.index
                ),
                'n_observations': len(data_clean),
                'original_shape': yield_changes.shape,
                'cleaned_shape': data_clean.shape
            }
            
            return results
            
        except Exception as e:
            logger.error(f"PCA analysis failed: {str(e)}")
            return {'error': f'PCA analysis failed: {str(e)}'}
    
    def fit_factor_processes(self, factor_loadings: pd.DataFrame) -> Dict:
        """Fit OU processes to PCA factors."""
        factor_params = {}
        
        for factor in factor_loadings.columns:
            factor_series = factor_loadings[factor].dropna()
            
            # Fit AR(1) process: x(t+1) = a*x(t) + noise
            # which gives OU parameter: mean_reversion = -log(a)/dt
            try:
                X = factor_series[:-1].values.reshape(-1, 1)
                y = factor_series[1:].values
                
                model = sm.OLS(y, sm.add_constant(X)).fit()
                a_coef = model.params[1]  # AR coefficient
                
                mean_reversion = -np.log(abs(a_coef)) if abs(a_coef) < 1 else 0.1
                volatility = np.sqrt(model.mse_resid)
                
                factor_params[factor] = {
                    'mean_reversion': mean_reversion,
                    'volatility': volatility,
                    'ar_coef': a_coef,
                    'r_squared': model.rsquared
                }
            except Exception as e:
                logger.warning(f"Error fitting {factor}: {e}")
                factor_params[factor] = {
                    'mean_reversion': 0.1,
                    'volatility': 0.01,
                    'ar_coef': 0.9,
                    'r_squared': 0.0
                }
        
        return factor_params

class MacroDataIntegrator:
    """Integrate macroeconomic data with yield curve modeling."""
    
    @staticmethod
    def load_macro_data(start_date: str = '2015-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
        """Load macroeconomic data from FRED."""
        if not DATAREADER_AVAILABLE:
            logger.warning("Cannot load macro data without pandas_datareader")
            return pd.DataFrame()
        
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        
        macro_tickers = {
            'CPI': 'CPIAUCSL',        # CPI
            'UNEMPLOYMENT': 'UNRATE',  # Unemployment Rate
            'FED_FUNDS': 'FEDFUNDS',   # Fed Funds Rate
            'GDP': 'GDP',              # GDP
            'INFLATION': 'T10YIE'      # 10Y Breakeven Inflation
        }
        
        macro_data = {}
        for name, ticker in macro_tickers.items():
            try:
                series = pdr.DataReader(ticker, 'fred', start_date, end_date)
                macro_data[name] = series.iloc[:, 0]
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
        
        if macro_data:
            df = pd.DataFrame(macro_data)
            # Forward fill and resample to daily frequency
            df = df.fillna(method='ffill').resample('D').fillna(method='ffill')
            return df
        else:
            return pd.DataFrame()
    
    @staticmethod
    def analyze_macro_yield_relationship(yield_data: pd.DataFrame, macro_data: pd.DataFrame) -> Dict:
        """Analyze relationship between macro variables and yield curve."""
        if macro_data.empty:
            return {}
        
        # Align data
        common_dates = yield_data.index.intersection(macro_data.index)
        if len(common_dates) < 50:
            logger.warning("Insufficient overlapping data for macro analysis")
            return {}
        
        yield_aligned = yield_data.loc[common_dates]
        macro_aligned = macro_data.loc[common_dates]
        
        results = {}
        
        # Analyze correlation with different maturities
        for yield_col in yield_aligned.columns:
            yield_series = yield_aligned[yield_col].dropna()
            correlations = {}
            
            for macro_col in macro_aligned.columns:
                macro_series = macro_aligned[macro_col]
                common_idx = yield_series.index.intersection(macro_series.index)
                
                if len(common_idx) > 20:
                    corr = np.corrcoef(
                        yield_series.loc[common_idx],
                        macro_series.loc[common_idx]
                    )[0, 1]
                    correlations[macro_col] = corr
            
            results[yield_col] = correlations
        
        return results

# -------------------------------------------------------------------
# In enhanced_ir_analyzer_draft2.py, replace the existing
# BacktestEngine.run_backtest(...) with this version:
# -------------------------------------------------------------------

class BacktestEngine:
    def __init__(self, model: EnhancedHullWhiteModel):
        self.model = model

    def run_backtest(
        self,
        yield_data: pd.DataFrame,
        lookback_days: int = 252,
        rebalance_freq: int = 30,
        vol_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Every `rebalance_freq` days after the first `lookback_days` days,
        fit the HW model to the prior `lookback_days` of yields, then forecast
        one-month-ahead yields for all tenors in `yield_data.columns`. Compare
        against actual one-month-ahead yields, compute MAE & RMSE. Also compute
        a naive “carry” benchmark MAE where one-month-ahead = today’s yields.
        """
        dates = yield_data.index
        n = len(dates)
        results: List[Dict[str, Any]] = []

        for i in range(lookback_days, n - rebalance_freq, rebalance_freq):
            date = dates[i]
            lookback_window = yield_data.iloc[i - lookback_days : i]
            try:
                # 1) Prepare data to fit
                train_curve = lookback_window.iloc[-1].dropna()
                maturities = [
                    float(col.replace('Y', '')) if 'Y' in col else (float(col.replace('M',''))/12)
                    for col in train_curve.index
                ]
                train_yields = list(train_curve.values)
                r0 = train_yields[0] if len(train_yields) > 0 else 0.02

                # 2) Calibrate HW (pass vol_data)
                calib = self.model.calibrate_to_market_data(
                    maturities, train_yields, r0, vol_data=vol_data
                )
                calib['calibration_date'] = date
                self.model.calibration_history.append(calib)

                # 3) Forecast one-month ahead
                # Build θ(t) using the final (a, sigma) & NS-forward from train period
                if not calib.get('success', False):
                    continue

                # Re‐fit NS on train yields to rebuild zero & forward curves
                ns_params, _ = self.model.ns_fitter.fit_nelson_siegel(maturities, train_yields)
                zero_curve_train = self.model.ns_fitter.get_zero_curve_func(ns_params)
                forward_curve_train = self.model.ns_fitter.get_forward_curve_func(zero_curve_train)

                # Set calibrated parameters and build theta
                self.model.set_parameters(calib['a'], calib['sigma'])
                self.model.build_theta_grid(forward_curve_train)

                # Now generate “predicted_yields” for each T_i (same maturities) at t = 1 month
                pred_date = dates[i + rebalance_freq]
                predicted_yields: List[float] = []
                for T_i in maturities:
                    P_pred = self.model.bond_price_analytical(r0, T_i)
                    predicted_yields.append(-np.log(P_pred) / T_i)

                # 4) Collect the actual yields on pred_date
                future_curve = yield_data.loc[pred_date].dropna()
                future_maturities = [
                    float(col.replace('Y', '')) if 'Y' in col else (float(col.replace('M',''))/12)
                    for col in future_curve.index
                ]
                actual_yields = list(future_curve.values)

                # Ensure we only compare tenors that matched exactly
                if len(maturities) != len(future_maturities):
                    # If tenor list mismatched, skip this iteration
                    continue

                arr_pred = np.array(predicted_yields)
                arr_act = np.array(actual_yields)

                # 5a) HW model errors
                mae_hw = np.mean(np.abs(arr_pred - arr_act))
                rmse_hw = np.sqrt(np.mean((arr_pred - arr_act) ** 2))

                # 5b) Naive “carry” benchmark: forecast = today’s yields
                today_yields = train_curve.values
                if len(today_yields) == len(arr_act):
                    mae_naive = np.mean(np.abs(today_yields - arr_act))
                else:
                    mae_naive = None

                # 6) Append results
                results.append({
                    'date': date,
                    'future_date': pred_date,
                    'mae_hw': mae_hw,
                    'rmse_hw': rmse_hw,
                    'mae_naive': mae_naive,
                    'model_params': {'a': calib['a'], 'sigma': calib['sigma']}
                })

            except Exception as e:
                # Skip any iteration that fails
                continue

        # Compute aggregate metrics
        all_mae_hw = [r['mae_hw'] for r in results if r.get('mae_hw') is not None]
        all_rmse_hw = [r['rmse_hw'] for r in results if r.get('rmse_hw') is not None]
        all_mae_naive = [r['mae_naive'] for r in results if r.get('mae_naive') is not None]

        avg_mae_hw = np.mean(all_mae_hw)    if all_mae_hw    else None
        avg_rmse_hw = np.mean(all_rmse_hw)  if all_rmse_hw   else None
        avg_mae_naive = np.mean(all_mae_naive) if all_mae_naive else None

        return {
            'success': True,
            'num_predictions': len(results),
            'avg_mae_hw': avg_mae_hw,
            'avg_rmse_hw': avg_rmse_hw,
            'avg_mae_naive': avg_mae_naive,
            'detailed_results': results
        }


class AdvancedAnalytics:
    """Advanced analytics and risk metrics."""
    
    @staticmethod
    def calculate_duration_convexity(
        zero_curve_func: Callable[[float], float],
        maturity: float,
        dy: float = 1e-4
    ) -> Tuple[float, float]:
        """
        Compute modified duration and convexity for a zero-coupon bond using finite differences.
        """
        y = zero_curve_func(maturity)

        # Price at current yield
        P0 = np.exp(-y * maturity)

        # Prices at bumped yields (manually perturbed)
        P_up = np.exp(-(y + dy) * maturity)
        P_down = np.exp(-(y - dy) * maturity)

        # Modified duration
        duration = (P_down - P_up) / (2 * P0 * dy)

        # Convexity
        convexity = (P_up + P_down - 2 * P0) / (P0 * dy**2)

        return duration, convexity


    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns: np.ndarray, confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        if len(returns) == 0:
            return 0.0
        var = AdvancedAnalytics.calculate_var(returns, confidence_level)
        return np.mean(returns[returns <= var])
    
    @staticmethod
    def stress_test_scenarios(model: EnhancedHullWhiteModel, r0: float, 
                             scenarios: Dict[str, Dict]) -> Dict:
        """Run stress test scenarios."""
        results = {}
        
        for scenario_name, params in scenarios.items():
            try:
                # Temporary parameter change
                original_a, original_sigma = model.a, model.sigma
                
                # Apply stress scenario
                stressed_a = original_a * params.get('a_multiplier', 1.0)
                stressed_sigma = original_sigma * params.get('sigma_multiplier', 1.0)
                
                model.set_parameters(stressed_a, stressed_sigma)
                
                # Calculate stressed bond prices for different maturities
                maturities = [0.25, 0.5, 1, 2, 5, 10, 30]
                stressed_prices = []
                
                for T in maturities:
                    price = model.bond_price_analytical(r0, T)
                    stressed_prices.append(price)
                
                results[scenario_name] = {
                    'maturities': maturities,
                    'stressed_prices': stressed_prices,
                    'stressed_params': {'a': stressed_a, 'sigma': stressed_sigma}
                }
                
                # Restore original parameters
                model.set_parameters(original_a, original_sigma)
                
            except Exception as e:
                logger.error(f"Error in stress scenario {scenario_name}: {e}")
                results[scenario_name] = {'error': str(e)}
        
        return results

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Callable
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

# -------------------------------------------------------------------
# Assume the following classes are defined elsewhere in your file:
#   - NelsonSiegelFitter
#   - EnhancedHullWhiteModel
#   - PCAAnalyzer
#   - BacktestEngine
#   - AdvancedAnalytics
#   - MacroDataIntegrator
# -------------------------------------------------------------------

class ComprehensiveIRAnalyzer:
    def __init__(self, use_real_data: bool = True):
        self.use_real_data = use_real_data
        self.yield_data: pd.DataFrame = pd.DataFrame()
        self.ns_fitter = NelsonSiegelFitter()
        self.hw_model = EnhancedHullWhiteModel()
        self.pca_analyzer = PCAAnalyzer()
        self.backtest_engine = BacktestEngine(self.hw_model)
        self.macro_data: pd.DataFrame = pd.DataFrame()  # if you have macro inputs
        self.analysis_results: Dict[str, Any] = {}


    def load_data(self, use_real_data: bool, start_date: str = '2020-01-01') -> None:
        """
        Load yield curve data into self.yield_data.
        If use_real_data=True, attempt to fetch from FRED via RealYieldCurveData.
        Otherwise, or on failure, fall back to synthetic data.
        
        Parameters:
        -----------
        use_real_data : bool
            If True, attempt to load real Treasury yield data from FRED.
            If False, skip real data and generate synthetic data.
        start_date : str
            YYYY-MM-DD string for the first date of data. Used by both real and synthetic.
        """
        if use_real_data:
            try:
                # Determine end_date as today (or last business day)
                today = datetime.today().strftime('%Y-%m-%d')
                # Attempt to load via RealYieldCurveData
                self.yield_data = RealYieldCurveData.load_fred_data(
                    start_date=start_date,
                    end_date=today
                )
                if self.yield_data is None or self.yield_data.empty:
                    raise ValueError("RealYieldCurveData returned no data.")
                print(f"Loaded real data from {start_date} to {today}.")
                return
            except Exception as e:
                # On any failure, warn and fall back to synthetic
                print(f"WARNING: Failed to load real data: {e}")
                print("Loading synthetic data instead...")
        
        # --- Fallback to synthetic data ---
        # We want roughly the same date range length as if from FRED.
        # For simplicity, generate synthetic for 1300 days (≈ 5 years).
        n_days = 1300
        try:
            self.yield_data = YieldCurveData.generate_synthetic_data(
                n_days=n_days,
                start_date=start_date
            )
            print(f"Generated synthetic data for {n_days} days starting {start_date}.")
        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic yield data: {e}")

    def enhanced_data_cleaning(self) -> None:
        """
        Loads (or generates) yield data, then performs any necessary
        cleaning (forward/backfill, clipping, drop too‐sparse columns/rows).
        Replace this placeholder with your actual loading/cleaning logic.
        """
        # Example placeholder: load synthetic or real data into self.yield_data
        # ...
        pass

    def plot_calibration_history(self) -> None:
        """
        Plot how the calibrated (a, sigma) evolved over each calibration date.
        """
        # Filter only successful calibrations that have a 'calibration_date'
        records = [
            {'date': entry['calibration_date'], 'a': entry['a'], 'sigma': entry['sigma']}
            for entry in self.hw_model.calibration_history
            if entry.get('success', False) and 'calibration_date' in entry
        ]
        if not records:
            print("No successful HW calibration history to plot.")
            return

        df_calib = pd.DataFrame(records).set_index('date').sort_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        ax1.plot(df_calib.index, df_calib['a'], marker='o', linestyle='-')
        ax1.set_title("HW Calibration: Mean Reversion (a) Over Time")
        ax1.set_ylabel("a")
        ax1.grid(True, alpha=0.3)

        ax2.plot(df_calib.index, df_calib['sigma'], marker='o', linestyle='-', color='orange')
        ax2.set_title("HW Calibration: Volatility (σ) Over Time")
        ax2.set_ylabel("σ")
        ax2.set_xlabel("Calibration Date")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Main pipeline:
           0) Data cleaning
           1) Fit Nelson-Siegel on latest curve
           2) Calibrate 1F Hull-White with cap-vol targets (store market/model yields)
           3) PCA on daily % changes (sorted tenors), fit OU
           4) (Optional) Macro analysis
           5) Risk metrics (duration/convexity, VaR/ES)
           6) Stress testing
           7) Monte Carlo simulation (store full paths)
           8) Backtesting (with naive carry benchmark)
           9) Store all results, plot calibration history
        """
        results: Dict[str, Any] = {}

        # --------------------
        # STEP 0: Load & clean data
        # --------------------
        self.enhanced_data_cleaning()  # your own cleaning logic populates self.yield_data

        if self.yield_data.empty:
            raise ValueError("No yield data available after cleaning.")

        # --------------------
        # STEP 1: Current Yield Curve & NS Fit
        # --------------------
        latest_date = self.yield_data.index[-1]
        latest_curve = self.yield_data.loc[latest_date].dropna()

        # Convert column names to numeric maturities
        maturities: List[float] = []
        for col in latest_curve.index:
            if 'M' in col:
                maturities.append(float(col.replace('M', '')) / 12.0)
            elif 'Y' in col:
                maturities.append(float(col.replace('Y', '')))
            else:
                continue
        market_yields = list(latest_curve.values)

        # Fit Nelson-Siegel
        ns_params, ns_error = self.ns_fitter.fit_nelson_siegel(maturities, market_yields)
        zero_curve_func = self.ns_fitter.get_zero_curve_func(ns_params)
        forward_curve_func = self.ns_fitter.get_forward_curve_func(zero_curve_func)

        results['nelson_siegel'] = {
            'parameters': {
                'beta0': ns_params[0],
                'beta1': ns_params[1],
                'beta2': ns_params[2],
                'lambda': ns_params[3]
            },
            'fitting_error': ns_error
        }

        # --------------------
        # STEP 2: Hull-White Calibration (with cap-vol targets)
        # --------------------
        r0 = market_yields[0] if len(market_yields) > 0 else 0.02

        vol_data = {
            'cap_vols': {
                2.0: 0.12,  # 2Y ATM cap vol = 12%
                5.0: 0.10,  # 5Y ATM cap vol = 10%
                10.0: 0.09  # 10Y ATM cap vol = 9%
            }
        }

        hw_calib = self.hw_model.calibrate_to_market_data(
            maturities, market_yields, r0, vol_data=vol_data
        )
        # If calibration succeeded, compute the model's fitted yields for these maturities
        if hw_calib.get('success', False):
            a_cal, sigma_cal = hw_calib['a'], hw_calib['sigma']
            self.hw_model.set_parameters(a_cal, sigma_cal)
            self.hw_model.build_theta_grid(forward_curve_func)

            model_yields: List[float] = []
            for T_i in maturities:
                P_i = self.hw_model.bond_price_analytical(r0, T_i)
                y_i = -np.log(P_i) / T_i
                model_yields.append(y_i)
        else:
            model_yields = []

        # Store raw market/model yields and date in hw_calib
        hw_calib['maturities']     = maturities
        hw_calib['market_yields']  = market_yields
        hw_calib['model_yields']   = model_yields
        hw_calib['calibration_date'] = latest_date
        self.hw_model.calibration_history.append(hw_calib)

        results['hull_white_calibration'] = hw_calib

        # --------------------
        # STEP 3: PCA on yield‐curve changes (sorted tenors)
        # --------------------
        pca_info: Dict[str, Any] = {}
        try:
            if len(self.yield_data) > 50:
                maturity_map = {
                    '3M': 0.25, '6M': 0.5, '1Y': 1.0,
                    '2Y': 2.0, '5Y': 5.0, '10Y': 10.0, '30Y': 30.0
                }
                available_cols = [c for c in self.yield_data.columns if c in maturity_map]
                available_cols_sorted = sorted(available_cols, key=lambda c: maturity_map[c])
                self.yield_data = self.yield_data[available_cols_sorted]

                yield_changes = self.yield_data.pct_change().dropna()
                yield_changes = yield_changes.replace([np.inf, -np.inf], np.nan).dropna()

                lower_q = yield_changes.quantile(0.01)
                upper_q = yield_changes.quantile(0.99)
                yield_changes_clipped = yield_changes.clip(lower=lower_q, upper=upper_q, axis=1)

                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(yield_changes_clipped.values)

                pca = PCA(n_components=3)
                pca.fit(scaled_data)
                loadings = pca.components_.T
                explained = pca.explained_variance_ratio_

                pcs = pca.transform(scaled_data)
                ou_params = []
                for i in range(3):
                    series = pcs[:, i]
                    phi = np.polyfit(series[:-1], series[1:], 1)[0]
                    kappa = -np.log(abs(phi))
                    residuals = series[1:] - phi * series[:-1]
                    sigma_ou = np.std(residuals) * np.sqrt(252)
                    ou_params.append({'kappa': kappa, 'sigma': sigma_ou})

                pca_info = {
                    'loadings': loadings,
                    'explained_variance_ratio': explained,
                    'ou_parameters': ou_params
                }
            else:
                pca_info = {'error': 'Not enough data for PCA'}
        except Exception as e:
            pca_info = {'error': str(e)}

        results['pca_analysis'] = pca_info

        # --------------------
        # STEP 4: Macro Analysis (if available)
        # --------------------
        if not self.macro_data.empty:
            try:
                macro_yield_corr = MacroDataIntegrator.analyze_macro_yield_relationship(
                    self.yield_data, self.macro_data
                )
                results['macro_analysis'] = macro_yield_corr
            except Exception as e:
                results['macro_analysis'] = {'error': str(e)}

        # --------------------
        # STEP 5: Risk Metrics (Duration, Convexity, VaR, ES)
        # --------------------
        try:
            duration, convexity = AdvancedAnalytics.calculate_duration_convexity(
                zero_curve_func, 10.0
            )

            if '10Y' in self.yield_data.columns:
                return_series = self.yield_data['10Y'].pct_change().dropna()
            else:
                return_series = self.yield_data[self.yield_data.columns[0]].pct_change().dropna()

            return_series = return_series.replace([np.inf, -np.inf], np.nan).dropna()
            return_series = return_series.clip(-0.5, 0.5)

            if len(return_series) > 10:
                var_95 = AdvancedAnalytics.calculate_var(return_series.values, 0.05)
                es_95  = AdvancedAnalytics.calculate_expected_shortfall(return_series.values, 0.05)
            else:
                var_95 = es_95 = 0.0

            results['risk_metrics'] = {
                'duration_10Y': duration,
                'convexity_10Y': convexity,
                'var_95': var_95,
                'expected_shortfall_95': es_95
            }
        except Exception as e:
            results['risk_metrics'] = {'error': str(e)}

        # --------------------
        # STEP 6: Stress Testing
        # --------------------
        try:
            stress_scenarios = {
                'rate_shock_up':       {'a_multiplier': 1.0, 'sigma_multiplier': 1.5},
                'rate_shock_down':     {'a_multiplier': 1.0, 'sigma_multiplier': 0.5},
                'volatility_spike':    {'a_multiplier': 0.5, 'sigma_multiplier': 2.0},
                'mean_reversion_slow': {'a_multiplier': 0.3, 'sigma_multiplier': 1.0}
            }
            stress_results = AdvancedAnalytics.stress_test_scenarios(
                self.hw_model, r0, stress_scenarios
            )
            results['stress_tests'] = stress_results
        except Exception as e:
            results['stress_tests'] = {'error': str(e)}

        # --------------------
        # STEP 7: Monte Carlo Simulation (1-year horizon)
        # --------------------
        try:
            sim_info: Dict[str, Any] = {}
            if hw_calib.get('success', False):
                # Simulate 1 year (252 steps) with 500 paths
                n_steps = 252
                n_paths = 500
                T = 1.0

                # Ensure your model has a simulate_paths_enhanced method
                sample_paths = self.hw_model.simulate_paths_enhanced(r0, T, n_steps, n_paths)
                final_rates = sample_paths[:, -1]
                stats = {
                    'mean': np.mean(final_rates),
                    'std': np.std(final_rates),
                    'percentiles': np.percentile(final_rates, [5, 25, 50, 75, 95])
                }
                sim_info['paths'] = sample_paths
                sim_info['final_rates_stats'] = stats
            else:
                sim_info['error'] = 'HW calibration failed; cannot simulate.'

            results['simulation'] = sim_info
        except Exception as e:
            results['simulation'] = {'error': str(e)}

        # --------------------
        # STEP 8: Backtesting (with naive carry benchmark)
        # --------------------
        backtest_metrics: Dict[str, Any] = {}
        try:
            backtest_metrics = self.backtest_engine.run_backtest(
                self.yield_data,
                lookback_days=252,
                rebalance_freq=30,
                vol_data=vol_data
            )
        except Exception as e:
            backtest_metrics = {'success': False, 'error': str(e)}

        results['backtesting'] = backtest_metrics

        # --------------------
        # STEP 9: Store & Plot
        # --------------------
        self.analysis_results = results
        self.plot_calibration_history()

        return results


    def generate_report(self) -> str:
        """
        Generate a text report of all analysis_results.
        """
        if not self.analysis_results:
            return "No analysis results available. Run run_comprehensive_analysis() first."

        report_lines: List[str] = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE INTEREST RATE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data Period: {self.yield_data.index[0].strftime('%Y-%m-%d')} to {self.yield_data.index[-1].strftime('%Y-%m-%d')}")
        report_lines.append("")

        # --- Nelson-Siegel ---
        ns = self.analysis_results.get('nelson_siegel', {})
        if ns:
            report_lines.append("NELSON-SIEGEL CURVE FITTING")
            report_lines.append("-" * 40)
            betas = ns['parameters']
            report_lines.append(f"Beta0 (Level):    {betas['beta0']:.6f}")
            report_lines.append(f"Beta1 (Slope):    {betas['beta1']:.6f}")
            report_lines.append(f"Beta2 (Curvature): {betas['beta2']:.6f}")
            report_lines.append(f"Lambda:            {betas['lambda']:.6f}")
            report_lines.append(f"Fitting Error:     {ns['fitting_error']:.8f}")
            report_lines.append("")

        # --- Hull-White Calibration ---
        hw = self.analysis_results.get('hull_white_calibration', {})
        if hw:
            report_lines.append("HULL-WHITE MODEL CALIBRATION")
            report_lines.append("-" * 40)
            if hw.get('success', False):
                report_lines.append(f"Mean Reversion (a): {hw['a']:.6f}")
                report_lines.append(f"Volatility (σ):     {hw['sigma']:.6f}")
                report_lines.append(f"Calibration RMSE:   {hw['rmse']:.6f}")
            else:
                report_lines.append("Calibration failed")
            report_lines.append("")

        # --- PCA ---
        pca = self.analysis_results.get('pca_analysis', {})
        if pca and 'loadings' in pca:
            report_lines.append("PRINCIPAL COMPONENT ANALYSIS")
            report_lines.append("-" * 40)
            explained = pca['explained_variance_ratio']
            comps = ['Level', 'Slope', 'Curvature']
            for i, var_ratio in enumerate(explained):
                name = comps[i] if i < len(comps) else f"PC{i+1}"
                report_lines.append(f"{name}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
            report_lines.append(f"Total Variance Explained: {sum(explained):.4f}")
            report_lines.append("")

        # --- Macro (if any) ---
        macro = self.analysis_results.get('macro_analysis', {})
        if macro:
            report_lines.append("MACRO ANALYSIS")
            report_lines.append("-" * 40)
            report_lines.append(str(macro))
            report_lines.append("")

        # --- Risk Metrics ---
        risk = self.analysis_results.get('risk_metrics', {})
        if risk:
            report_lines.append("RISK METRICS")
            report_lines.append("-" * 40)
            if 'duration_10Y' in risk:
                report_lines.append(f"10Y Duration:            {risk['duration_10Y']:.4f}")
                report_lines.append(f"10Y Convexity:           {risk['convexity_10Y']:.4f}")
                report_lines.append(f"95% VaR:                 {risk['var_95']:.6f}")
                report_lines.append(f"95% Expected Shortfall:  {risk['expected_shortfall_95']:.6f}")
            else:
                report_lines.append(str(risk.get('error', 'Risk metrics missing')))
            report_lines.append("")

        # --- Stress Tests ---
        stress = self.analysis_results.get('stress_tests', {})
        if stress:
            report_lines.append("STRESS TEST RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(str(stress))
            report_lines.append("")

        # --- Simulation ---
        sim = self.analysis_results.get('simulation', {})
        if sim and 'final_rates_stats' in sim:
            stats = sim['final_rates_stats']
            report_lines.append("MONTE CARLO SIMULATION (1Y horizon)")
            report_lines.append("-" * 40)
            report_lines.append(f"Mean Final Rate:    {stats['mean']:.4f}")
            report_lines.append(f"Standard Deviation: {stats['std']:.4f}")
            report_lines.append(f"5th Percentile:     {stats['percentiles'][0]:.4f}")
            report_lines.append(f"95th Percentile:    {stats['percentiles'][4]:.4f}")
            report_lines.append("")
        elif sim:
            report_lines.append("MONTE CARLO SIMULATION")
            report_lines.append("-" * 40)
            report_lines.append(str(sim.get('error', 'No simulation results')))
            report_lines.append("")

        # --- Backtesting ---
        bt = self.analysis_results.get('backtesting', {})
        if bt.get('success', False):
            report_lines.append("BACKTESTING RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Number of Predictions:           {bt['num_predictions']}")
            report_lines.append(f"HW Model Average MAE (1-month):  {bt['avg_mae_hw']:.6f}")
            report_lines.append(f"Naive Carry Average MAE (1-month): {bt['avg_mae_naive']:.6f}")
            report_lines.append(f"HW Model Average RMSE (1-month): {bt['avg_rmse_hw']:.6f}")
            report_lines.append("")
        else:
            report_lines.append("BACKTESTING: Error or no results.")
            if 'error' in bt:
                report_lines.append(str(bt['error']))
            report_lines.append("")

        report_lines.append("=" * 80)
        return "\n".join(report_lines)

    def plot_results(self) -> None:
        """
        Generate two figures:
          1) A 2×3 grid containing five subplots:
             (1) Current yield curve + NS fit
             (2) Yield curve evolution (2Y, 10Y, 30Y)
             (3) PCA component loadings
             (4) Hull-White simulation paths
             (5) Model fit quality (market vs model yields)
             (6) (Intentionally left blank)
          2) A separate figure for Stress Test Results.
        """
        if not self.analysis_results:
            print("No analysis results to plot.")
            return

        # ------------------------------------------------------------
        # FIGURE 1: Five‐panel grid (2×3) WITHOUT the stress test chart.
        # ------------------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Interest Rate Analysis', fontsize=16)

        # ----------------------------------------
        # (1) Current Yield Curve + NS Fit
        # ----------------------------------------
        ax1 = axes[0, 0]
        latest_yields = pd.Series(self.yield_data.iloc[-1]).dropna()
        maturities_pts: List[float] = []
        yields_pts: List[float] = []
        for col in latest_yields.index:
            if 'M' in col:
                maturities_pts.append(float(col.replace('M', '')) / 12.0)
            elif 'Y' in col:
                maturities_pts.append(float(col.replace('Y', '')))
            else:
                continue
            yields_pts.append(latest_yields[col])

        ax1.scatter(maturities_pts, yields_pts, color='red', label='Market', s=50)

        ns_params = self.analysis_results['nelson_siegel']['parameters']
        beta0 = ns_params['beta0']
        beta1 = ns_params['beta1']
        beta2 = ns_params['beta2']
        lam   = ns_params['lambda']
        T_plot = np.linspace(0.1, 30, 100)
        ns_yields_plot = NelsonSiegelFitter.nelson_siegel_yield(
            np.array([beta0, beta1, beta2, lam]), T_plot
        )
        ax1.plot(T_plot, ns_yields_plot, 'b-', label='Nelson-Siegel Fit')

        ax1.set_xlabel('Maturity (Years)')
        ax1.set_ylabel('Yield')
        ax1.set_title('Current Yield Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ----------------------------------------
        # (2) Yield Curve Evolution (2Y, 10Y, 30Y)
        # ----------------------------------------
        ax2 = axes[0, 1]
        plot_cols = ['2Y', '10Y', '30Y']
        existing_cols = [c for c in plot_cols if c in self.yield_data.columns]
        if len(existing_cols) < 3:
            existing_cols = list(self.yield_data.columns[:3])

        for col in existing_cols:
            ax2.plot(self.yield_data.index, self.yield_data[col], label=col, alpha=0.8)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Yield')
        ax2.set_title('Yield Curve Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # ----------------------------------------
        # (3) PCA Component Loadings
        # ----------------------------------------
        ax3 = axes[0, 2]
        pca_info = self.analysis_results.get('pca_analysis', {})
        if 'loadings' in pca_info:
            loadings = pca_info['loadings']       # shape = (n_tenors, 3)
            explained = pca_info['explained_variance_ratio']
            numeric_mats: List[float] = []
            for col in self.yield_data.columns:
                if 'M' in col:
                    numeric_mats.append(float(col.replace('M', '')) / 12.0)
                elif 'Y' in col:
                    numeric_mats.append(float(col.replace('Y', '')))
                else:
                    numeric_mats.append(np.nan)

            labels = ['Level', 'Slope', 'Curvature']
            for i in range(min(3, loadings.shape[1])):
                ax3.plot(
                    numeric_mats,
                    loadings[:, i],
                    'o-',
                    label=f"{labels[i]} ({explained[i]*100:.1f}%)"
                )

            ax3.set_xlabel('Maturity (Years)')
            ax3.set_ylabel('Component Loading')
            ax3.set_title('PCA Components')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, "No PCA data", ha='center', va='center')
            ax3.set_axis_off()

        # ----------------------------------------
        # (4) Hull-White Simulation Paths
        # ----------------------------------------
        ax4 = axes[1, 0]
        sim_info = self.analysis_results.get('simulation', {})
        if 'paths' in sim_info:
            sample_paths = sim_info['paths']  # shape = (n_paths, n_steps + 1)
            n_paths, n_time = sample_paths.shape
            n_plot = min(10, n_paths)

            # Reconstruct one-year time axis
            time_grid = np.linspace(0, 1.0, n_time)

            cmap = plt.get_cmap('tab10')
            for i in range(n_plot):
                color = cmap(i % 10)
                ax4.plot(time_grid, sample_paths[i, :], color=color, alpha=0.6)

            # Plot initial rate
            r0 = sample_paths[0, 0]
            ax4.axhline(y=r0, color='red', linestyle='--', label='Initial Rate')

            ax4.set_xlabel('Time (Years)')
            ax4.set_ylabel('Short Rate')
            ax4.set_title('Hull-White Simulation Paths')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "No simulation data", ha='center', va='center')
            ax4.set_axis_off()

        # ----------------------------------------
        # (5) Model Fit Quality (Market vs Model Yields)
        # ----------------------------------------
        ax5 = axes[1, 1]
        hw_calib = self.analysis_results.get('hull_white_calibration', {})
        if hw_calib.get('success', False):
            market_yields = hw_calib.get('market_yields', [])
            model_yields  = hw_calib.get('model_yields', [])
            if len(market_yields) == len(model_yields) and len(market_yields) > 0:
                ax5.scatter(market_yields, model_yields, alpha=0.7, s=50)

                min_y = min(min(market_yields), min(model_yields))
                max_y = max(max(market_yields), max(model_yields))
                ax5.plot([min_y, max_y], [min_y, max_y], 'r--', label='Perfect Fit')

                rmse_val = hw_calib.get('rmse', 0.0)
                ax5.set_title(f'Model Fit Quality (RMSE: {rmse_val:.4f})')
                ax5.set_xlabel('Market Yields')
                ax5.set_ylabel('Model Yields')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, "Market/model yields missing", ha='center', va='center')
                ax5.set_axis_off()
        else:
            ax5.text(0.5, 0.5, "HW calibration failed", ha='center', va='center')
            ax5.set_axis_off()

        # ----------------------------------------------------------------
        # (6) Leave bottom-right subplot intentionally blank
        # ----------------------------------------------------------------
        ax6 = axes[1, 2]
        ax6.set_axis_off()

        plt.tight_layout()
        plt.show()

        # ------------------------------------------------------------
        # FIGURE 2: Stress Test Results in a separate figure
        # ------------------------------------------------------------
        stress_info = self.analysis_results.get('stress_tests', {})
        if isinstance(stress_info, dict) and stress_info:
            # We want to plot in this order:
            desired_order = [
                'mean_reversion_slow',
                'volatility_spike',
                'rate_shock_down',
                'rate_shock_up'
            ]
            scenario_names: List[str] = []
            price_changes:    List[float] = []
            for key in desired_order:
                data = stress_info.get(key, None)
                if data and 'stressed_prices' in data:
                    avg_price = np.mean(data['stressed_prices'])
                    pct_change = (avg_price - 1.0) * 100.0
                    scenario_names.append(key.replace('_', ' ').title())
                    price_changes.append(pct_change)

            # Plot in its own figure
            fig2, ax_s = plt.subplots(figsize=(8, 6))
            colors = ['red' if pc < 0 else 'green' for pc in price_changes]
            ax_s.barh(scenario_names, price_changes, color=colors, alpha=0.7)
            ax_s.set_xlabel('Average Price Change (%)')
            ax_s.set_title('Stress Test Results')
            ax_s.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            print("No stress test data to plot.")



def main():
    """Main execution function with enhanced error handling."""
    logger.info("Starting Comprehensive Interest Rate Analysis")
    
    # Initialize analyzer
    analyzer = ComprehensiveIRAnalyzer()
    
    # Load data (try real data first, fall back to synthetic)
    try:
        logger.info("Attempting to load real data...")
        analyzer.load_data(use_real_data=True, start_date='2020-01-01')
        logger.info("Real data loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load real data: {e}")
        logger.info("Loading synthetic data as fallback...")
        analyzer.load_data(use_real_data=False)
        logger.info("Synthetic data loaded successfully")
    
    # Run comprehensive analysis with detailed error handling
    try:
        logger.info("Starting comprehensive analysis...")
        results = analyzer.run_comprehensive_analysis()
        
        # Check if we have any successful results
        successful_analyses = [k for k, v in results.items() if not (isinstance(v, dict) and 'error' in v)]
        logger.info(f"Successful analyses: {successful_analyses}")
        
        if not successful_analyses:
            logger.error("No analyses completed successfully")
            return
        
        # Generate and print report
        report = analyzer.generate_report()
        print(report)
        
        # Create plots (with error handling)
        try:
            analyzer.plot_results()
        except Exception as e:
            logger.warning(f"Plotting failed: {e}")
        
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to provide some diagnostic information
        if hasattr(analyzer, 'yield_data') and analyzer.yield_data is not None:
            logger.info(f"Data shape: {analyzer.yield_data.shape}")
            logger.info(f"Data columns: {list(analyzer.yield_data.columns)}")
            logger.info(f"Data index range: {analyzer.yield_data.index[0]} to {analyzer.yield_data.index[-1]}")
            logger.info(f"Data statistics:\n{analyzer.yield_data.describe()}")
        
        raise
if __name__ == "__main__":
    main()