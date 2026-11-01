import argparse
from pathlib import Path
import numpy as np
import pandas as pd

D_DEFAULT = 100000
T_DEFAULT = 100
START_DATE_DEFAULT = "2015-01-01"
RANDOM_SEED_DEFAULT = 42

# =========================================================================
# MATHEMATICAL GUIDE: generate_synthetic_stock_data
# MATHEMATICAL MODEL (synthetic data, NOT the fitted optimization model)
# For asset j and return step t, generate a clipped log return
#  r[j,t] = mu[j] + Delta[j]*1{t >= tau[j]}
#           + beta_market[j]*r_market[t]
#           + beta_sector[j]*r_sector[sector[j],t]
#           + sigma_idio[j]*v_idio[j,t]*epsilon[j,t].
# Here r_market[t] = mu_market + sigma_market*v_market[t]*epsilon_market[t];
# sector returns are independent N(0,sigma_sector^2) draws. Both v factors
# are autoregressive functions of the preceding absolute innovation.
# Prices: Y[j,0]=P0[j], Y[j,t]=P0[j]*exp(sum_{s=0}^{t-1} r[j,s]).
# The generated data are therefore NOT exactly piecewise-linear in price,
# and tau[j] is ASSET-SPECIFIC, whereas the fitted optimization assumes one
# SHARED breakpoint z across selected dimensions. This is intentional model
# misspecification, important when interpreting recovered z.
# Array convention: Y has shape (J,T), x has shape (T,).
# =========================================================================
def generate_synthetic_stock_data(
    D=D_DEFAULT,
    T=T_DEFAULT,
    start_date=START_DATE_DEFAULT,
    random_seed=RANDOM_SEED_DEFAULT,
    number_sectors=20,
    initial_price_low=20.0,
    initial_price_high=250.0,
    annual_market_drift=0.08,
    annual_market_volatility=0.18,
    annual_sector_volatility=0.10,
    annual_idiosyncratic_volatility=0.16,
    change_probability=0.70,
    change_magnitude=0.0015,
):
    """Generate D correlated positive stock-like price series of length T."""
    if D <= 0 or T < 6:
        raise ValueError("Require D > 0 and T >= 6.")
    if number_sectors <= 0:
        raise ValueError("number_sectors must be positive.")

    rng = np.random.default_rng(random_seed)
    trading_days = 252.0

    market_daily_drift = annual_market_drift / trading_days
    market_daily_volatility = annual_market_volatility / np.sqrt(trading_days)
    sector_daily_volatility = annual_sector_volatility / np.sqrt(trading_days)
    idio_daily_volatility = annual_idiosyncratic_volatility / np.sqrt(trading_days)

    dates = pd.bdate_range(start=pd.Timestamp(start_date), periods=T)
    x = np.arange(T, dtype=np.float64)

    market_shocks = rng.normal(size=T - 1)
    market_vol_scale = np.ones(T - 1, dtype=np.float64)
    for t in range(1, T - 1):
        market_vol_scale[t] = (
            0.90
            + 0.08 * market_vol_scale[t - 1]
            + 0.12 * min(abs(market_shocks[t - 1]), 3.0)
        )

    market_returns = (
        market_daily_drift
        + market_daily_volatility * market_vol_scale * market_shocks
    )

    sector_returns = rng.normal(
        loc=0.0,
        scale=sector_daily_volatility,
        size=(number_sectors, T - 1),
    )

    sector_id = rng.integers(0, number_sectors, size=D)
    market_beta = np.clip(rng.normal(1.0, 0.25, size=D), 0.25, 2.0)
    sector_beta = np.clip(rng.normal(0.70, 0.20, size=D), 0.10, 1.50)
    idio_scale = np.clip(
        rng.lognormal(mean=np.log(idio_daily_volatility), sigma=0.25, size=D),
        0.002,
        0.050,
    )
    stock_daily_drift = rng.normal(
        loc=0.02 / trading_days,
        scale=0.06 / trading_days,
        size=D,
    )
    initial_prices = rng.uniform(initial_price_low, initial_price_high, size=D)

    has_change = rng.random(D) < change_probability
    center = 0.52 * (T - 1)
    spread = max(2.0, 0.10 * T)
    change_index = np.rint(rng.normal(center, spread, size=D)).astype(np.int64)
    change_index = np.clip(change_index, 2, T - 3)

    drift_change = rng.normal(loc=0.0, scale=change_magnitude, size=D)
    drift_change[~has_change] = 0.0

    log_returns = np.empty((D, T - 1), dtype=np.float64)
    for d in range(D):
        idiosyncratic_shocks = rng.normal(size=T - 1)
        idio_vol_scale = np.ones(T - 1, dtype=np.float64)
        for t in range(1, T - 1):
            idio_vol_scale[t] = (
                0.92
                + 0.05 * idio_vol_scale[t - 1]
                + 0.10 * min(abs(idiosyncratic_shocks[t - 1]), 3.0)
            )

        local_drift = np.full(T - 1, stock_daily_drift[d], dtype=np.float64)
        if has_change[d]:
            local_drift[change_index[d]:] += drift_change[d]

        log_returns[d, :] = (
            local_drift
            + market_beta[d] * market_returns
            + sector_beta[d] * sector_returns[sector_id[d], :]
            + idio_scale[d] * idio_vol_scale * idiosyncratic_shocks
        )
        log_returns[d, :] = np.clip(log_returns[d, :], -0.18, 0.18)

    Y = np.empty((D, T), dtype=np.float64)
    Y[:, 0] = initial_prices
    cumulative_log_return = np.cumsum(log_returns, axis=1)
    Y[:, 1:] = initial_prices[:, None] * np.exp(cumulative_log_return)
    Y = np.ascontiguousarray(Y)

    selected_tickers = [f"SYN{d + 1:06d}" for d in range(D)]
    price_frame = pd.DataFrame(Y.T, index=dates, columns=selected_tickers)

    metadata = pd.DataFrame(
        {
            "ticker": selected_tickers,
            "sector_id": sector_id,
            "initial_price": initial_prices,
            "market_beta": market_beta,
            "sector_beta": sector_beta,
            "idiosyncratic_daily_volatility": idio_scale,
            "baseline_daily_drift": stock_daily_drift,
            "has_slope_regime_change": has_change,
            "synthetic_change_index": change_index,
            "synthetic_change_date": [dates[int(i)].date() for i in change_index],
            "daily_drift_change": drift_change,
        }
    )

    return x, Y, dates, selected_tickers, price_frame, metadata


# =========================================================================
# MATHEMATICAL GUIDE: main
# Persist x, Y and metadata. No fitting or optimization occurs here.
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Simulate stock-like multivariate data")
    parser.add_argument("--D", type=int, default=10000)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", default=START_DATE_DEFAULT)
    parser.add_argument("--output", default="synthetic_stock_data.npz")
    parser.add_argument("--sectors", type=int, default=20)
    parser.add_argument("--change-probability", type=float, default=0.70)
    parser.add_argument("--change-magnitude", type=float, default=0.0015)
    args = parser.parse_args()
    x, Y, dates, tickers, _, metadata = generate_synthetic_stock_data(
        D=args.D, T=args.T, random_seed=args.seed, start_date=args.start_date,
        number_sectors=args.sectors, change_probability=args.change_probability,
        change_magnitude=args.change_magnitude)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, Y=Y, dates=dates.strftime("%Y-%m-%d").to_numpy(dtype=str),
                        tickers=np.asarray(tickers), metadata_json=np.asarray(metadata.to_json(orient="records", date_format="iso")))
    print(f"Saved D={args.D}, T={args.T} dataset to {path.resolve()}")

if __name__ == "__main__":
    main()
