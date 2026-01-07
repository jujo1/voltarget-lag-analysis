"""
Generate test fixtures from real CSV data.
Selects instruments with actual non-zero allocations.
"""
import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class DataBundle:
    """Container for all portfolio data."""
    returns: pd.DataFrame
    alloc_lots: pd.DataFrame
    alloc_raw: pd.DataFrame
    alloc_usd: pd.DataFrame
    contract_values: pd.DataFrame
    prices: pd.DataFrame


def load_csv_data(data_dir: Path) -> DataBundle:
    """Load all CSV files from project directory."""
    csv_files = {
        "returns": "ret_cc_usd_20251209_133002_052948d.csv",
        "alloc_raw": "alloc_raw_final_20251209_133002_052948d.csv",
        "alloc_usd": "alloc_usd_20251209_133002_052948d.csv",
        "alloc_lots": "alloc_lots_20251209_133002_052948d.csv",
        "contract_values": "contract_value_lc_20251209_133002_052948d.csv",
        "prices": "prices_settle_20251209_133002_052948d.csv"
    }
    
    data = {}
    for name, filename in csv_files.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        data[name] = df
        logger.info(f"Loaded {name}: {df.shape}")
    
    return DataBundle(
        returns=data["returns"],
        alloc_lots=data["alloc_lots"],
        alloc_raw=data["alloc_raw"],
        alloc_usd=data["alloc_usd"],
        contract_values=data["contract_values"],
        prices=data["prices"]
    )


def find_active_instruments(bundle: DataBundle, min_coverage: float = 0.5) -> List[str]:
    """
    Find instruments with actual non-zero allocations.
    
    Args:
        bundle: DataBundle containing allocation data
        min_coverage: Minimum fraction of days with non-zero allocations
        
    Returns:
        List of instrument names with sufficient allocation coverage
    """
    active = []
    total_days = len(bundle.alloc_raw)
    
    for col in bundle.alloc_raw.columns:
        vals = bundle.alloc_raw[col].fillna(0).values
        non_zero_days = np.count_nonzero(vals)
        coverage = non_zero_days / total_days
        
        if coverage >= min_coverage:
            active.append({
                "instrument": col,
                "non_zero_days": non_zero_days,
                "coverage": coverage
            })
    
    # Sort by coverage descending
    active.sort(key=lambda x: x["coverage"], reverse=True)
    logger.info(f"Found {len(active)} instruments with >= {min_coverage:.0%} coverage")
    
    return [a["instrument"] for a in active]


def create_fixture(
    bundle: DataBundle,
    instruments: List[str],
    n_days: int,
    output_path: Path
) -> None:
    """
    Create a fixture file with specified instruments and date range.
    
    Args:
        bundle: Full DataBundle
        instruments: List of instrument names to include
        n_days: Number of days from tail to include
        output_path: Path to save pickle file
    """
    # Validate instruments exist
    missing = set(instruments) - set(bundle.returns.columns)
    if missing:
        raise ValueError(f"Instruments not found in data: {missing}")
    
    # Extract tail data for selected instruments
    fixture = DataBundle(
        returns=bundle.returns[instruments].tail(n_days).copy(),
        alloc_lots=bundle.alloc_lots[instruments].tail(n_days).copy(),
        alloc_raw=bundle.alloc_raw[instruments].tail(n_days).copy(),
        alloc_usd=bundle.alloc_usd[instruments].tail(n_days).copy(),
        contract_values=bundle.contract_values[instruments].tail(n_days).copy(),
        prices=bundle.prices[instruments].tail(n_days).copy()
    )
    
    # Validate non-zero allocations exist
    alloc_sum = fixture.alloc_raw.abs().sum().sum()
    if alloc_sum < 1e-10:
        raise ValueError("Fixture contains no non-zero allocations!")
    
    with open(output_path, 'wb') as f:
        pickle.dump(fixture, f)
    
    # Log metadata
    date_range = {
        "start": str(bundle.returns.tail(n_days).index[0]),
        "end": str(bundle.returns.tail(n_days).index[-1])
    }
    
    logger.info(f"Created fixture: {output_path}")
    logger.info(f"  Instruments: {len(instruments)}")
    logger.info(f"  Days: {n_days}")
    logger.info(f"  Date range: {date_range}")
    logger.info(f"  Total allocation magnitude: {alloc_sum:.4f}")


def main():
    """Generate all required fixtures."""
    data_dir = Path("/mnt/project")
    output_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FIXTURE GENERATION - USING REAL ALLOCATION DATA")
    logger.info("=" * 60)
    
    # Load all data
    bundle = load_csv_data(data_dir)
    
    # Find instruments with real allocations
    active_instruments = find_active_instruments(bundle, min_coverage=0.5)
    
    if len(active_instruments) < 10:
        raise ValueError(f"Insufficient active instruments: {len(active_instruments)}")
    
    # Create full fixture (300 days, all active instruments)
    create_fixture(
        bundle=bundle,
        instruments=active_instruments,
        n_days=300,
        output_path=output_dir / "fixture_tail_300.pkl"
    )
    
    # Create subset fixture (300 days, 20 instruments)
    create_fixture(
        bundle=bundle,
        instruments=active_instruments[:20],
        n_days=300,
        output_path=output_dir / "fixture_active_20.pkl"
    )
    
    # Create small fixture for quick tests (100 days, 10 instruments)
    create_fixture(
        bundle=bundle,
        instruments=active_instruments[:10],
        n_days=100,
        output_path=output_dir / "fixture_quick_10.pkl"
    )
    
    logger.info("=" * 60)
    logger.info("FIXTURE GENERATION COMPLETE")
    logger.info("=" * 60)
    
    # Return summary for verification
    return {
        "active_instruments": active_instruments,
        "fixtures_created": [
            "fixture_tail_300.pkl",
            "fixture_active_20.pkl", 
            "fixture_quick_10.pkl"
        ]
    }


if __name__ == "__main__":
    result = main()
    print(f"\nActive instruments: {result['active_instruments'][:10]}...")
