# data_generator.py

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import pytz # For timezone-aware dates, crucial for financial data

# Initialize Faker (choose locale as needed, 'en_US' is good for general finance)
fake = Faker('en_US') 

# --- Configuration ---
# Define a broad universe of assets
ASSET_UNIVERSE = {
    "Equity_ET": ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "JPM", "BAC", "NVDA", "V", "MA", "INTC", "CSCO", "PEP", "KO", "PFE"],
    "FixedIncome_ET": ["US10YT", "GBPBOND", "EURBOND", "JPYBOND", "CORPBONDAAA", "MUNIBOND"],
    "ETF_ET": ["SPY", "QQQ", "DIA", "IWV", "VWO", "EMB"],
    "Crypto_ET": ["BTC", "ETH", "SOL", "XRP"], # Treated as ET for simplified pricing
    "FX_ET": ["EURUSD", "GBPUSD", "USDJPY", "EURGBP"], # Treated as ET for simplified pricing

    "FixedIncome_OTC": ["Corporate Debt Private Placement", "CMBS Tranche A", "ABS Auto Loan"],
    "Equity_OTC": ["Pre-IPO Tech Startup Shares", "Private Company Shares Series B"],
    "Commodity_OTC": ["Crude Oil Futures Contract", "Gold Spot Contract"],

    "FundOfFunds": [], # Placeholder, populated by generated funds (actual fund IDs will be used)
    "Loan": ["Commercial Real Estate Loan", "Corporate Term Loan B", "Aircraft Lease Loan"],
    "HardToPrice": ["Private Equity Fund X", "Hedge Fund Y Side Pocket", "Direct Real Estate Property Dublin", "Art Collection A"],
}

# Characteristics for different fund types
FUND_TYPE_CONFIG = {
    "Equity Fund": {
        "asset_types_pool": ["Equity_ET", "ETF_ET", "FixedIncome_ET", "Crypto_ET"], # Could hold some bonds/ETFs
        "num_holdings_range": (10, 50),
        "aum_range": (50_000_000, 5_000_000_000),
        "fee_range": (0.0075, 0.015), # 0.75% - 1.5%
        "geography": ["North America", "Europe", "Global"],
        "strategies": ["Growth", "Value", "Index", "Sector Specific"],
    },
    "Fixed Income Fund": {
        "asset_types_pool": ["FixedIncome_ET", "FixedIncome_OTC", "Loan", "ETF_ET"],
        "num_holdings_range": (15, 60),
        "aum_range": (100_000_000, 10_000_000_000),
        "fee_range": (0.003, 0.008), # 0.3% - 0.8%
        "geography": ["Global", "US", "Eurozone"],
        "strategies": ["High Yield", "Investment Grade", "Government Bonds", "Mortgage-Backed"],
    },
    "Real Estate Fund": {
        "asset_types_pool": ["HardToPrice", "Equity_ET", "Loan"], # REITs are equity ET
        "num_holdings_range": (3, 15),
        "aum_range": (200_000_000, 8_000_000_000),
        "fee_range": (0.01, 0.025), # 1% - 2.5%
        "geography": ["Europe (incl. Ireland)", "North America", "Asia"],
        "strategies": ["Core", "Value-Add", "Opportunistic"],
    },
    "Multi-Asset Fund": {
        "asset_types_pool": ["Equity_ET", "FixedIncome_ET", "ETF_ET", "Crypto_ET", "FixedIncome_OTC", "HardToPrice"],
        "num_holdings_range": (20, 100),
        "aum_range": (80_000_000, 15_000_000_000),
        "fee_range": (0.006, 0.018), # 0.6% - 1.8%
        "geography": ["Global"],
        "strategies": ["Balanced", "Aggressive", "Conservative", "Target Date"],
    },
    "Hedge Fund": {
        "asset_types_pool": ["Equity_ET", "FixedIncome_ET", "ETF_ET", "Crypto_ET", "FX_ET", "Equity_OTC", "Commodity_OTC"],
        "num_holdings_range": (5, 30), # Can be concentrated
        "aum_range": (20_000_000, 20_000_000_000),
        "fee_range": (0.015, 0.03), # 1.5% - 3% (management fee)
        "geography": ["Global"],
        "strategies": ["Long/Short Equity", "Global Macro", "Event Driven"],
        "performance_fee_range": (0.1, 0.2), # 10% - 20%
    },
    "Private Equity Fund": {
        "asset_types_pool": ["Equity_OTC", "Loan", "HardToPrice"],
        "num_holdings_range": (5, 15),
        "aum_range": (500_000_000, 25_000_000_000),
        "fee_range": (0.02, 0.025), # 2% - 2.5%
        "geography": ["Global", "North America", "Europe"],
        "strategies": ["Buyout", "Venture Capital", "Growth Equity"],
        "illiquidity_premium": (0.05, 0.15) # Example for hard-to-price
    },
    "Fund of Funds (FoF)": {
        "asset_types_pool": ["FundOfFunds"], # Special type, holds other funds
        "num_holdings_range": (5, 15), # Number of underlying funds
        "aum_range": (100_000_000, 10_000_000_000),
        "fee_range": (0.005, 0.01), # 0.5% - 1% (on top of underlying fund fees)
        "geography": ["Global"],
        "strategies": ["Diversified", "Sector Specific", "Geographic Focus"],
    }
}

# Asset-level cost factor definitions based on research
# These are the *base* cost factors for each unit of effort/complexity
ASSET_TYPE_COST_PROFILES = {
    "Equity_ET": {"base_holding_cost": 10.0, "processing_touchpoints": 10, "reconciliation_effort": 8, "regulatory_effort": 5, "pricing_complexity": 1},
    "FixedIncome_ET": {"base_holding_cost": 12.0, "processing_touchpoints": 12, "reconciliation_effort": 9, "regulatory_effort": 6, "pricing_complexity": 2},
    "ETF_ET": {"base_holding_cost": 10.0, "processing_touchpoints": 10, "reconciliation_effort": 8, "regulatory_effort": 5, "pricing_complexity": 1},
    "Crypto_ET": {"base_holding_cost": 25.0, "processing_touchpoints": 20, "reconciliation_effort": 25, "regulatory_effort": 30, "pricing_complexity": 3},
    "FX_ET": {"base_holding_cost": 15.0, "processing_touchpoints": 15, "reconciliation_effort": 12, "regulatory_effort": 10, "pricing_complexity": 2},
    "FixedIncome_OTC": {"base_holding_cost": 40.0, "processing_touchpoints": 30, "reconciliation_effort": 35, "regulatory_effort": 25, "pricing_complexity": 4},
    "Equity_OTC": {"base_holding_cost": 50.0, "processing_touchpoints": 40, "reconciliation_effort": 45, "regulatory_effort": 35, "pricing_complexity": 5},
    "Commodity_OTC": {"base_holding_cost": 45.0, "processing_touchpoints": 35, "reconciliation_effort": 40, "regulatory_effort": 30, "pricing_complexity": 4},
    "FundOfFunds": {"base_holding_cost": 70.0, "processing_touchpoints": 60, "reconciliation_effort": 50, "regulatory_effort": 40, "pricing_complexity": 6},
    "Loan": {"base_holding_cost": 80.0, "processing_touchpoints": 70, "reconciliation_effort": 60, "regulatory_effort": 50, "pricing_complexity": 7},
    "HardToPrice": {"base_holding_cost": 100.0, "processing_touchpoints": 80, "reconciliation_effort": 70, "regulatory_effort": 60, "pricing_complexity": 8},
}


# --- Utility Functions for Data Generation ---

def generate_asset_master(asset_universe, start_date, end_date, price_variance=0.01, loan_rate_variance=0.001):
    """Generates a master list of synthetic assets with historical data."""
    assets_data = []
    asset_prices = {} # To store time series data

    # Timezone aware start/end dates
    london_tz = pytz.timezone('Europe/London')
    start_dt = london_tz.localize(datetime.combine(start_date, datetime.min.time()))
    end_dt = london_tz.localize(datetime.combine(end_date, datetime.min.time()))
    
    # Generate prices for every day from start_dt to end_dt
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D', tz=london_tz)

    for asset_type, assets in asset_universe.items():
        for asset_name in assets:
            asset_id = f"{asset_type}_{asset_name.replace(' ', '_').replace('-', '_')}"
            
            # Initial asset data
            initial_val = round(random.uniform(50, 2000), 2)
            if asset_type == "Loan":
                initial_val = round(random.uniform(1_000_000, 100_000_000), 2) # Loan principal
            elif asset_type == "HardToPrice":
                initial_val = round(random.uniform(5_000_000, 500_000_000), 2)

            assets_data.append({
                "asset_id": asset_id,
                "asset_name": asset_name,
                "asset_type_category": asset_type,
                "is_exchange_traded": "ET" in asset_type,
                "is_over_the_counter": "OTC" in asset_type,
                "is_loan": asset_type == "Loan",
                "is_hard_to_price": asset_type == "HardToPrice",
                "inception_date": fake.date_between(start_date='-5y', end_date=start_date),
                "initial_price_or_value": initial_val,
                "currency": random.choice(["USD", "EUR", "GBP"]), # Could refine this based on asset
                "liquidity_score": round(random.uniform(0.1, 1.0), 2) # 1.0 is highly liquid
            })
            
            # Generate price history for this asset
            current_price = initial_val
            daily_prices = []
            for _ in date_range: # Loop through each date
                if "ET" in asset_type or "FX" in asset_type or "Crypto" in asset_type:
                    change = np.random.normal(0, price_variance)
                    current_price *= (1 + change)
                    current_price = max(1.0, current_price) # Price doesn't go below 1
                elif asset_type == "Loan":
                    change = np.random.normal(0, loan_rate_variance)
                    current_price *= (1 + change) # Loan values are relatively stable
                elif asset_type == "HardToPrice":
                    if random.random() < 0.05: # 5% chance of update per day
                        change = np.random.normal(0, price_variance * 0.1)
                        current_price *= (1 + change)
                else: # OTC
                    if random.random() < 0.15: # 15% chance of update per day
                        change = np.random.normal(0, price_variance * 0.5)
                        current_price *= (1 + change)
                daily_prices.append(round(current_price, 2))
            asset_prices[asset_id] = daily_prices

    df_assets = pd.DataFrame(assets_data).drop_duplicates(subset=["asset_id"]).set_index("asset_id")

    # Create daily prices/valuations DataFrame
    price_history_records = []
    for asset_id, prices in asset_prices.items():
        for i, price in enumerate(prices):
            # Ensure we don't go out of bounds if prices list is longer than date_range (shouldn't be)
            if i < len(date_range): 
                price_history_records.append({
                    "valuation_date": date_range[i].date(), # Store as date object
                    "asset_id": asset_id,
                    "price_or_value": price
                })
    df_asset_valuations = pd.DataFrame(price_history_records)

    return df_assets, df_asset_valuations


def generate_funds_and_holdings(num_funds, fund_type_config, df_assets_master, start_date_transactions, end_date_transactions):
    """Generates synthetic fund and holding data."""
    funds = []
    holdings = []
    
    # Pre-filter assets by type for quicker lookup
    et_assets = df_assets_master[df_assets_master['is_exchange_traded']].index.tolist()
    otc_assets = df_assets_master[df_assets_master['is_over_the_counter']].index.tolist()
    loan_assets = df_assets_master[df_assets_master['is_loan']].index.tolist()
    hard_to_price_assets = df_assets_master[df_assets_master['is_hard_to_price']].index.tolist()

    potential_underlying_funds = [] # To be populated with fund_ids as they are created

    for i in range(num_funds):
        fund_id = fake.uuid4()
        fund_type_name, config = random.choice(list(fund_type_config.items()))

        fund_record = {
            "fund_id": fund_id,
            "fund_name": fake.company() + " " + fund_type_name,
            "fund_type": fund_type_name,
            "inception_date": fake.date_between(start_date="-10y", end_date=start_date_transactions),
            "management_fee_bps": round(random.uniform(config["fee_range"][0], config["fee_range"][1]) * 10000), # Basis Points
            "initial_aum": round(random.uniform(config["aum_range"][0], config["aum_range"][1]), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]), # Fund's base currency
            "strategy": random.choice(config["strategies"]),
            "geography_focus": random.choice(config["geography"]),
            "is_fo_fund": fund_type_name == "Fund of Funds (FoF)"
        }
        if "performance_fee_range" in config:
            fund_record["performance_fee_bps"] = round(random.uniform(config["performance_fee_range"][0], config["performance_fee_range"][1]) * 10000)
        
        funds.append(fund_record)

        if not fund_record["is_fo_fund"]:
            potential_underlying_funds.append(fund_id) # Only non-FoF funds can be underlying funds here

        num_holdings = random.randint(*config["num_holdings_range"])
        
        selected_holding_assets = []
        for _ in range(num_holdings):
            asset_category = random.choice(config["asset_types_pool"])
            
            asset_choice = None
            # Prioritize picking from relevant asset lists, fall back if list is empty
            if asset_category == "Equity_ET" and et_assets: asset_choice = random.choice(et_assets)
            elif asset_category == "FixedIncome_ET" and et_assets: asset_choice = random.choice(et_assets)
            elif asset_category == "ETF_ET" and et_assets: asset_choice = random.choice(et_assets)
            elif asset_category == "Crypto_ET" and et_assets: asset_choice = random.choice(et_assets)
            elif asset_category == "FX_ET" and et_assets: asset_choice = random.choice(et_assets)
            elif asset_category == "FixedIncome_OTC" and otc_assets: asset_choice = random.choice(otc_assets)
            elif asset_category == "Equity_OTC" and otc_assets: asset_choice = random.choice(otc_assets)
            elif asset_category == "Commodity_OTC" and otc_assets: asset_choice = random.choice(otc_assets)
            elif asset_category == "Loan" and loan_assets: asset_choice = random.choice(loan_assets)
            elif asset_category == "HardToPrice" and hard_to_price_assets: asset_choice = random.choice(hard_to_price_assets)
            elif asset_category == "FundOfFunds" and potential_underlying_funds:
                asset_choice = random.choice(potential_underlying_funds)
            
            # Ensure unique assets for each fund's holdings for simplicity, and check if asset_choice was successfully made
            if asset_choice and asset_choice not in selected_holding_assets:
                selected_holding_assets.append(asset_choice)

        # Special handling for FoF if it couldn't pick enough unique underlying funds
        if fund_type_name == "Fund of Funds (FoF)" and not selected_holding_assets and potential_underlying_funds:
             selected_holding_assets = random.sample(potential_underlying_funds, min(num_holdings, len(potential_underlying_funds)))

        total_allocation = 1.0
        if selected_holding_assets:
            # Distribute allocation, ensuring sum is 1.0
            allocations = [random.uniform(0.01, 0.2) for _ in selected_holding_assets]
            total_sum = sum(allocations)
            
            # Handle case where total_sum might be zero (e.g., if selected_holding_assets was empty)
            if total_sum > 0:
                allocations = [round(alloc / total_sum, 4) for alloc in allocations] # Normalize and round
                # Ensure sum is exactly 1.0 by adjusting the last allocation
                allocations[-1] = round(allocations[-1] + (1.0 - sum(allocations)), 4)
            else:
                allocations = [round(1.0 / len(selected_holding_assets), 4)] * len(selected_holding_assets)


            for j, asset_id in enumerate(selected_holding_assets):
                holdings.append({
                    "holding_id": fake.uuid4(),
                    "fund_id": fund_id,
                    "asset_id": asset_id,
                    "allocation_percentage": allocations[j],
                    "acquisition_date": fake.date_between(start_date=fund_record["inception_date"], end_date=start_date_transactions)
                })

    return pd.DataFrame(funds), pd.DataFrame(holdings)


def generate_transactions(df_funds, df_holdings, df_asset_valuations, start_date_transactions, end_date_transactions):
    """
    Generates synthetic transaction data for holdings.
    Handles asset valuations more robustly.
    """
    transactions = []
    
    # Prepare valuation data for quick lookup
    # Ensure valuation_date is datetime and set as part of MultiIndex for efficient .loc access
    df_asset_valuations['valuation_date'] = pd.to_datetime(df_asset_valuations['valuation_date'])
    df_asset_valuations_indexed = df_asset_valuations.set_index(['asset_id', 'valuation_date']).sort_index()

    # Timezone aware dates for transaction period
    london_tz = pytz.timezone('Europe/London')
    
    transaction_types = ["Buy", "Sell", "Dividend", "Interest", "Fee", "Expense", "Capital Call", "Distribution"]

    # Simple model to track fund balances (not full NAV, just conceptual)
    fund_conceptual_balances = {f_id: {"cash": f_init_aum, "assets_value": 0} for f_id, f_init_aum in df_funds.set_index("fund_id")["initial_aum"].items()}

    for index, holding in df_holdings.iterrows():
        fund_id = holding["fund_id"]
        asset_id = holding["asset_id"]
        fund_currency = df_funds.loc[df_funds["fund_id"] == fund_id, "currency"].iloc[0] # Get fund currency
        
        # Get historical prices for the asset
        # Check if asset_id exists in the multi-index before trying to slice
        asset_valuation_series = pd.Series(dtype=float) # Initialize as empty Series
        if asset_id in df_asset_valuations_indexed.index:
            asset_valuation_series = df_asset_valuations_indexed.loc[asset_id, 'price_or_value']
            # Ensure the series index is datetime and apply timezone for asof comparison
            asset_valuation_series.index = pd.to_datetime(asset_valuation_series.index).tz_localize(london_tz, errors='coerce')
            # Filter for relevant dates (from acquisition to end of transaction period)
            asset_valuation_series = asset_valuation_series[
                (asset_valuation_series.index.date >= holding["acquisition_date"]) &
                (asset_valuation_series.index.date <= end_date_transactions)
            ].sort_index() # Crucial for asof to work correctly

        # If asset_valuation_series is empty after filtering, assign a default price.
        # This can happen if the asset's inception_date is after the transaction period.
        default_price_fallback = 100.0 

        # --- Initial acquisition transaction (if applicable) ---
        if holding["acquisition_date"] >= start_date_transactions:
            # Get acquisition price: prefer exact date, then asof, then default
            acquisition_price_ts = pd.Timestamp(holding["acquisition_date"], tz=london_tz)
            acquisition_price = asset_valuation_series.get(acquisition_price_ts, None)

            if acquisition_price is None and not asset_valuation_series.empty:
                asof_price = asset_valuation_series.asof(acquisition_price_ts)
                if pd.isna(asof_price): # If asof also returns NaN
                    acquisition_price = asset_valuation_series.iloc[0] if not asset_valuation_series.empty else default_price_fallback
                else:
                    acquisition_price = asof_price
            elif acquisition_price is None: # Series was empty to begin with
                acquisition_price = default_price_fallback

            initial_units = (fund_conceptual_balances[fund_id]["cash"] * holding["allocation_percentage"]) / acquisition_price if acquisition_price > 0 else 0
            
            if initial_units > 0:
                transactions.append({
                    "transaction_id": fake.uuid4(),
                    "fund_id": fund_id,
                    "asset_id": asset_id,
                    "transaction_date": holding["acquisition_date"],
                    "transaction_type": "Buy",
                    "amount": round(initial_units * acquisition_price, 2), # Total cost
                    "units": round(initial_units, 4),
                    "price_per_unit": round(acquisition_price, 2),
                    "currency": fund_currency,
                    "description": f"Initial acquisition of {df_assets_master.loc[asset_id]['asset_name']}" # Use asset_name for description
                })
                fund_conceptual_balances[fund_id]["cash"] -= round(initial_units * acquisition_price, 2)
                fund_conceptual_balances[fund_id]["assets_value"] += round(initial_units * acquisition_price, 2)
        
        # --- Simulate ongoing transactions (Buys/Sells/Income/Fees) ---
        num_holding_transactions = random.randint(1, 10) # 1-10 transactions per holding over period
        for _ in range(num_holding_transactions):
            # Generate transaction date within the holding's relevant period
            tx_date_dt = fake.date_between(start_date=max(holding["acquisition_date"], start_date_transactions), end_date=end_date_transactions)
            tx_date_ts = pd.Timestamp(tx_date_dt, tz=london_tz) # Convert to Timestamp for asof
            
            tx_type = random.choice(transaction_types)
            amount = round(random.uniform(100, 500000), 2)
            
            # Determine price for transaction date using robust logic
            current_price_for_date = asset_valuation_series.get(tx_date_ts, None)
            if current_price_for_date is None and not asset_valuation_series.empty:
                asof_price = asset_valuation_series.asof(tx_date_ts)
                if pd.isna(asof_price):
                    current_price_for_date = asset_valuation_series.iloc[0] if not asset_valuation_series.empty else default_price_fallback
                else:
                    current_price_for_date = asof_price
            elif current_price_for_date is None:
                current_price_for_date = default_price_fallback

            units = 0
            if current_price_for_date is not None and current_price_for_date > 0 and tx_type in ["Buy", "Sell"]:
                units = round(amount / current_price_for_date, 4)
            
            transactions.append({
                "transaction_id": fake.uuid4(),
                "fund_id": fund_id,
                "asset_id": asset_id,
                "transaction_date": tx_date_dt, # Store as date object
                "transaction_type": tx_type,
                "amount": amount,
                "units": units,
                "price_per_unit": round(current_price_for_date, 2),
                "currency": fund_currency,
                "description": fake.sentence(nb_words=6)
            })

            # Update conceptual balance (very simplified)
            if tx_type == "Buy":
                fund_conceptual_balances[fund_id]["cash"] -= amount
                fund_conceptual_balances[fund_id]["assets_value"] += amount
            elif tx_type == "Sell":
                fund_conceptual_balances[fund_id]["cash"] += amount
                fund_conceptual_balances[fund_id]["assets_value"] -= amount
            elif tx_type in ["Dividend", "Interest", "Distribution"]:
                fund_conceptual_balances[fund_id]["cash"] += amount
            elif tx_type in ["Fee", "Expense", "Capital Call"]:
                 if tx_type == "Capital Call":
                    fund_conceptual_balances[fund_id]["cash"] += amount
                 else:
                    fund_conceptual_balances[fund_id]["cash"] -= amount

    return pd.DataFrame(transactions)
