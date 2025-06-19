# data_generator.py (rewritten for fund accountant use and compatibility)

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import pytz

fake = Faker('en_US')

# --- Constants ---
LONDON_TZ = pytz.timezone('Europe/London')
CURRENCIES = ["USD", "EUR", "GBP"]
TRANSACTION_TYPES = ["Buy", "Sell", "Dividend", "Interest", "Fee", "Expense", "Capital Call", "Distribution"]

# --- Helper Functions ---
def safe_localize(index):
    dt_index = pd.to_datetime(index, errors='coerce')
    dt_index = dt_index[~dt_index.isna()]
    return dt_index.tz_localize(LONDON_TZ)

# --- Sample Asset Universe ---
ASSET_UNIVERSE = {
    "Equity_ET": ["AAPL", "MSFT", "GOOG"],
    "FixedIncome_ET": ["US10YT", "CORPBONDAAA"],
    "ETF_ET": ["SPY", "QQQ"],
    "Crypto_ET": ["BTC", "ETH"],
    "Loan": ["Term Loan B"],
    "HardToPrice": ["Private Equity X"]
}

# --- Main Generation Functions ---
def generate_asset_master(start_date, end_date):
    assets, prices = [], {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='D', tz=LONDON_TZ)

    for asset_type, asset_list in ASSET_UNIVERSE.items():
        for asset_name in asset_list:
            asset_id = f"{asset_type}_{asset_name.replace(' ', '_')}"
            price = round(random.uniform(100, 1000), 2)
            currency = random.choice(CURRENCIES)

            assets.append({
                "asset_id": asset_id,
                "asset_name": asset_name,
                "asset_type": asset_type,
                "currency": currency,
                "inception_date": fake.date_between(start_date='-5y', end_date=start_date),
                "liquidity_score": round(random.uniform(0.1, 1.0), 2)
            })

            daily_prices = []
            for _ in date_range:
                change = np.random.normal(0, 0.01)
                price = max(price * (1 + change), 1)
                daily_prices.append(round(price, 2))

            prices[asset_id] = daily_prices

    df_assets = pd.DataFrame(assets).set_index("asset_id")
    df_valuations = pd.DataFrame([
        {"valuation_date": date_range[i], "asset_id": aid, "price": price}
        for aid, plist in prices.items()
        for i, price in enumerate(plist)
    ])
    return df_assets, df_valuations

def generate_funds(num_funds, assets):
    funds, holdings = [], []
    asset_ids = list(assets.index)

    for _ in range(num_funds):
        fund_id = fake.uuid4()
        inception_date = fake.date_between(start_date='-10y', end_date='-1y')
        aum = round(random.uniform(1e7, 5e8), 2)

        fund = {
            "fund_id": fund_id,
            "fund_name": fake.company(),
            "aum": aum,
            "currency": random.choice(CURRENCIES),
            "administrator": fake.company_suffix(),
            "reporting_currency": random.choice(CURRENCIES),
            "inception_date": inception_date
        }
        funds.append(fund)

        selected_assets = random.sample(asset_ids, k=random.randint(2, 4))
        allocations = np.random.dirichlet(np.ones(len(selected_assets)), size=1)[0]

        for i, asset_id in enumerate(selected_assets):
            holdings.append({
                "holding_id": fake.uuid4(),
                "fund_id": fund_id,
                "asset_id": asset_id,
                "allocation": round(allocations[i], 4),
                "acquisition_date": fake.date_between(start_date=inception_date, end_date='-1y')
            })

    return pd.DataFrame(funds), pd.DataFrame(holdings)

def generate_transactions(funds, holdings, valuations, start_date, end_date):
    txns = []
    valuations["valuation_date"] = pd.to_datetime(valuations["valuation_date"])
    valuations = valuations.set_index(["asset_id", "valuation_date"]).sort_index()

    for _, row in holdings.iterrows():
        asset_id = row["asset_id"]
        fund_id = row["fund_id"]
        currency = funds.loc[funds["fund_id"] == fund_id, "currency"].values[0]

        price_series = valuations.loc[asset_id, :].squeeze()
        if price_series.empty: continue

        try:
            localized_index = safe_localize(price_series.index)
            price_series.index = localized_index
            price_series = price_series[~price_series.index.isna()]
        except Exception as e:
            continue

        tx_dates = pd.date_range(start=start_date, end=end_date, periods=random.randint(5, 10))
        for tx_date in tx_dates:
            tx_type = random.choice(TRANSACTION_TYPES)
            price = price_series.asof(tx_date) if not price_series.empty else 100.0
            amount = round(random.uniform(1000, 100000), 2)
            units = round(amount / price, 4)

            txns.append({
                "transaction_id": fake.uuid4(),
                "fund_id": fund_id,
                "asset_id": asset_id,
                "transaction_date": tx_date.date(),
                "transaction_type": tx_type,
                "amount": amount,
                "units": units if tx_type in ["Buy", "Sell"] else 0,
                "price_per_unit": price,
                "currency": currency
            })

    return pd.DataFrame(txns)

# --- Entry Point Example ---
if __name__ == "__main__":
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)

    assets, valuations = generate_asset_master(start_date, end_date)
    funds, holdings = generate_funds(5, assets)
    transactions = generate_transactions(funds, holdings, valuations, start_date, end_date)

    # Save or view sample
    print(assets.head())
    print(funds.head())
    print(transactions.head())
