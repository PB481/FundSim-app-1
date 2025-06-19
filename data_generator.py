import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Simulated minimal data for testing
data = {
    'valuation_date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
    'asset_id': ['ASSET_1'] * 5,
    'price_or_value': [100, 101, 102, 103, 104]
}
df_asset_valuations = pd.DataFrame(data)

# Prepare DataFrame to simulate multi-index
asset_id = 'ASSET_1'
df_asset_valuations['valuation_date'] = pd.to_datetime(df_asset_valuations['valuation_date'])
df_asset_valuations_indexed = df_asset_valuations.set_index(['asset_id', 'valuation_date']).sort_index()

# Extract series for test
try:
    temp_series_raw_index = df_asset_valuations_indexed.loc[asset_id, 'price_or_value']

    # Force index to datetime and remove tz
    index_as_datetime = pd.to_datetime(pd.Index(temp_series_raw_index.index), errors='coerce')
    index_as_datetime = index_as_datetime.tz_localize(None)
    index_as_datetime = index_as_datetime[~pd.isnull(index_as_datetime)]

    # Localize to London timezone
    london_tz = pytz.timezone('Europe/London')
    localized_index = index_as_datetime.tz_localize(london_tz, errors='coerce')

    # Apply the new index
    asset_valuation_series = temp_series_raw_index.set_axis(localized_index).dropna()

    print("Success:")
    print(asset_valuation_series)

except Exception as e:
    print(f"Failed with error: {e}")
