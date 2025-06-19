import streamlit as st
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import pytz
import altair as alt

# Assuming you've created a file named `data_generator.py`
# and moved the generation functions and config dictionaries into it.
from data_generator import generate_asset_master, generate_funds_and_holdings, generate_transactions, ASSET_UNIVERSE, FUND_TYPE_CONFIG

# --- Database Connection Configuration (IMPORTANT: Use Streamlit Secrets) ---
# For Snowflake:
# st.secrets["snowflake"]["user"]
# st.secrets["snowflake"]["password"]
# st.secrets["snowflake"]["account"]
# ... etc.

# For MotherDuck:
# st.secrets["motherduck"]["token"]

# --- 1. Data Generation Configuration (from previous fund_data_generator_app.py) ---
# Define ASSET_UNIVERSE, FUND_TYPE_CONFIG
# ... (copy ASSET_UNIVERSE and FUND_TYPE_CONFIG dictionaries here) ...
# You can also move these to a separate `config.py` file and import them.

# --- 2. Data Generation Functions (from previous fund_data_generator_app.py) ---
# Copy generate_asset_master, generate_funds_and_holdings, generate_transactions here
# (or import from a `data_generator.py` file if preferred for modularity)
# Make sure to remove st.cache_data from these generation functions if they are *always*
# triggered when the app starts, or use a custom caching mechanism based on user input flags.
# However, if generation is a button click, st.cache_data is still useful.
# ... (copy generate_asset_master, generate_funds_and_holdings, generate_transactions functions here) ...

# --- 3. Cost Model Specifics (from previous cost_estimator_app.py) ---
# Define ASSET_TYPE_COST_PROFILES
# ... (copy ASSET_TYPE_COST_PROFILES dictionary here) ...

# --- 4. Cost Model Functions (from previous cost_estimator_app.py) ---
# Copy enhance_asset_master_with_cost_factors and calculate_fund_administration_costs here
# ... (copy enhance_asset_master_with_cost_factors, calculate_fund_administration_costs functions here) ...

# --- 5. Database Interaction Functions ---
# THESE ARE PLACEHOLDERS. YOU NEED TO IMPLEMENT THEM WITH YOUR ACTUAL CREDENTIALS AND TABLE SCHEMAS.

def get_db_connection(db_type):
    if db_type == "Snowflake":
        # Install: pip install snowflake-connector-python
        import snowflake.connector
        try:
            conn = snowflake.connector.connect(
                user=st.secrets["snowflake"]["user"],
                password=st.secrets["snowflake"]["password"],
                account=st.secrets["snowflake"]["account"],
                warehouse=st.secrets["snowflake"]["warehouse"],
                database=st.secrets["snowflake"]["database"],
                schema=st.secrets["snowflake"]["schema"]
            )
            return conn
        except Exception as e:
            st.error(f"Snowflake connection failed: {e}")
            return None
    elif db_type == "MotherDuck":
        # Install: pip install duckdb motherduck
        import duckdb # Make sure this is installed: pip install duckdb motherduck

def get_db_connection(db_type):
    if db_type == "Snowflake":
        # ... (Snowflake connection logic from previous code) ...
        pass # Placeholder for brevity, assuming it's already there
    elif db_type == "MotherDuck":
        try:
            # Get token securely from Streamlit secrets
            md_token = st.secrets["motherduck"]["token"]
            
            # Connect to MotherDuck. 
            # You can optionally specify a database name, e.g., "md:my_database_name"
            # If you don't specify one, it often defaults to 'my_db' if you have one.
            # Using ?motherduck_token= ensures the token is passed
            conn = duckdb.connect(f'md:?motherduck_token={md_token}')
            
            # Optionally, you can set the token as a session variable if you prefer
            # conn.execute(f"SET motherduck.token = '{md_token}';") # If connecting without token in string

            # Load MotherDuck extension (required for cloud access)
            conn.execute("INSTALL motherduck; LOAD motherduck;") 
            
            return conn
        except Exception as e:
            st.error(f"MotherDuck connection failed: {e}")
            st.info("Ensure `motherduck_token` is set correctly in `.streamlit/secrets.toml` and `duckdb` and `motherduck` packages are installed.")
            return None
    return None

def write_data_to_db(df, table_name, db_type, conn):
    if conn is None: return False
    st.info(f"Attempting to write {len(df)} rows to {table_name} in {db_type}...")
    try:
        if db_type == "Snowflake":
            # ... (Snowflake writing logic) ...
            pass # Placeholder
        elif db_type == "MotherDuck":
            # DuckDB (MotherDuck) can directly read from Pandas DataFrames in the Python environment.
            # Convert table_name to lowercase as is common practice in DuckDB
            md_table_name = table_name.lower() 
            
            # Drop table if it exists (for fresh writes)
            conn.execute(f"DROP TABLE IF EXISTS {md_table_name};")
            
            # Create table and insert data from DataFrame
            conn.execute(f"CREATE TABLE {md_table_name} AS SELECT * FROM df") 
            st.success(f"Successfully wrote {len(df)} rows to MotherDuck table `{md_table_name}`.")
            return True
        return False
    except Exception as e:
        st.error(f"Failed to write data to {db_type} table {table_name}: {e}")
        return False

@st.cache_data(show_spinner="Loading data from database...")
def read_data_from_db(table_name, db_type, conn):
    if conn is None: return pd.DataFrame()
    try:
        if db_type == "Snowflake":
            # ... (Snowflake reading logic) ...
            pass # Placeholder
        elif db_type == "MotherDuck":
            md_table_name = table_name.lower()
            df = conn.execute(f"SELECT * FROM {md_table_name}").fetchdf()
            st.success(f"Loaded {len(df)} rows from MotherDuck table `{md_table_name}`.")
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to read data from {db_type} table {table_name}: {e}")
        return pd.DataFrame()

# --- Streamlit UI Layout ---
st.set_page_config(layout="wide", page_title="Fund Automation Insights", page_icon="📈")

st.sidebar.title("App Controls")
app_mode = st.sidebar.radio("Choose App Mode", ["Generate Synthetic Data", "Analyze Fund Admin Costs"])

# --- Database Connection UI in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.subheader("Database Connection")
db_enabled = st.sidebar.checkbox("Enable Database Connection", value=False)
db_type_selected = st.sidebar.selectbox("Database Type", ["Snowflake", "MotherDuck"], disabled=not db_enabled)
db_conn = None
if db_enabled:
    db_conn = get_db_connection(db_type_selected)
    if db_conn:
        st.sidebar.success(f"Connected to {db_type_selected}.")
    else:
        st.sidebar.warning(f"Could not connect to {db_type_selected}. Check secrets.")

# --- Main App Logic based on Mode ---
if app_mode == "Generate Synthetic Data":
    st.header("📊 Generate Synthetic Fund Accounting Data")
    st.markdown("""
    Use this section to create realistic-looking synthetic data for your fund accounting system.
    This data can then be used for testing, demonstrations, and populating your cost analysis.
    """)

    # Data Generation Parameters
    st.sidebar.subheader("Generation Settings")
    num_funds_to_generate = st.sidebar.slider("Number of Funds", 10, 200, 50, step=10, key="gen_num_funds")
    num_years_transactions = st.sidebar.slider("Transaction History (Years)", 0.5, 5.0, 1.0, step=0.5, key="gen_years_trans")
    random_seed = st.sidebar.number_input("Random Seed (for reproducibility)", value=42, step=1, key="gen_random_seed")

    start_date_transactions = datetime.now().date() - timedelta(days=int(num_years_transactions * 365))
    end_date_transactions = datetime.now().date()

    if st.button("Generate New Data", type="primary"):
        with st.spinner("Generating data... this may take a moment for larger datasets."):
            Faker.seed(random_seed)
            random.seed(random_seed)
            np.random.seed(random_seed)

            # Generate Data
            df_assets_master, df_asset_valuations = generate_asset_master(
                ASSET_UNIVERSE,
                start_date=start_date_transactions - timedelta(days=365),
                end_date=end_date_transactions
            )
            df_funds, df_holdings = generate_funds_and_holdings(
                num_funds_to_generate, 
                FUND_TYPE_CONFIG, 
                df_assets_master, 
                start_date_transactions, 
                end_date_transactions
            )
            df_transactions = generate_transactions(
                df_funds, 
                df_holdings, 
                df_asset_valuations, 
                start_date_transactions, 
                end_date_transactions
            )
        
        st.session_state['generated_df_funds'] = df_funds
        st.session_state['generated_df_holdings'] = df_holdings
        st.session_state['generated_df_assets_master'] = df_assets_master
        st.session_state['generated_df_transactions'] = df_transactions # Store transactions too
        
        st.success("Data generation complete! You can now view or push this data.")

        # Display generated data (optional, can be moved to a separate "View Data" section)
        st.subheader("Newly Generated Data Overviews")
        col1_gen, col2_gen = st.columns(2)
        with col1_gen:
            st.write("### Funds")
            st.dataframe(df_funds.head())
            st.download_button(label="Download Funds CSV", data=df_funds.to_csv(index=False).encode('utf-8'), file_name="synthetic_funds.csv", mime="text/csv")
        with col2_gen:
            st.write("### Holdings")
            st.dataframe(df_holdings.head())
            st.download_button(label="Download Holdings CSV", data=df_holdings.to_csv(index=False).encode('utf-8'), file_name="synthetic_holdings.csv", mime="text/csv")
        
        st.write("### Assets Master")
        st.dataframe(df_assets_master.head())
        st.download_button(label="Download Assets CSV", data=df_assets_master.to_csv(index=False).encode('utf-8'), file_name="synthetic_assets_master.csv", mime="text/csv")

        # Database Push Option
        if db_enabled and db_conn:
            st.markdown("---")
            st.subheader(f"Push Generated Data to {db_type_selected}")
            if st.button(f"Push Data to {db_type_selected} Tables"):
                if write_data_to_db(df_funds, "funds", db_type_selected, db_conn):
                    st.write("Funds data written.")
                if write_data_to_db(df_holdings, "holdings", db_type_selected, db_conn):
                    st.write("Holdings data written.")
                if write_data_to_db(df_assets_master.reset_index(), "assets_master", db_type_selected, db_conn): # reset index for writing
                    st.write("Assets Master data written.")
                if write_data_to_db(df_transactions, "transactions", db_type_selected, db_conn):
                    st.write("Transactions data written.")
                st.success("All selected data pushed to database!")
        else:
            st.info("Enable and connect to a database in the sidebar to push generated data.")

elif app_mode == "Analyze Fund Admin Costs":
    st.header("💰 Analyze Fund Administration Costs")
    st.markdown("""
    Estimate fund administration costs and visualize the impact of AI-driven hyper-automation.
    """)

    # --- Data Loading for Analysis ---
    st.sidebar.subheader("Load Data for Analysis")
    load_analysis_data_source = st.sidebar.radio("Source for Cost Analysis Data", 
                                                ("Use Last Generated Data", "Load from Database (if enabled)", "Use Sample Data"))

    df_funds_for_analysis = pd.DataFrame()
    df_holdings_for_analysis = pd.DataFrame()
    df_assets_master_for_analysis = pd.DataFrame()

    if load_analysis_data_source == "Use Last Generated Data":
        if 'generated_df_funds' in st.session_state:
            df_funds_for_analysis = st.session_state['generated_df_funds']
            df_holdings_for_analysis = st.session_state['generated_df_holdings']
            df_assets_master_for_analysis = st.session_state['generated_df_assets_master']
            st.info("Using data from last generation session.")
        else:
            st.warning("No data generated in this session yet. Please go to 'Generate Synthetic Data' mode or select another source.")
    
    elif load_analysis_data_source == "Load from Database (if enabled)":
        if db_enabled and db_conn:
            if st.button("Load Data from DB"):
                df_funds_for_analysis = read_data_from_db("funds", db_type_selected, db_conn)
                df_holdings_for_analysis = read_data_from_db("holdings", db_type_selected, db_conn)
                df_assets_master_for_analysis = read_data_from_db("assets_master", db_type_selected, db_conn).set_index('asset_id')
                st.success("Data loaded from database.")
        else:
            st.warning("Database connection is not enabled or established. Please enable it in the sidebar.")
    
    elif load_analysis_data_source == "Use Sample Data":
        # Copy sample data from generation mode's 'Load Sample' logic
        df_funds_for_analysis = pd.DataFrame({
            'fund_id': ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'],
            'fund_name': ['Equity Growth Fund', 'Global Bond Fund', 'Irish Property Fund', 'Multi-Strat HF', 'PE Buyout Fund', 'Emerging Markets FoF', 'Direct Lending Pool', 'Crypto Fund X'],
            'fund_type': ['Equity Fund', 'Fixed Income Fund', 'Real Estate Fund', 'Hedge Fund', 'Private Equity Fund', 'Fund of Funds (FoF)', 'Fixed Income Fund', 'Equity Fund'],
            'initial_aum': [500_000_000, 1_000_000_000, 250_000_000, 150_000_000, 750_000_000, 300_000_000, 400_000_000, 80_000_000],
            'management_fee_bps': [100, 50, 150, 200, 200, 80, 70, 250],
            'currency': ['USD', 'EUR', 'EUR', 'USD', 'GBP', 'USD', 'EUR', 'USD'],
            'strategy': ['Growth', 'Investment Grade', 'Core', 'Global Macro', 'Buyout', 'Diversified', 'Direct', 'HODL'],
            'geography_focus': ['Global', 'Eurozone', 'Europe (incl. Ireland)', 'Global', 'Europe', 'Global', 'Europe', 'Global'],
            'is_fo_fund': [False, False, False, False, False, True, False, False]
        })
        df_assets_master_for_analysis = pd.DataFrame({
            'asset_id': ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12'],
            'asset_name': ['AAPL', 'US10YT', 'Pre-IPO Tech', 'CRE Loan', 'Private Equity X', 'SPY', 'Corporate Bond PP', 'HF Alpha Fund', 'Ether', 'Oil Futures OTC', 'EURUSD Spot', 'Commercial Building Dublin'],
            'asset_type_category': ['Equity_ET', 'FixedIncome_ET', 'Equity_OTC', 'Loan', 'HardToPrice', 'ETF_ET', 'FixedIncome_OTC', 'FundOfFunds', 'Crypto_ET', 'Commodity_OTC', 'FX_ET', 'HardToPrice'],
            'is_exchange_traded': [True, True, False, False, False, True, False, False, True, False, True, False],
            'is_over_the_counter': [False, False, True, False, False, False, True, False, False, True, False, False],
            'is_loan': [False, False, False, True, False, False, False, True, False, False, False],
            'is_hard_to_price': [False, False, False, False, True, False, False, False, False, False, False, True],
            'liquidity_score': [0.9, 0.8, 0.3, 0.2, 0.1, 0.9, 0.4, 0.2, 0.7, 0.5, 0.9, 0.1],
        }).set_index('asset_id')
        df_holdings_for_analysis = pd.DataFrame({
            'holding_id': ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8', 'h9', 'h10', 'h11', 'h12', 'h13', 'h14', 'h15', 'h16', 'h17', 'h18'],
            'fund_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3', 'f4', 'f4', 'f5', 'f5', 'f6', 'f6', 'f7', 'f7', 'f8', 'f8', 'f1', 'f2'],
            'asset_id': ['a1', 'a6', 'a2', 'a7', 'a4', 'a12', 'a1', 'a8', 'a3', 'a5', 'f1', 'f2', 'a4', 'a10', 'a9', 'a11', 'a9', 'a11'],
            'allocation_percentage': [0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.7, 0.3, 0.1, 0.1],
        })
        # Apply investor count and reporting freq to sample funds
        df_funds_for_analysis = enhance_asset_master_with_cost_factors(df_assets_master_for_analysis).pipe(lambda x: df_funds_for_analysis.merge(x[['investor_count', 'reporting_frequency', 'custom_reporting_requirements_flag']], left_on='fund_id', right_index=True, how='left', suffixes=('_x','')))
        df_funds_for_analysis = df_funds_for_analysis.loc[:,~df_funds_for_analysis.columns.duplicated()]
        df_funds_for_analysis[['investor_count', 'custom_reporting_requirements_flag']] = df_funds_for_analysis[['investor_count', 'custom_reporting_requirements_flag']].fillna({'investor_count': 20, 'custom_reporting_requirements_flag': False})
        df_funds_for_analysis['reporting_frequency'] = df_funds_for_analysis['reporting_frequency'].fillna("Monthly")

    # Only show the rest of the app if data is loaded for analysis
    if not df_funds_for_analysis.empty and not df_holdings_for_analysis.empty and not df_assets_master_for_analysis.empty:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Cost Parameters (Per Month)")

        # --- User-definable Cost Parameters ---
        cost_params = {}
        cost_params['base_cost_per_fund_per_month'] = st.sidebar.number_input("Base Fund Cost (€)", value=500.0, step=50.0, key="cp_base_fund_cost")
        cost_params['aum_based_cost_bps_per_month'] = st.sidebar.number_input("AUM-based Cost (bps)", value=0.005, step=0.001, format="%.3f", help="Cost per 10,000 units of AUM", key="cp_aum_cost")
        cost_params['investor_count_cost_per_investor_per_month'] = st.sidebar.number_input("Cost per Investor (€)", value=5.0, step=1.0, key="cp_investor_cost")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Asset Activity & Complexity Costs")
        cost_params['cost_per_touchpoint'] = st.sidebar.number_input("Cost per Processing Touchpoint (€)", value=0.5, step=0.1, key="cp_touchpoint")
        cost_params['cost_per_reconciliation_effort_unit'] = st.sidebar.number_input("Cost per Recon Effort Unit (€)", value=1.5, step=0.1, key="cp_recon_unit")
        cost_params['cost_per_regulatory_effort_unit'] = st.sidebar.number_input("Cost per Regulatory Effort Unit (€)", value=2.0, step=0.1, key="cp_reg_unit")
        cost_params['cost_per_pricing_complexity_unit'] = st.sidebar.number_input("Cost per Pricing Complexity Unit (€)", value=3.0, step=0.1, key="cp_pricing_unit")

        st.sidebar.markdown("---")
        st.sidebar.subheader("Reporting Costs")
        cost_params['cost_per_holding_reporting'] = st.sidebar.number_input("Base Reporting Cost per Holding (€)", value=2.0, step=0.5, key="cp_holding_report")
        cost_params['reporting_freq_multiplier_daily'] = st.sidebar.number_input("Daily Reporting Multiplier", value=2.5, step=0.1, key="cp_daily_mult")
        cost_params['reporting_freq_multiplier_weekly'] = st.sidebar.number_input("Weekly Reporting Multiplier", value=1.5, step=0.1, key="cp_weekly_mult")
        cost_params['reporting_freq_multiplier_monthly'] = st.sidebar.number_input("Monthly Reporting Multiplier", value=1.0, step=0.1, key="cp_monthly_mult")
        cost_params['reporting_freq_multiplier_quarterly'] = st.sidebar.number_input("Quarterly Reporting Multiplier", value=0.5, step=0.1, key="cp_quarterly_mult")
        cost_params['cost_per_custom_report_flag'] = st.sidebar.number_input("Cost for Custom Reporting (€)", value=500.0, step=100.0, key="cp_custom_report")

        st.sidebar.markdown("---")
        st.sidebar.subheader("3. AI-Driven Hyper-Automation Impact")
        automation_efficiency_gain_pct = st.sidebar.slider(
            "Overall Automation Efficiency Gain (%)",
            0, 50, 25, step=5,
            help="Percentage reduction in *automatable* administrative costs due to AI-driven hyper-automation. (e.g. 25% means 25% saving on human effort components)",
            key="auto_eff_gain"
        )

        # --- Perform Calculation ---
        df_funds_with_costs = calculate_fund_administration_costs(
            df_funds_for_analysis, df_holdings_for_analysis, df_assets_master_for_analysis, cost_params, automation_efficiency_gain_pct
        )

        st.header("Cost Analysis Results (Per Fund, Per Month)")

        # ... (Metrics, Tables, Charts - copy directly from the previous cost_estimator_app.py) ...
        # Display key metrics
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Total Funds Analyzed", len(df_funds_with_costs))
        with colB:
            avg_gross_cost = df_funds_with_costs['gross_admin_cost_per_month'].mean()
            st.metric("Avg. Gross Cost per Fund", f"€{avg_gross_cost:,.2f}")
        with colC:
            avg_net_cost = df_funds_with_costs['net_admin_cost_per_month'].mean()
            st.metric(f"Avg. Net Cost (After Automation)", f"€{avg_net_cost:,.2f}")
            st.markdown(f"**Avg. Savings per Fund:** €{(avg_gross_cost - avg_net_cost):,.2f} ({df_funds_with_costs['percentage_savings'].mean():,.1f}%)")

        st.markdown("---")
        st.subheader("Cost Breakdown by Fund Type")

        # Group by fund type and calculate average costs
        fund_type_cost_summary = df_funds_with_costs.groupby('fund_type').agg(
            NumFunds=('fund_id', 'count'),
            AvgGrossCost=('gross_admin_cost_per_month', 'mean'),
            AvgNetCost=('net_admin_cost_per_month', 'mean'),
            AvgAutomationSavings=('automation_savings', 'mean'),
            AvgAUM=('initial_aum', 'mean'),
            AvgCostPerAUM_Gross_bps=('cost_per_aum_bps_gross', 'mean'),
            AvgCostPerAUM_Net_bps=('cost_per_aum_bps_net', 'mean')
        ).reset_index().sort_values(by="AvgGrossCost", ascending=False)

        fund_type_cost_summary['AvgGrossCost'] = fund_type_cost_summary['AvgGrossCost'].map("€{:,.2f}".format)
        fund_type_cost_summary['AvgNetCost'] = fund_type_cost_summary['AvgNetCost'].map("€{:,.2f}".format)
        fund_type_cost_summary['AvgAutomationSavings'] = fund_type_cost_summary['AvgAutomationSavings'].map("€{:,.2f}".format)
        fund_type_cost_summary['AvgAUM'] = fund_type_cost_summary['AvgAUM'].map("€{:,.0f}".format)
        fund_type_cost_summary['AvgCostPerAUM_Gross_bps'] = fund_type_cost_summary['AvgCostPerAUM_Gross_bps'].map("{:,.2f} bps".format)
        fund_type_cost_summary['AvgCostPerAUM_Net_bps'] = fund_type_cost_summary['AvgCostPerAUM_Net_bps'].map("{:,.2f} bps".format)


        st.dataframe(fund_type_cost_summary.set_index("fund_type"))

        # Detailed table of all funds
        st.markdown("---")
        st.subheader("Detailed Fund Costs")
        display_cols = [
            'fund_name', 'fund_type', 'initial_aum', 'num_holdings', 'investor_count', 'reporting_frequency',
            'gross_admin_cost_per_month', 'automation_savings', 'net_admin_cost_per_month',
            'cost_per_aum_bps_gross', 'cost_per_aum_bps_net', 'percentage_savings'
        ]
        df_display = df_funds_with_costs[display_cols].copy()
        df_display['initial_aum'] = df_display['initial_aum'].map("€{:,.0f}".format)
        df_display['gross_admin_cost_per_month'] = df_display['gross_admin_cost_per_month'].map("€{:,.2f}".format)
        df_display['automation_savings'] = df_display['automation_savings'].map("€{:,.2f}".format)
        df_display['net_admin_cost_per_month'] = df_display['net_admin_cost_per_month'].map("€{:,.2f}".format)
        df_display['cost_per_aum_bps_gross'] = df_display['cost_per_aum_bps_gross'].map("{:,.2f} bps".format)
        df_display['cost_per_aum_bps_net'] = df_display['cost_per_aum_bps_net'].map("{:,.2f} bps".format)
        df_display['percentage_savings'] = df_display['percentage_savings'].map("{:,.1f}%".format)

        st.dataframe(df_display, use_container_width=True)

        st.markdown("---")
        st.subheader("Cost Component Breakdown (Average per Fund)")
        avg_cost_components = df_funds_with_costs[[
            'base_admin_cost', 'aum_based_cost', 'investor_based_cost',
            'total_gross_holding_cost',
            'total_reporting_cost', 'custom_reporting_cost'
        ]].mean().reset_index()
        avg_cost_components.columns = ['Cost Component', 'Average Cost (€)']
        avg_cost_components['Average Cost (€)'] = avg_cost_components['Average Cost (€)'].map("€{:,.2f}".format)
        st.dataframe(avg_cost_components, use_container_width=True)
        
        # --- Visualization ---
        st.markdown("---")
        st.subheader("Visualizing Automation Impact & Cost Drivers")

        # Chart 1: Gross vs. Net Admin Cost by Fund Type
        chart_data_fund_type = fund_type_cost_summary.melt(
            id_vars=['fund_type'], 
            value_vars=['AvgGrossCost', 'AvgNetCost'], 
            var_name='Cost Type', 
            value_name='Cost'
        ).assign(Cost=lambda x: x['Cost'].str.replace('€|,', '', regex=True).astype(float))

        chart1 = alt.Chart(chart_data_fund_type).mark_bar().encode(
            x=alt.X('fund_type', title='Fund Type', sort='-y'),
            y=alt.Y('Cost', title='Monthly Admin Cost (€)'),
            color=alt.Color('Cost Type', legend=alt.Legend(title="Cost Scenario")),
            tooltip=['fund_type', 'Cost Type', alt.Tooltip('Cost', format='€,.2f')]
        ).properties(
            title=f'Avg. Gross vs. Net Admin Cost by Fund Type ({automation_efficiency_gain_pct}% Automation Gain)'
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)

        # Chart 2: Savings by Fund Type
        savings_data = fund_type_cost_summary[['fund_type', 'AvgAutomationSavings']].copy()
        savings_data['AvgAutomationSavings'] = savings_data['AvgAutomationSavings'].str.replace('€|,', '', regex=True).astype(float)

        chart2 = alt.Chart(savings_data).mark_bar(color='teal').encode(
            x=alt.X('fund_type', title='Fund Type', sort='-y'),
            y=alt.Y('AvgAutomationSavings', title='Avg. Monthly Savings (€)'),
            tooltip=['fund_type', alt.Tooltip('AvgAutomationSavings', format='€,.2f')]
        ).properties(
            title='Average Monthly Automation Savings per Fund Type'
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)


        # Chart 3: Average Cost Components Breakdown (Pie Chart or Stacked Bar)
        chart_data_components = avg_cost_components.copy()
        chart_data_components['Average Cost (€)'] = chart_data_components['Average Cost (€)'].str.replace('€|,', '', regex=True).astype(float)

        chart3 = alt.Chart(chart_data_components).mark_arc(outerRadius=120).encode(
            theta=alt.Theta(field="Average Cost (€)", type="quantitative"),
            color=alt.Color(field="Cost Component", type="nominal"),
            order=alt.Order("Average Cost (€)", sort="descending"),
            tooltip=["Cost Component", alt.Tooltip("Average Cost (€)", format='€,.2f')]
        ).properties(
            title="Average Cost Components Before Automation"
        )
        text = chart3.mark_text(radius=140).encode(
            text=alt.Text("Average Cost (€)", format="€,.0f"),
            order=alt.Order("Average Cost (€)", sort="descending"),
            color=alt.value("black")
        )
        st.altair_chart(chart3 + text, use_container_width=True)


    else:
        st.warning("No data available for cost analysis. Please generate or load data.")


st.markdown("---")
st.markdown("### How this tool helps to terminate potential costs:")
st.markdown("""
1.  **Quantify ROI:** Directly shows the estimated savings (€ and %) from hyper-automation at various efficiency levels.
2.  **Identify High-Cost Funds/Assets:** The detailed tables and charts highlight which fund types or individual holdings are the most expensive to administer. These are your prime targets for automation.
3.  **Prioritize Automation Efforts:** By seeing the "Cost Component Breakdown," you can identify which operational areas (e.g., asset processing, reconciliation, regulatory reporting) contribute most to gross costs, indicating where automation will have the largest impact.
4.  **"What-If" Scenarios:** Adjusting the 'Automation Efficiency Gain (%)' slider allows you to model different levels of success and build a business case for investment in AI tools.
5.  **Benchmark:** The "Cost per AUM (bps)" allows comparison to industry TERs, ensuring your model's outputs are in a realistic range.
""")
