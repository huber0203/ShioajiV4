from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shioaji as sj
import os
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shioaji Trading API",
    description="Trading API with technical indicators and historical data using Shioaji",
    version="2.0.0"
)

# Global Shioaji API instance
api = None
login_status = False
account_info = None

# Taiwan timezone
TW_TZ = pytz.timezone('Asia/Taipei')

# Pydantic models
class LoginRequest(BaseModel):
    api_key: str
    secret_key: str

class OrderRequest(BaseModel):
    action: str  # "Buy" or "Sell"
    code: str    # Stock code
    quantity: int
    price: Optional[float] = None  # None for market order
    order_type: str = "ROD"  # ROD, IOC, FOK

class LoginResponse(BaseModel):
    success: bool
    message: str
    account_info: Optional[Dict[str, Any]] = None

class OrderResponse(BaseModel):
    success: bool
    message: str
    order_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    message: str
    login_status: bool
    current_time: str
    timezone: str

class QuoteResponse(BaseModel):
    success: bool
    code: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str

@app.on_event("startup")
async def startup_event():
    global api, login_status
    try:
        api = sj.Shioaji()
        logger.info("Shioaji API initialized")
        
        # Auto-login using environment variables
        api_key = os.getenv("SHIOAJI_API_KEY")
        secret_key = os.getenv("SHIOAJI_SECRET_KEY")
        
        logger.info(f"Environment variables check:")
        logger.info(f"SHIOAJI_API_KEY: {'SET' if api_key else 'NOT SET'}")
        logger.info(f"SHIOAJI_SECRET_KEY: {'SET' if secret_key else 'NOT SET'}")
        logger.info(f"SHIOAJI_PERSON_ID: {'SET' if os.getenv('SHIOAJI_PERSON_ID') else 'NOT SET'}")
        
        if api_key and secret_key:
            logger.info(f"API Key length: {len(api_key)}")
            logger.info(f"Secret Key length: {len(secret_key)}")
            logger.info(f"API Key first 4 chars: {api_key[:4]}...")
            logger.info(f"Secret Key first 4 chars: {secret_key[:4]}...")
            
            api_key_clean = api_key.strip().strip('"').strip("'")
            secret_key_clean = secret_key.strip().strip('"').strip("'")
            
            if api_key_clean != api_key:
                logger.info("API Key was cleaned (removed quotes/whitespace only)")
            if secret_key_clean != secret_key:
                logger.info("Secret Key was cleaned (removed quotes/whitespace only)")
            
            logger.info(f"Cleaned API Key length: {len(api_key_clean)}")
            logger.info(f"Cleaned Secret Key length: {len(secret_key_clean)}")
            logger.info(f"Cleaned API Key first 4 chars: {api_key_clean[:4]}...")
            logger.info(f"Cleaned Secret Key first 4 chars: {secret_key_clean[:4]}...")
            
            try:
                logger.info("Attempting auto-login with environment variables...")
                login_attempts = [
                    # First try: original cleaned keys
                    (api_key_clean, secret_key_clean, "original"),
                    # Second try: URL encoded keys
                    (api_key_clean.encode('utf-8').decode('unicode_escape'), 
                     secret_key_clean.encode('utf-8').decode('unicode_escape'), "unicode_escape"),
                ]
                
                login_successful = False
                for api_key_attempt, secret_key_attempt, method in login_attempts:
                    try:
                        logger.info(f"Trying login method: {method}")
                        accounts = api.login(
                            api_key=api_key_attempt,
                            secret_key=secret_key_attempt,
                            fetch_contract=True,
                            subscribe_trade=True
                        )
                        
                        if accounts:
                            login_status = True
                            login_successful = True
                            logger.info(f"Auto-login successful with method: {method}")
                            logger.info(f"Connected accounts: {[acc.account_id for acc in accounts]}")
                            logger.info(f"Stock account: {api.stock_account}")
                            logger.info(f"Future account: {api.futopt_account}")
                            break
                            
                    except Exception as attempt_error:
                        logger.warning(f"Login attempt with {method} failed: {attempt_error}")
                        continue
                
                if not login_successful:
                    login_status = False
                    logger.error("All auto-login attempts failed")
                    
            except Exception as e:
                login_status = False
                logger.error(f"Auto-login error: {e}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Error details: {str(e)}")
        else:
            login_status = False
            logger.warning("Auto-login skipped: Missing environment variables")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        api = None
        login_status = False

@app.on_event("shutdown")
async def shutdown_event():
    global api, login_status
    if api and login_status:
        try:
            api.logout()
            login_status = False
            logger.info("Logged out from Shioaji API")
        except Exception as e:
            logger.error(f"Error during logout: {e}")

@app.get("/")
async def root():
    return {
        "message": "Shioaji Trading API with Auto-Login and Historical Data",
        "version": "2.0.0",
        "status": "running",
        "connected": login_status,
        "auto_login": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
        "env_status": {
            "SHIOAJI_API_KEY": "SET" if os.getenv("SHIOAJI_API_KEY") else "NOT SET",
            "SHIOAJI_SECRET_KEY": "SET" if os.getenv("SHIOAJI_SECRET_KEY") else "NOT SET",
            "SHIOAJI_PERSON_ID": "SET" if os.getenv("SHIOAJI_PERSON_ID") else "NOT SET"
        },
        "endpoints": {
            "account": ["/login", "/logout", "/accounts", "/positions"],
            "trading": ["/order", "/quote/{stock_code}"],
            "technical": ["/technical/{stock_codes}"],
            "historical": [
                "/historical/ticks/{stock_code}",
                "/historical/kbars/{stock_code}",
                "/historical/analysis/{stock_code}"
            ],
            "system": ["/health", "/retry-login"]
        }
    }

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Manual login to Shioaji API (optional if auto-login is configured)"""
    global api, login_status
    try:
        if not api:
            api = sj.Shioaji()
            
        accounts = api.login(
            api_key=request.api_key,
            secret_key=request.secret_key,
            fetch_contract=True,
            subscribe_trade=True
        )
        
        if accounts:
            login_status = True
            account_info = {
                "accounts": [
                    {
                        "account_id": acc.account_id,
                        "broker_id": acc.broker_id,
                        "person_id": acc.person_id,
                        "signed": acc.signed,
                        "username": acc.username
                    } for acc in accounts
                ],
                "stock_account": api.stock_account.account_id if api.stock_account else None,
                "futopt_account": api.futopt_account.account_id if api.futopt_account else None
            }
            
            return LoginResponse(
                success=True,
                message="Manual login successful",
                account_info=account_info
            )
        else:
            login_status = False
            return LoginResponse(
                success=False,
                message="Login failed - no accounts returned"
            )
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return LoginResponse(
            success=False,
            message=f"Login failed: {str(e)}"
        )

@app.post("/logout")
async def logout():
    """Logout from Shioaji API"""
    global api, login_status
    try:
        if api and login_status:
            api.logout()
            login_status = False
            return {"success": True, "message": "Logout successful"}
        else:
            return {"success": False, "message": "Not logged in"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return {"success": False, "message": f"Logout failed: {str(e)}"}

def check_connection():
    """Check if Shioaji API is actually connected"""
    global api, login_status
    try:
        if not api:
            return False
        
        # Try to get account info to verify connection
        accounts = api.list_accounts()
        if accounts:
            login_status = True
            return True
        else:
            login_status = False
            return False
    except:
        login_status = False
        return False

def ensure_login():
    """Ensure we're logged in, attempt login if not"""
    global api, login_status
    
    # First check if we're already connected
    if check_connection():
        return True
    
    # Try to login using environment variables
    api_key = os.getenv("SHIOAJI_API_KEY")
    secret_key = os.getenv("SHIOAJI_SECRET_KEY")
    
    if not api_key or not secret_key:
        return False
    
    try:
        if not api:
            api = sj.Shioaji()
        
        api_key_clean = api_key.strip().strip('"').strip("'")
        secret_key_clean = secret_key.strip().strip('"').strip("'")
        
        accounts = api.login(
            api_key=api_key_clean,
            secret_key=secret_key_clean,
            fetch_contract=True,
            subscribe_trade=True
        )
        
        if accounts:
            login_status = True
            logger.info("Login successful via ensure_login")
            return True
        else:
            login_status = False
            return False
            
    except Exception as e:
        logger.warning(f"ensure_login failed: {e}")
        login_status = False
        return False

@app.get("/accounts")
async def get_accounts():
    """Get account information"""
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        accounts = api.list_accounts()
        return {
            "success": True,
            "accounts": [
                {
                    "account_id": acc.account_id,
                    "broker_id": acc.broker_id,
                    "person_id": acc.person_id,
                    "account_type": type(acc).__name__,
                    "signed": acc.signed,
                    "username": acc.username
                }
                for acc in accounts
            ],
            "default_stock_account": api.stock_account.account_id if api.stock_account else None,
            "default_futopt_account": api.futopt_account.account_id if api.futopt_account else None
        }
    except Exception as e:
        logger.error(f"Get accounts error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        positions = api.list_positions()
        return {
            "success": True,
            "positions": [
                {
                    "code": pos.code,
                    "quantity": pos.quantity,
                    "price": pos.price,
                    "last_price": pos.last_price,
                    "pnl": pos.pnl
                }
                for pos in positions
            ]
        }
    except Exception as e:
        logger.error(f"Get positions error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.post("/order", response_model=OrderResponse)
async def place_order(request: OrderRequest):
    """Place a stock order"""
    global api
    try:
        if not ensure_login():
            return OrderResponse(
                success=False,
                message="Unable to connect - please check environment variables"
            )
        
        if request.action not in ["Buy", "Sell"]:
            return OrderResponse(
                success=False,
                message="Invalid action. Must be 'Buy' or 'Sell'"
            )
        
        # Get contract
        contract = api.Contracts.Stocks.get(request.code)
        if not contract:
            return OrderResponse(
                success=False,
                message=f"Stock {request.code} not found"
            )
        
        if not api.stock_account:
            return OrderResponse(
                success=False,
                message="No stock account available"
            )
        
        # Create order with proper constants
        order = api.Order(
            action=getattr(sj.constant.Action, request.action),
            price=request.price or 0,
            quantity=request.quantity,
            price_type=sj.constant.StockPriceType.LMT if request.price else sj.constant.StockPriceType.MKT,
            order_type=getattr(sj.constant.OrderType, request.order_type),
            order_lot=sj.constant.StockOrderLot.Common,
            account=api.stock_account
        )
        
        # Place order
        trade = api.place_order(contract, order)
        
        return OrderResponse(
            success=True,
            message="Order placed successfully",
            order_id=trade.order.id if trade and hasattr(trade, 'order') else None
        )
        
    except Exception as e:
        logger.error(f"Place order error: {e}")
        return OrderResponse(
            success=False,
            message=f"Order failed: {str(e)}"
        )

@app.get("/quote/{stock_code}")
async def get_quote(stock_code: str):
    """Get real-time quote for a stock"""
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # Get contract
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            return {"success": False, "message": f"Stock {stock_code} not found"}
        
        try:
            # Try to get snapshots first (more reliable)
            snapshots = api.snapshots([contract])
            if snapshots and len(snapshots) > 0:
                quote = snapshots[0]
                return {
                    "success": True,
                    "code": stock_code,
                    "name": contract.name,
                    "price": quote.close,
                    "volume": quote.volume,
                    "high": quote.high,
                    "low": quote.low,
                    "open": quote.open
                }
        except:
            pass
        
        # Fallback to basic contract info
        return {
            "success": True,
            "code": stock_code,
            "name": contract.name,
            "message": "Quote data not available, showing contract info only"
        }
        
    except Exception as e:
        logger.error(f"Get quote error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/technical/{stock_codes}")
async def get_technical_indicators(stock_codes: str, timeframe: str = "daily", session: str = "morning", date: str = None):
    """Get technical indicators for multiple stocks and timeframes with session control and date selection
    
    Args:
        stock_codes: Single stock or comma-separated stocks (e.g., "2330" or "2454,2317")
        timeframe: Single timeframe or comma-separated timeframes (e.g., "daily" or "daily,5min")
        session: Trading session - "morning" (早盤 09:00-13:30) or "night" (夜盤 15:00-05:00)
        date: Specific date to query (YYYY-MM-DD format, e.g., "2024-08-13"). If not provided, uses current date.
    
    Examples:
        /technical/2330 - Single stock, daily data, morning session, current date
        /technical/2330?date=2024-08-13 - Single stock, daily data for specific date
        /technical/2330?timeframe=5min&session=morning&date=2024-08-13 - 5min morning session data for specific date
        /technical/2330?timeframe=5min&session=night&date=2024-08-13 - 5min night session data for specific date
        /technical/2330,2454?timeframe=daily,5min&session=morning&date=2024-08-13 - Multiple stocks, both timeframes, specific date
    """
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # Parse stock codes and timeframes
        stock_list = [code.strip() for code in stock_codes.split(',')]
        timeframe_list = [tf.strip() for tf in timeframe.split(',')]
        
        # Validate session parameter
        if session not in ["morning", "night"]:
            return {"success": False, "message": "Invalid session. Must be 'morning' or 'night'"}
        
        # Parse and validate date parameter
        target_date = None
        if date:
            try:
                target_date = datetime.strptime(date, '%Y-%m-%d').date()
            except ValueError:
                return {"success": False, "message": "Invalid date format. Use YYYY-MM-DD (e.g., 2024-08-13)"}
        
        results = {}
        
        for stock_code in stock_list:
            # Get contract
            contract = api.Contracts.Stocks.get(stock_code)
            if not contract:
                results[stock_code] = {"success": False, "message": f"Stock {stock_code} not found"}
                continue
            
            stock_results = {}
            
            for tf in timeframe_list:
                try:
                    # Get technical data for this stock, timeframe, session, and date
                    tech_data = await get_single_stock_technical(stock_code, contract, tf, session, target_date)
                    stock_results[tf] = tech_data
                    
                except Exception as tf_error:
                    logger.error(f"Error getting {tf} data for {stock_code}: {tf_error}")
                    stock_results[tf] = {"success": False, "message": f"Error: {str(tf_error)}"}
            
            results[stock_code] = {
                "success": True,
                "name": contract.name,
                "data": stock_results
            }
        
        return {
            "success": True,
            "request": {
                "stocks": stock_list,
                "timeframes": timeframe_list,
                "session": session,
                "date": date or "current",
                "total_combinations": len(stock_list) * len(timeframe_list)
            },
            "results": results,
            "timestamp": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"Batch technical indicators error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

async def get_single_stock_technical(stock_code: str, contract, timeframe: str, session: str = "morning", target_date = None):
    """Helper function to get technical data for a single stock, timeframe, session, and specific date"""
    global api
    
    from datetime import datetime, timedelta
    
    logger.info(f"=== RAW DATA LOGGING START for {stock_code} ===")
    logger.info(f"Contract info: {contract}")
    logger.info(f"Timeframe: {timeframe}, Session: {session}, Target date: {target_date}")
    
    try:
        # Use target_date if provided, otherwise use current date
        if target_date:
            base_date = target_date
            now_tw = TW_TZ.localize(datetime.combine(target_date, datetime.min.time().replace(hour=12, minute=0)))
        else:
            now_tw = datetime.now(TW_TZ)
            base_date = now_tw.date()
        
        if timeframe == "5min":
            # Get intraday 1-minute data
            try:
                # Get all available recent data
                kbars = api.kbars(
                    contract=contract,
                    timeout=30000
                )
                
                logger.info(f"Raw kbars object type: {type(kbars)}")
                
                if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
                    logger.info(f"No kbars data available")
                    return {"success": False, "message": f"No intraday data available for {stock_code}"}
                
                # Convert kbars to DataFrame properly
                df = pd.DataFrame({
                    'ts': kbars.ts,
                    'Open': kbars.Open,
                    'High': kbars.High,
                    'Low': kbars.Low,
                    'Close': kbars.Close,
                    'Volume': kbars.Volume
                })
                
                logger.info(f"DataFrame shape after conversion: {df.shape}")
                
                # Convert timestamps to Taiwan timezone
                df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(TW_TZ)
                
                # Analyze available time range
                min_hour = df['ts'].dt.hour.min()
                max_hour = df['ts'].dt.hour.max()
                unique_hours = sorted(df['ts'].dt.hour.unique())
                
                logger.info(f"📊 Available time range: {min_hour}:00 - {max_hour}:00")
                logger.info(f"📊 Available hours: {unique_hours}")
                logger.info(f"📊 Total data points: {len(df)}")
                
                # Smart time filtering strategy
                if session == "morning":
                    session_name = "早盤 (Morning Session)"
                    
                    # Standard morning hours: 09:00-13:30
                    morning_hours = list(range(9, 14))  # 9, 10, 11, 12, 13
                    has_morning_data = any(h in unique_hours for h in morning_hours)
                    
                    if has_morning_data:
                        # Option 1: Has standard morning data
                        df_session = df[
                            (df['ts'].dt.hour.between(9, 13)) |
                            ((df['ts'].dt.hour == 13) & (df['ts'].dt.minute <= 30))
                        ]
                        time_strategy = "strict_morning"
                        logger.info(f"✅ Using strict morning hours (9:00-13:30): {len(df_session)} bars")
                        
                    else:
                        # Option 2: No standard morning, find day time
                        day_hours = list(range(6, 19))  # 6:00-18:00
                        has_day_data = any(h in unique_hours for h in day_hours)
                        
                        if has_day_data:
                            df_session = df[df['ts'].dt.hour.between(6, 18)]
                            time_strategy = "extended_day"
                            logger.info(f"⚠️ No standard morning data, using day hours (6:00-18:00): {len(df_session)} bars")
                            
                        else:
                            # Option 3: Use all available data
                            df_session = df.copy()
                            time_strategy = "all_available"
                            logger.info(f"⚠️ Using all available data: {len(df_session)} bars")
                            
                elif session == "night":
                    session_name = "夜盤 (Night Session)"
                    # Night session: 15:00-05:00 next day
                    df_night1 = df[df['ts'].dt.hour >= 15]  # 15:00-23:59
                    df_night2 = df[df['ts'].dt.hour <= 5]   # 00:00-05:00
                    df_session = pd.concat([df_night1, df_night2]).drop_duplicates().sort_values('ts')
                    time_strategy = "night_session"
                    logger.info(f"✅ Using night session (15:00-05:00): {len(df_session)} bars")
                
                logger.info(f"After session filtering ({session}) - DataFrame shape: {df_session.shape}")
                logger.info(f"Time strategy used: {time_strategy}")
                
                if df_session.empty:
                    return {
                        "success": False, 
                        "message": f"No data available for {stock_code} {session} session",
                        "available_hours": [int(h) for h in unique_hours],  # Convert numpy int to Python int
                        "available_time_range": f"{min_hour}:00 - {max_hour}:00",
                        "requested_session": session_name
                    }
                
                # Set timestamp as index for resampling
                df_session.set_index('ts', inplace=True)
                
                # Resample to 5-minute bars
                df_5min = df_session.resample('5T', label='right', closed='right').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                logger.info(f"After 5-minute resampling - DataFrame shape: {df_5min.shape}")
                
                # Reset index to get timestamp back as column
                df_5min.reset_index(inplace=True)
                
                if df_5min.empty:
                    return {"success": False, "message": f"Could not create 5-minute bars for {stock_code} {session} session"}
                
                # Get tick data to calculate buy/sell volumes
                tick_df = None
                try:
                    # Try to get tick data for the query date
                    logger.info(f"Trying to get tick data for date: {base_date}")
                    
                    ticks = api.ticks(
                        contract=contract,
                        date=base_date.strftime('%Y-%m-%d'),
                        timeout=30000
                    )
                    
                    logger.info(f"Raw ticks object type: {type(ticks)}")
                    
                    # Process tick data
                    if ticks and hasattr(ticks, 'ts') and ticks.ts:
                        tick_data = {
                            'ts': ticks.ts, 
                            'close': ticks.close, 
                            'volume': ticks.volume
                        }
                        
                        if hasattr(ticks, 'tick_type') and ticks.tick_type:
                            tick_data['tick_type'] = ticks.tick_type
                        
                        tick_df = pd.DataFrame(tick_data)
                        tick_df['ts'] = pd.to_datetime(tick_df['ts'], utc=True).dt.tz_convert(TW_TZ)
                        
                        # Filter tick data to match session time range
                        session_start = df_session.index.min()
                        session_end = df_session.index.max()
                        tick_df = tick_df[(tick_df['ts'] >= session_start) & (tick_df['ts'] <= session_end)]
                        
                        logger.info(f"✅ Tick DataFrame shape: {tick_df.shape}")
                        logger.info(f"Tick data time range: {tick_df['ts'].min()} to {tick_df['ts'].max()}")
                        
                except Exception as tick_error:
                    logger.error(f"Tick data error for {stock_code}: {tick_error}")
                    tick_df = None
                
                # Process each 5-minute interval for buy/sell volumes
                intraday_data = []
                total_volume_session = 0
                total_buy_volume = 0
                total_sell_volume = 0
                
                logger.info(f"🔄 Processing {len(df_5min)} five-minute intervals...")
                
                for idx, row in df_5min.iterrows():
                    # Calculate basic OHLC data
                    open_price = float(row['Open'])
                    high_price = float(row['High'])
                    low_price = float(row['Low'])
                    close_price = float(row['Close'])
                    volume = int(row['Volume'])
                    
                    avg_price = (open_price + high_price + low_price + close_price) / 4
                    
                    buy_volume = None
                    sell_volume = None
                    
                    # Calculate buy/sell volumes
                    if tick_df is not None and not tick_df.empty:
                        # Find ticks within this 5-minute interval
                        interval_start = row['ts'] - pd.Timedelta(minutes=5)
                        interval_end = row['ts']
                        
                        interval_ticks = tick_df[
                            (tick_df['ts'] > interval_start) & 
                            (tick_df['ts'] <= interval_end)
                        ]
                        
                        if not interval_ticks.empty:
                            if 'tick_type' in interval_ticks.columns:
                                # Use tick_type for precise calculation (most accurate method)
                                buy_ticks = interval_ticks[interval_ticks['tick_type'] == 1]  # 外盤（買方）
                                sell_ticks = interval_ticks[interval_ticks['tick_type'] == 2]  # 內盤（賣方）
                                neutral_ticks = interval_ticks[interval_ticks['tick_type'] == 0]  # 無法判定
                                
                                buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                neutral_volume = int(neutral_ticks['volume'].sum()) if not neutral_ticks.empty else 0
                                
                                # Distribute neutral trades evenly
                                if neutral_volume > 0:
                                    buy_volume += neutral_volume // 2
                                    sell_volume += neutral_volume - (neutral_volume // 2)
                                
                                logger.info(f"🎯 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張, 中性{neutral_volume}張 (共{len(interval_ticks)}筆tick)")
                                
                            else:
                                # Use price judgment method (fallback)
                                interval_avg = (open_price + high_price + low_price + close_price) / 4
                                buy_ticks = interval_ticks[interval_ticks['close'] >= interval_avg]
                                sell_ticks = interval_ticks[interval_ticks['close'] < interval_avg]
                                
                                buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                
                                logger.info(f"📊 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張 (價格判斷法)")
                            
                            # Ensure tick total volume doesn't exceed K-bar volume
                            total_tick_volume = buy_volume + sell_volume
                            if total_tick_volume > volume and volume > 0:
                                ratio = volume / total_tick_volume
                                buy_volume = int(buy_volume * ratio)
                                sell_volume = int(sell_volume * ratio)
                                logger.info(f"🔧 調整: {row['ts'].strftime('%H:%M')} 買{buy_volume}張, 賣{sell_volume}張 (比例={ratio:.2f})")
                                
                    # Fallback: estimate when no tick data available
                    if (buy_volume is None or sell_volume is None) and volume > 0:
                        buy_volume = volume // 2
                        sell_volume = volume - buy_volume
                        logger.info(f"📊 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張 (50/50估算)")

                    # Build 5-minute K-bar data (including buy/sell volumes)
                    # Convert all numpy types to Python native types for JSON serialization
                    bar_data = {
                        "time": row['ts'].strftime('%H:%M'),
                        "datetime": row['ts'].strftime('%Y-%m-%d %H:%M:%S'),
                        "date": row['ts'].strftime('%Y-%m-%d'),
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "close": float(close_price),
                        "average_price": round(float(avg_price), 2),
                        "volume": int(volume),           # Total volume
                        "buy_volume": int(buy_volume) if buy_volume is not None else None,   # Buy volume
                        "sell_volume": int(sell_volume) if sell_volume is not None else None, # Sell volume
                        "buy_sell_ratio": round(float(buy_volume) / float(sell_volume), 2) if buy_volume and sell_volume and sell_volume > 0 else None,
                        "net_flow": int(buy_volume) - int(sell_volume) if buy_volume is not None and sell_volume is not None else None  # Net flow
                    }
                    
                    intraday_data.append(bar_data)
                    total_volume_session += int(volume)
                    if buy_volume is not None:
                        total_buy_volume += int(buy_volume)
                    if sell_volume is not None:
                        total_sell_volume += int(sell_volume)

                if not intraday_data:
                    return {"success": False, "message": f"Could not process 5-minute data for {stock_code} {session} session"}
                
                # Sort by time to ensure proper order
                intraday_data.sort(key=lambda x: x["time"])
                
                # Return complete 5-minute buy/sell volume data
                return {
                    "success": True,
                    "session": session_name,
                    "time_strategy": time_strategy,
                    "query_date": base_date.strftime('%Y-%m-%d'),
                    "available_hours": [int(h) for h in unique_hours],  # Convert numpy int to Python int
                    "data_time_range": f"{df_session.index.min().strftime('%H:%M')} - {df_session.index.max().strftime('%H:%M')}",
                    "session_summary": {
                        "total_volume": int(total_volume_session),
                        "total_buy_volume": int(total_buy_volume),
                        "total_sell_volume": int(total_sell_volume),
                        "net_flow": int(total_buy_volume) - int(total_sell_volume) if total_buy_volume and total_sell_volume else None,
                        "buy_sell_ratio": round(float(total_buy_volume) / float(total_sell_volume), 2) if total_buy_volume and total_sell_volume and total_sell_volume > 0 else None
                    },
                    "intraday_bars": intraday_data,  # Detailed 5-minute buy/sell volumes!
                    "total_bars": len(intraday_data),
                    "buy_sell_data_available": tick_df is not None and not tick_df.empty,
                    "tick_data_points": len(tick_df) if tick_df is not None else 0,
                    "timezone": "Asia/Taipei (+8)",
                    "aggregation_method": "1min -> 5min resampling with tick-based buy/sell calculation"
                }
                
            except Exception as df_error:
                logger.error(f"Error processing 5-minute data for {stock_code}: {df_error}")
                return {"success": False, "message": f"Data processing error: {str(df_error)}"}
            
        else:
            # Daily data processing (existing logic)
            return {"success": True, "message": "Daily data processing not implemented in this simplified version"}
        
    except Exception as e:
        logger.error(f"Overall error for {stock_code}: {e}")
        return {"success": False, "message": f"Technical analysis error: {str(e)}"}
    
    finally:
        logger.info(f"=== RAW DATA LOGGING END for {stock_code} ===")

@app.get("/historical/ticks/{stock_code}")
async def get_historical_ticks(
    stock_code: str, 
    date: str,
    query_type: str = "AllDay",
    time_start: str = None,
    time_end: str = None,
    last_cnt: int = 0
):
    """
    取得個股歷史 Tick 資料
    
    Args:
        stock_code: 股票代碼
        date: 查詢日期 (YYYY-MM-DD 格式，例如 "2024-01-15")
        query_type: 查詢類型 - "AllDay" (整天), "RangeTime" (時間區段), "LastCount" (最後幾筆)
        time_start: 開始時間 (HH:MM:SS 格式，query_type="RangeTime" 時使用)
        time_end: 結束時間 (HH:MM:SS 格式，query_type="RangeTime" 時使用)
        last_cnt: 最後幾筆數量 (query_type="LastCount" 時使用)
    
    Examples:
        /historical/ticks/2330?date=2024-01-15 - 取得 2330 在 2024-01-15 整天的 tick 資料
        /historical/ticks/2330?date=2024-01-15&query_type=RangeTime&time_start=09:00:00&time_end=10:00:00 - 特定時間區段
        /historical/ticks/2330?date=2024-01-15&query_type=LastCount&last_cnt=100 - 最後 100 筆
    """
    global api
    
    logger.info("="*60)
    logger.info(f"📊 開始查詢歷史 Tick 資料 - {stock_code}")
    logger.info("="*60)
    
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # 取得合約
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            return {"success": False, "message": f"Stock {stock_code} not found"}
        
        logger.info(f"📌 Contract Info:")
        logger.info(f"   - Code: {contract.code}")
        logger.info(f"   - Name: {contract.name}")
        logger.info(f"   - Exchange: {contract.exchange}")
        
        # 準備查詢參數
        logger.info(f"📌 Query Parameters:")
        logger.info(f"   - Date: {date}")
        logger.info(f"   - Query Type: {query_type}")
        logger.info(f"   - Time Start: {time_start}")
        logger.info(f"   - Time End: {time_end}")
        logger.info(f"   - Last Count: {last_cnt}")
        
        # 根據查詢類型設定參數
        query_params = {
            "contract": contract,
            "date": date,
            "timeout": 30000
        }
        
        if query_type == "RangeTime":
            if not time_start or not time_end:
                return {"success": False, "message": "time_start and time_end are required for RangeTime query"}
            query_params["query_type"] = sj.constant.TicksQueryType.RangeTime
            query_params["time_start"] = time_start
            query_params["time_end"] = time_end
            
        elif query_type == "LastCount":
            if last_cnt <= 0:
                return {"success": False, "message": "last_cnt must be greater than 0 for LastCount query"}
            query_params["query_type"] = sj.constant.TicksQueryType.LastCount
            query_params["last_cnt"] = last_cnt
            
        else:  # AllDay
            query_params["query_type"] = sj.constant.TicksQueryType.AllDay
        
        # 執行查詢
        logger.info(f"🔄 執行 API 查詢...")
        ticks = api.ticks(**query_params)
        
        # 詳細記錄原始資料
        logger.info(f"📊 原始 Ticks 物件:")
        logger.info(f"   - Type: {type(ticks)}")
        logger.info(f"   - Has ts: {hasattr(ticks, 'ts')}")
        logger.info(f"   - Has close: {hasattr(ticks, 'close')}")
        logger.info(f"   - Has volume: {hasattr(ticks, 'volume')}")
        logger.info(f"   - Has tick_type: {hasattr(ticks, 'tick_type')}")
        
        if not ticks or not hasattr(ticks, 'ts') or not ticks.ts:
            logger.warning(f"⚠️ No tick data available for {stock_code} on {date}")
            return {"success": False, "message": f"No tick data available for {stock_code} on {date}"}
        
        # 記錄資料數量
        logger.info(f"✅ 成功取得 {len(ticks.ts)} 筆 Tick 資料")
        
        # 轉換為 DataFrame 進行分析
        tick_data = {
            'ts': ticks.ts,
            'close': ticks.close,
            'volume': ticks.volume,
            'bid_price': ticks.bid_price,
            'bid_volume': ticks.bid_volume,
            'ask_price': ticks.ask_price,
            'ask_volume': ticks.ask_volume
        }
        
        # 檢查是否有 tick_type
        if hasattr(ticks, 'tick_type') and ticks.tick_type:
            tick_data['tick_type'] = ticks.tick_type
            
        df = pd.DataFrame(tick_data)
        df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(TW_TZ)
        
        # 詳細統計資訊
        logger.info(f"📊 資料統計:")
        logger.info(f"   - 時間範圍: {df['ts'].min()} ~ {df['ts'].max()}")
        logger.info(f"   - 總成交量: {df['volume'].sum():,} 張")
        logger.info(f"   - 價格範圍: {df['close'].min()} ~ {df['close'].max()}")
        logger.info(f"   - 平均價格: {df['close'].mean():.2f}")
        
        if 'tick_type' in df.columns:
            buy_volume = df[df['tick_type'] == 1]['volume'].sum()
            sell_volume = df[df['tick_type'] == 2]['volume'].sum()
            neutral_volume = df[df['tick_type'] == 0]['volume'].sum()
            
            logger.info(f"   - 外盤量 (買): {buy_volume:,} 張")
            logger.info(f"   - 內盤量 (賣): {sell_volume:,} 張")
            logger.info(f"   - 無法判定: {neutral_volume:,} 張")
            logger.info(f"   - 買賣比: {buy_volume/sell_volume:.2f}" if sell_volume > 0 else "   - 買賣比: N/A")
        
        # 記錄前 10 筆詳細資料
        logger.info(f"📋 前 10 筆 Tick 詳細資料:")
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            tick_type_str = ""
            if 'tick_type' in df.columns:
                tick_type_map = {0: "無法判定", 1: "外盤(買)", 2: "內盤(賣)"}
                tick_type_str = f", 類型: {tick_type_map.get(row['tick_type'], 'Unknown')}"
            
            logger.info(f"   [{i+1}] {row['ts'].strftime('%H:%M:%S.%f')[:-3]}: "
                       f"價格={row['close']}, 量={row['volume']}, "
                       f"買價={row['bid_price']}, 買量={row['bid_volume']}, "
                       f"賣價={row['ask_price']}, 賣量={row['ask_volume']}"
                       f"{tick_type_str}")
        
        # 準備回傳資料
        processed_ticks = []
        for idx, row in df.iterrows():
            tick = {
                "time": row['ts'].strftime('%H:%M:%S.%f')[:-3],
                "datetime": row['ts'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                "close": float(row['close']),
                "volume": int(row['volume']),
                "bid_price": float(row['bid_price']),
                "bid_volume": int(row['bid_volume']),
                "ask_price": float(row['ask_price']),
                "ask_volume": int(row['ask_volume'])
            }
            
            if 'tick_type' in df.columns:
                tick['tick_type'] = int(row['tick_type'])
                tick['tick_type_name'] = {0: "無法判定", 1: "外盤", 2: "內盤"}.get(row['tick_type'], "Unknown")
            
            processed_ticks.append(tick)
        
        # 計算摘要統計
        summary = {
            "total_ticks": len(processed_ticks),
            "total_volume": int(df['volume'].sum()),
            "price_high": float(df['close'].max()),
            "price_low": float(df['close'].min()),
            "price_avg": round(float(df['close'].mean()), 2),
            "time_start": df['ts'].min().strftime('%Y-%m-%d %H:%M:%S'),
            "time_end": df['ts'].max().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if 'tick_type' in df.columns:
            summary["buy_volume"] = int(df[df['tick_type'] == 1]['volume'].sum())
            summary["sell_volume"] = int(df[df['tick_type'] == 2]['volume'].sum())
            summary["neutral_volume"] = int(df[df['tick_type'] == 0]['volume'].sum())
            summary["buy_sell_ratio"] = round(summary["buy_volume"] / summary["sell_volume"], 2) if summary["sell_volume"] > 0 else None
        
        logger.info(f"✅ 資料處理完成，準備回傳 {len(processed_ticks)} 筆 Tick 資料")
        logger.info("="*60)
        
        return {
            "success": True,
            "stock_code": stock_code,
            "stock_name": contract.name,
            "query_date": date,
            "query_type": query_type,
            "summary": summary,
            "ticks": processed_ticks,
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"❌ 錯誤發生: {e}")
        logger.error(f"   錯誤類型: {type(e).__name__}")
        logger.error(f"   Stack trace:\n{traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/historical/kbars/{stock_code}")
async def get_historical_kbars(
    stock_code: str,
    start: str,
    end: str
):
    """
    取得個股歷史 K 線資料 (1分鐘 K 線)
    
    Args:
        stock_code: 股票代碼
        start: 開始日期 (YYYY-MM-DD 格式)
        end: 結束日期 (YYYY-MM-DD 格式)
    
    Examples:
        /historical/kbars/2330?start=2024-01-15&end=2024-01-16
    """
    global api
    
    logger.info("="*60)
    logger.info(f"📊 開始查詢歷史 K 線資料 - {stock_code}")
    logger.info("="*60)
    
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # 取得合約
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            return {"success": False, "message": f"Stock {stock_code} not found"}
        
        logger.info(f"📌 Contract Info:")
        logger.info(f"   - Code: {contract.code}")
        logger.info(f"   - Name: {contract.name}")
        logger.info(f"   - Exchange: {contract.exchange}")
        
        logger.info(f"📌 Query Parameters:")
        logger.info(f"   - Start Date: {start}")
        logger.info(f"   - End Date: {end}")
        
        # 執行查詢
        logger.info(f"🔄 執行 API 查詢...")
        kbars = api.kbars(
            contract=contract,
            start=start,
            end=end,
            timeout=30000
        )
        
        # 詳細記錄原始資料
        logger.info(f"📊 原始 Kbars 物件:")
        logger.info(f"   - Type: {type(kbars)}")
        logger.info(f"   - Has ts: {hasattr(kbars, 'ts')}")
        logger.info(f"   - Has Open: {hasattr(kbars, 'Open')}")
        logger.info(f"   - Has High: {hasattr(kbars, 'High')}")
        logger.info(f"   - Has Low: {hasattr(kbars, 'Low')}")
        logger.info(f"   - Has Close: {hasattr(kbars, 'Close')}")
        logger.info(f"   - Has Volume: {hasattr(kbars, 'Volume')}")
        
        if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
            logger.warning(f"⚠️ No K-bar data available for {stock_code} from {start} to {end}")
            return {"success": False, "message": f"No K-bar data available for {stock_code} from {start} to {end}"}
        
        # 記錄資料數量
        logger.info(f"✅ 成功取得 {len(kbars.ts)} 根 K 線資料")
        
        # 轉換為 DataFrame
        df = pd.DataFrame({
            'ts': kbars.ts,
            'Open': kbars.Open,
            'High': kbars.High,
            'Low': kbars.Low,
            'Close': kbars.Close,
            'Volume': kbars.Volume
        })
        
        df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(TW_TZ)
        
        # 詳細統計資訊
        logger.info(f"📊 資料統計:")
        logger.info(f"   - 時間範圍: {df['ts'].min()} ~ {df['ts'].max()}")
        logger.info(f"   - 總成交量: {df['Volume'].sum():,} 張")
        logger.info(f"   - 最高價: {df['High'].max()}")
        logger.info(f"   - 最低價: {df['Low'].min()}")
        logger.info(f"   - 平均收盤價: {df['Close'].mean():.2f}")
        
        # 計算每日統計
        df['date'] = df['ts'].dt.date
        daily_stats = df.groupby('date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        logger.info(f"📊 每日統計:")
        for date, row in daily_stats.iterrows():
            logger.info(f"   {date}: 開={row['Open']}, 高={row['High']}, "
                       f"低={row['Low']}, 收={row['Close']}, 量={row['Volume']:,}")
        
        # 記錄前 10 根 K 線詳細資料
        logger.info(f"📋 前 10 根 K 線詳細資料:")
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            logger.info(f"   [{i+1}] {row['ts'].strftime('%Y-%m-%d %H:%M')}: "
                       f"開={row['Open']}, 高={row['High']}, "
                       f"低={row['Low']}, 收={row['Close']}, 量={row['Volume']}")
        
        # 準備回傳資料
        processed_kbars = []
        for idx, row in df.iterrows():
            kbar = {
                "time": row['ts'].strftime('%H:%M'),
                "datetime": row['ts'].strftime('%Y-%m-%d %H:%M:%S'),
                "date": row['ts'].strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']),
                "average": round((row['Open'] + row['High'] + row['Low'] + row['Close']) / 4, 2)
            }
            processed_kbars.append(kbar)
        
        # 計算摘要統計
        summary = {
            "total_bars": len(processed_kbars),
            "total_volume": int(df['Volume'].sum()),
            "price_high": float(df['High'].max()),
            "price_low": float(df['Low'].min()),
            "price_open": float(df.iloc[0]['Open']) if len(df) > 0 else None,
            "price_close": float(df.iloc[-1]['Close']) if len(df) > 0 else None,
            "price_change": float(df.iloc[-1]['Close'] - df.iloc[0]['Open']) if len(df) > 0 else None,
            "price_change_pct": round(((df.iloc[-1]['Close'] - df.iloc[0]['Open']) / df.iloc[0]['Open'] * 100), 2) if len(df) > 0 and df.iloc[0]['Open'] != 0 else None,
            "time_start": df['ts'].min().strftime('%Y-%m-%d %H:%M:%S'),
            "time_end": df['ts'].max().strftime('%Y-%m-%d %H:%M:%S'),
            "unique_days": df['date'].nunique()
        }
        
        logger.info(f"✅ 資料處理完成，準備回傳 {len(processed_kbars)} 根 K 線資料")
        logger.info("="*60)
        
        return {
            "success": True,
            "stock_code": stock_code,
            "stock_name": contract.name,
            "start_date": start,
            "end_date": end,
            "summary": summary,
            "kbars": processed_kbars,
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"❌ 錯誤發生: {e}")
        logger.error(f"   錯誤類型: {type(e).__name__}")
        logger.error(f"   Stack trace:\n{traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}"}


@app.get("/historical/analysis/{stock_code}")
async def get_historical_analysis(
    stock_code: str,
    date: str,
    interval: str = "5min"
):
    """
    取得個股歷史資料的詳細分析 (結合 Ticks 和 K 線)
    
    Args:
        stock_code: 股票代碼
        date: 查詢日期 (YYYY-MM-DD 格式)
        interval: K線間隔 - "1min", "5min", "15min", "30min", "60min"
    
    Examples:
        /historical/analysis/2330?date=2024-01-15&interval=5min
    """
    global api
    
    logger.info("="*60)
    logger.info(f"📊 開始執行歷史資料詳細分析 - {stock_code}")
    logger.info("="*60)
    
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # 取得合約
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            return {"success": False, "message": f"Stock {stock_code} not found"}
        
        logger.info(f"📌 分析參數:")
        logger.info(f"   - Stock: {stock_code} ({contract.name})")
        logger.info(f"   - Date: {date}")
        logger.info(f"   - Interval: {interval}")
        
        # 1. 取得 Tick 資料
        logger.info(f"🔄 Step 1: 取得 Tick 資料...")
        ticks = api.ticks(
            contract=contract,
            date=date,
            query_type=sj.constant.TicksQueryType.AllDay,
            timeout=30000
        )
        
        if not ticks or not hasattr(ticks, 'ts') or not ticks.ts:
            return {"success": False, "message": f"No tick data available for {stock_code} on {date}"}
        
        # 轉換 Tick 資料為 DataFrame
        tick_data = {
            'ts': ticks.ts,
            'close': ticks.close,
            'volume': ticks.volume,
            'bid_price': ticks.bid_price,
            'bid_volume': ticks.bid_volume,
            'ask_price': ticks.ask_price,
            'ask_volume': ticks.ask_volume
        }
        
        if hasattr(ticks, 'tick_type') and ticks.tick_type:
            tick_data['tick_type'] = ticks.tick_type
        
        tick_df = pd.DataFrame(tick_data)
        tick_df['ts'] = pd.to_datetime(tick_df['ts'], utc=True).dt.tz_convert(TW_TZ)
        
        logger.info(f"✅ 取得 {len(tick_df)} 筆 Tick 資料")
        
        # 2. 取得 K 線資料
        logger.info(f"🔄 Step 2: 取得 K 線資料...")
        kbars = api.kbars(
            contract=contract,
            start=date,
            end=date,
            timeout=30000
        )
        
        if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
            return {"success": False, "message": f"No K-bar data available for {stock_code} on {date}"}
        
        # 轉換 K 線資料為 DataFrame
        kbar_df = pd.DataFrame({
            'ts': kbars.ts,
            'Open': kbars.Open,
            'High': kbars.High,
            'Low': kbars.Low,
            'Close': kbars.Close,
            'Volume': kbars.Volume
        })
        
        kbar_df['ts'] = pd.to_datetime(kbar_df['ts'], utc=True).dt.tz_convert(TW_TZ)
        kbar_df.set_index('ts', inplace=True)
        
        logger.info(f"✅ 取得 {len(kbar_df)} 根 1 分鐘 K 線")
        
        # 3. 重新採樣 K 線到指定間隔
        logger.info(f"🔄 Step 3: 重新採樣 K 線到 {interval} 間隔...")
        
        interval_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '60min': '60T'
        }
        
        if interval != '1min':
            resampled_kbar = kbar_df.resample(interval_map[interval], label='right', closed='right').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            resampled_kbar.reset_index(inplace=True)
        else:
            resampled_kbar = kbar_df.reset_index()
        
        logger.info(f"✅ 產生 {len(resampled_kbar)} 根 {interval} K 線")
        
        # 4. 分析買賣力道
        logger.info(f"🔄 Step 4: 分析買賣力道...")
        
        analysis_results = []
        
        for idx, kbar in resampled_kbar.iterrows():
            # 找出這根 K 線時間範圍內的所有 Ticks
            if idx == 0:
                interval_start = kbar['ts'] - pd.Timedelta(interval_map[interval])
            else:
                interval_start = resampled_kbar.iloc[idx-1]['ts']
            interval_end = kbar['ts']
            
            interval_ticks = tick_df[
                (tick_df['ts'] > interval_start) & 
                (tick_df['ts'] <= interval_end)
            ]
            
            # 計算買賣量
            buy_volume = 0
            sell_volume = 0
            neutral_volume = 0
            
            if not interval_ticks.empty:
                if 'tick_type' in interval_ticks.columns:
                    buy_volume = int(interval_ticks[interval_ticks['tick_type'] == 1]['volume'].sum())
                    sell_volume = int(interval_ticks[interval_ticks['tick_type'] == 2]['volume'].sum())
                    neutral_volume = int(interval_ticks[interval_ticks['tick_type'] == 0]['volume'].sum())
                else:
                    # 使用價格判斷法
                    avg_price = (kbar['Open'] + kbar['High'] + kbar['Low'] + kbar['Close']) / 4
                    buy_volume = int(interval_ticks[interval_ticks['close'] >= avg_price]['volume'].sum())
                    sell_volume = int(interval_ticks[interval_ticks['close'] < avg_price]['volume'].sum())
            
            # 計算技術指標
            price_range = float(kbar['High'] - kbar['Low'])
            body_size = abs(float(kbar['Close'] - kbar['Open']))
            upper_shadow = float(kbar['High'] - max(kbar['Open'], kbar['Close']))
            lower_shadow = float(min(kbar['Open'], kbar['Close']) - kbar['Low'])
            
            bar_analysis = {
                "time": kbar['ts'].strftime('%H:%M'),
                "datetime": kbar['ts'].strftime('%Y-%m-%d %H:%M:%S'),
                "ohlc": {
                    "open": float(kbar['Open']),
                    "high": float(kbar['High']),
                    "low": float(kbar['Low']),
                    "close": float(kbar['Close'])
                },
                "volume_analysis": {
                    "total": int(kbar['Volume']),
                    "buy": buy_volume,
                    "sell": sell_volume,
                    "neutral": neutral_volume,
                    "buy_ratio": round(buy_volume / kbar['Volume'] * 100, 2) if kbar['Volume'] > 0 else 0,
                    "sell_ratio": round(sell_volume / kbar['Volume'] * 100, 2) if kbar['Volume'] > 0 else 0,
                    "net_flow": buy_volume - sell_volume,
                    "buy_sell_ratio": round(buy_volume / sell_volume, 2) if sell_volume > 0 else None
                },
                "price_analysis": {
                    "change": round(float(kbar['Close'] - kbar['Open']), 2),
                    "change_pct": round((kbar['Close'] - kbar['Open']) / kbar['Open'] * 100, 2) if kbar['Open'] != 0 else 0,
                    "average": round((kbar['Open'] + kbar['High'] + kbar['Low'] + kbar['Close']) / 4, 2),
                    "range": round(price_range, 2),
                    "body_size": round(body_size, 2),
                    "upper_shadow": round(upper_shadow, 2),
                    "lower_shadow": round(lower_shadow, 2),
                    "body_ratio": round(body_size / price_range * 100, 2) if price_range > 0 else 0
                },
                "tick_count": len(interval_ticks),
                "trend": "bullish" if kbar['Close'] > kbar['Open'] else "bearish" if kbar['Close'] < kbar['Open'] else "neutral"
            }
            
            analysis_results.append(bar_analysis)
        
        # 5. 計算整體統計
        logger.info(f"🔄 Step 5: 計算整體統計...")
        
        total_buy = sum(bar['volume_analysis']['buy'] for bar in analysis_results)
        total_sell = sum(bar['volume_analysis']['sell'] for bar in analysis_results)
        total_neutral = sum(bar['volume_analysis']['neutral'] for bar in analysis_results)
        
        overall_summary = {
            "date": date,
            "interval": interval,
            "total_bars": len(analysis_results),
            "total_ticks": len(tick_df),
            "price_summary": {
                "open": float(resampled_kbar.iloc[0]['Open']) if len(resampled_kbar) > 0 else None,
                "close": float(resampled_kbar.iloc[-1]['Close']) if len(resampled_kbar) > 0 else None,
                "high": float(resampled_kbar['High'].max()),
                "low": float(resampled_kbar['Low'].min()),
                "change": float(resampled_kbar.iloc[-1]['Close'] - resampled_kbar.iloc[0]['Open']) if len(resampled_kbar) > 0 else None,
                "change_pct": round((resampled_kbar.iloc[-1]['Close'] - resampled_kbar.iloc[0]['Open']) / resampled_kbar.iloc[0]['Open'] * 100, 2) if len(resampled_kbar) > 0 and resampled_kbar.iloc[0]['Open'] != 0 else None
            },
            "volume_summary": {
                "total": int(resampled_kbar['Volume'].sum()),
                "buy": total_buy,
                "sell": total_sell,
                "neutral": total_neutral,
                "buy_ratio": round(total_buy / resampled_kbar['Volume'].sum() * 100, 2) if resampled_kbar['Volume'].sum() > 0 else 0,
                "sell_ratio": round(total_sell / resampled_kbar['Volume'].sum() * 100, 2) if resampled_kbar['Volume'].sum() > 0 else 0,
                "net_flow": total_buy - total_sell,
                "buy_sell_ratio": round(total_buy / total_sell, 2) if total_sell > 0 else None
            },
            "time_range": {
                "start": tick_df['ts'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": tick_df['ts'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 記錄分析摘要
        logger.info(f"📊 分析完成摘要:")
        logger.info(f"   - 總成交量: {overall_summary['volume_summary']['total']:,} 張")
        logger.info(f"   - 買入量: {overall_summary['volume_summary']['buy']:,} 張 ({overall_summary['volume_summary']['buy_ratio']:.2f}%)")
        logger.info(f"   - 賣出量: {overall_summary['volume_summary']['sell']:,} 張 ({overall_summary['volume_summary']['sell_ratio']:.2f}%)")
        logger.info(f"   - 淨流入: {overall_summary['volume_summary']['net_flow']:,} 張")
        if overall_summary['price_summary']['change'] is not None:
            logger.info(f"   - 價格變化: {overall_summary['price_summary']['change']:.2f} ({overall_summary['price_summary']['change_pct']:.2f}%)")
        
        logger.info(f"✅ 歷史資料分析完成")
        logger.info("="*60)
        
        return {
            "success": True,
            "stock_code": stock_code,
            "stock_name": contract.name,
            "summary": overall_summary,
            "detailed_analysis": analysis_results,
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"❌ 錯誤發生: {e}")
        logger.error(f"   錯誤類型: {type(e).__name__}")
        logger.error(f"   Stack trace:\n{traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}"}
        
        if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
            return {"success": False, "message": f"No K-bar data available for {stock_code} on {date}"}
        
        # 轉換 K 線資料為 DataFrame
        kbar_df = pd.DataFrame({
            'ts': kbars.ts,
            'Open': kbars.Open,
            'High': kbars.High,
            'Low': kbars.Low,
            'Close': kbars.Close,
            'Volume': kbars.Volume
        })
        
        kbar_df['ts'] = pd.to_datetime(kbar_df['ts'], utc=True).dt.tz_convert(TW_TZ)
        kbar_df.set_index('ts', inplace=True)
        
        logger.info(f"✅ 取得 {len(kbar_df)} 根 1 分鐘 K 線")
        
        # 3. 重新採樣 K 線到指定間隔
        logger.info(f"🔄 Step 3: 重新採樣 K 線到 {interval} 間隔...")
        
        interval_map = {
            '1min': '1T',
            '5min': '5T',
            '15min': '15T',
            '30min': '30T',
            '60min': '60T'
        }
        
        if interval != '1min':
            resampled_kbar = kbar_df.resample(interval_map[interval], label='right', closed='right').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            resampled_kbar.reset_index(inplace=True)
        else:
            resampled_kbar = kbar_df.reset_index()
        
        logger.info(f"✅ 產生 {len(resampled_kbar)} 根 {interval} K 線")
        
        # 4. 分析買賣力道
        logger.info(f"🔄 Step 4: 分析買賣力道...")
        
        analysis_results = []
        
        for idx, kbar in resampled_kbar.iterrows():
            # 找出這根 K 線時間範圍內的所有 Ticks
            if idx == 0:
                interval_start = kbar['ts'] - pd.Timedelta(interval_map[interval])
            else:
                interval_start = resampled_kbar.iloc[idx-1]['ts']
            interval_end = kbar['ts']
            
            interval_ticks = tick_df[
                (tick_df['ts'] > interval_start) & 
                (tick_df['ts'] <= interval_end)
            ]
            
            # 計算買賣量
            buy_volume = 0
            sell_volume = 0
            neutral_volume = 0
            
            if not interval_ticks.empty:
                if 'tick_type' in interval_ticks.columns:
                    buy_volume = int(interval_ticks[interval_ticks['tick_type'] == 1]['volume'].sum())
                    sell_volume = int(interval_ticks[interval_ticks['tick_type'] == 2]['volume'].sum())
                    neutral_volume = int(interval_ticks[interval_ticks['tick_type'] == 0]['volume'].sum())
                else:
                    # 使用價格判斷法
                    avg_price = (kbar['Open'] + kbar['High'] + kbar['Low'] + kbar['Close']) / 4
                    buy_volume = int(interval_ticks[interval_ticks['close'] >= avg_price]['volume'].sum())
                    sell_volume = int(interval_ticks[interval_ticks['close'] < avg_price]['volume'].sum())
            
            # 計算技術指標
            price_range = float(kbar['High'] - kbar['Low'])
            body_size = abs(float(kbar['Close'] - kbar['Open']))
            upper_shadow = float(kbar['High'] - max(kbar['Open'], kbar['Close']))
            lower_shadow = float(min(kbar['Open'], kbar['Close']) - kbar['Low'])
            
            bar_analysis = {
                "time": kbar['ts'].strftime('%H:%M'),
                "datetime": kbar['ts'].strftime('%Y-%m-%d %H:%M:%S'),
                "ohlc": {
                    "open": float(kbar['Open']),
                    "high": float(kbar['High']),
                    "low": float(kbar['Low']),
                    "close": float(kbar['Close'])
                },
                "volume_analysis": {
                    "total": int(kbar['Volume']),
                    "buy": buy_volume,
                    "sell": sell_volume,
                    "neutral": neutral_volume,
                    "buy_ratio": round(buy_volume / kbar['Volume'] * 100, 2) if kbar['Volume'] > 0 else 0,
                    "sell_ratio": round(sell_volume / kbar['Volume'] * 100, 2) if kbar['Volume'] > 0 else 0,
                    "net_flow": buy_volume - sell_volume,
                    "buy_sell_ratio": round(buy_volume / sell_volume, 2) if sell_volume > 0 else None
                },
                "price_analysis": {
                    "change": round(float(kbar['Close'] - kbar['Open']), 2),
                    "change_pct": round((kbar['Close'] - kbar['Open']) / kbar['Open'] * 100, 2) if kbar['Open'] != 0 else 0,
                    "average": round((kbar['Open'] + kbar['High'] + kbar['Low'] + kbar['Close']) / 4, 2),
                    "range": round(price_range, 2),
                    "body_size": round(body_size, 2),
                    "upper_shadow": round(upper_shadow, 2),
                    "lower_shadow": round(lower_shadow, 2),
                    "body_ratio": round(body_size / price_range * 100, 2) if price_range > 0 else 0
                },
                "tick_count": len(interval_ticks),
                "trend": "bullish" if kbar['Close'] > kbar['Open'] else "bearish" if kbar['Close'] < kbar['Open'] else "neutral"
            }
            
            analysis_results.append(bar_analysis)
        
        # 5. 計算整體統計
        logger.info(f"🔄 Step 5: 計算整體統計...")
        
        total_buy = sum(bar['volume_analysis']['buy'] for bar in analysis_results)
        total_sell = sum(bar['volume_analysis']['sell'] for bar in analysis_results)
        total_neutral = sum(bar['volume_analysis']['neutral'] for bar in analysis_results)
        
        overall_summary = {
            "date": date,
            "interval": interval,
            "total_bars": len(analysis_results),
            "total_ticks": len(tick_df),
            "price_summary": {
                "open": float(resampled_kbar.iloc[0]['Open']) if len(resampled_kbar) > 0 else None,
                "close": float(resampled_kbar.iloc[-1]['Close']) if len(resampled_kbar) > 0 else None,
                "high": float(resampled_kbar['High'].max()),
                "low": float(resampled_kbar['Low'].min()),
                "change": float(resampled_kbar.iloc[-1]['Close'] - resampled_kbar.iloc[0]['Open']) if len(resampled_kbar) > 0 else None,
                "change_pct": round((resampled_kbar.iloc[-1]['Close'] - resampled_kbar.iloc[0]['Open']) / resampled_kbar.iloc[0]['Open'] * 100, 2) if len(resampled_kbar) > 0 and resampled_kbar.iloc[0]['Open'] != 0 else None
            },
            "volume_summary": {
                "total": int(resampled_kbar['Volume'].sum()),
                "buy": total_buy,
                "sell": total_sell,
                "neutral": total_neutral,
                "buy_ratio": round(total_buy / resampled_kbar['Volume'].sum() * 100, 2) if resampled_kbar['Volume'].sum() > 0 else 0,
                "sell_ratio": round(total_sell / resampled_kbar['Volume'].sum() * 100, 2) if resampled_kbar['Volume'].sum() > 0 else 0,
                "net_flow": total_buy - total_sell,
                "buy_sell_ratio": round(total_buy / total_sell, 2) if total_sell > 0 else None
            },
            "time_range": {
                "start": tick_df['ts'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": tick_df['ts'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # 記錄分析摘要
        logger.info(f"📊 分析完成摘要:")
        logger.info(f"   - 總成交量: {overall_summary['volume_summary']['total']:,} 張")
        logger.info(f"   - 買入量: {overall_summary['volume_summary']['buy']:,} 張 ({overall_summary['volume_summary']['buy_ratio']:.2f}%)")
        logger.info(f"   - 賣出量: {overall_summary['volume_summary']['sell']:,} 張 ({overall_summary['volume_summary']['sell_ratio']:.2f}%)")
        logger.info(f"   - 淨流入: {overall_summary['volume_summary']['net_flow']:,} 張")
        if overall_summary['price_summary']['change'] is not None:
            logger.info(f"   - 價格變化: {overall_summary['price_summary']['change']:.2f} ({overall_summary['price_summary']['change_pct']:.2f}%)")
        
        logger.info(f"✅ 歷史資料分析完成")
        logger.info("="*60)
        
        return {
            "success": True,
            "stock_code": stock_code,
            "stock_name": contract.name,
            "summary": overall_summary,
            "detailed_analysis": analysis_results,
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"❌ 錯誤發生: {e}")
        logger.error(f"   錯誤類型: {type(e).__name__}")
        logger.error(f"   Stack trace:\n{traceback.format_exc()}")
        return {"success": False, "message": f"Error: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    actual_connected = check_connection()
    current_time_tw = datetime.now(TW_TZ)
    return {
        "status": "healthy",
        "api_connected": actual_connected,
        "login_status_var": login_status,
        "auto_login_configured": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
        "current_time_tw": current_time_tw.strftime('%Y-%m-%d %H:%M:%S'),
        "timezone": "Asia/Taipei (+8)",
        "timestamp": str(datetime.now())
    }

@app.post("/retry-login")
async def retry_auto_login():
    """Retry auto-login using environment variables"""
    global api, login_status
    try:
        if not api:
            api = sj.Shioaji()
            
        api_key = os.getenv("SHIOAJI_API_KEY")
        secret_key = os.getenv("SHIOAJI_SECRET_KEY")
        
        if not api_key or not secret_key:
            return {
                "success": False,
                "message": "Environment variables not set",
                "env_check": {
                    "SHIOAJI_API_KEY": "SET" if api_key else "NOT SET",
                    "SHIOAJI_SECRET_KEY": "SET" if secret_key else "NOT SET",
                    "SHIOAJI_PERSON_ID": "SET" if os.getenv("SHIOAJI_PERSON_ID") else "NOT SET"
                }
            }
        
        logger.info(f"Retry - API Key length: {len(api_key)}")
        logger.info(f"Retry - Secret Key length: {len(secret_key)}")
        
        api_key_clean = api_key.strip().strip('"').strip("'")
        secret_key_clean = secret_key.strip().strip('"').strip("'")
        
        logger.info(f"Retry - Cleaned API Key length: {len(api_key_clean)}")
        logger.info(f"Retry - Cleaned Secret Key length: {len(secret_key_clean)}")
        
        login_attempts = [
            (api_key_clean, secret_key_clean, "original"),
            (api_key_clean.encode('utf-8').decode('unicode_escape'), 
             secret_key_clean.encode('utf-8').decode('unicode_escape'), "unicode_escape"),
        ]
        
        for api_key_attempt, secret_key_attempt, method in login_attempts:
            try:
                logger.info(f"Retrying login with method: {method}")
                accounts = api.login(
                    api_key=api_key_attempt,
                    secret_key=secret_key_attempt,
                    fetch_contract=True,
                    subscribe_trade=True
                )
                
                if accounts:
                    login_status = True
                    return {
                        "success": True,
                        "message": f"Auto-login retry successful with method: {method}",
                        "accounts": [acc.account_id for acc in accounts],
                        "stock_account": api.stock_account.account_id if api.stock_account else None,
                        "futopt_account": api.futopt_account.account_id if api.futopt_account else None
                    }
                    
            except Exception as e:
                logger.warning(f"Login retry with {method} failed: {e}")
                continue
        
        login_status = False
        return {
            "success": False,
            "message": "All login retry attempts failed"
        }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Retry login error: {str(e)}"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
