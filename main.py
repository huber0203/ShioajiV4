from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shioaji as sj
import os
from typing import Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shioaji Trading API",
    description="Trading API with technical indicators using Shioaji",
    version="1.0.0"
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
        "message": "Shioaji Trading API with Auto-Login",
        "version": "1.0.0",
        "status": "running",
        "connected": login_status,
        "auto_login": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
        "env_status": {
            "SHIOAJI_API_KEY": "SET" if os.getenv("SHIOAJI_API_KEY") else "NOT SET",
            "SHIOAJI_SECRET_KEY": "SET" if os.getenv("SHIOAJI_SECRET_KEY") else "NOT SET",
            "SHIOAJI_PERSON_ID": "SET" if os.getenv("SHIOAJI_PERSON_ID") else "NOT SET"
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
            # 🔴 修正：5分鐘資料需要指定日期範圍
            date_str = base_date.strftime('%Y-%m-%d')
            
            logger.info(f"Getting intraday kbars for date: {date_str}")
            
            try:
                # Get intraday 1-minute data with date range
                kbars = api.kbars(
                    contract=contract,
                    start=date_str,  # 🔴 加入 start 參數
                    end=date_str,    # 🔴 加入 end 參數（同一天）
                    timeout=30000
                )
                
                logger.info(f"Raw kbars object type: {type(kbars)}")
                
                if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
                    logger.info(f"No kbars data available for {date_str}")
                    return {"success": False, "message": f"No intraday data available for {stock_code} on {date_str}"}
                
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
                
                # 🔴 修正：正確轉換時間戳記（nanosecond to datetime）
                df['ts'] = pd.to_datetime(df['ts'], unit='ns')
                df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert(TW_TZ)
                
                # Analyze available time range
                min_hour = df['ts'].dt.hour.min()
                max_hour = df['ts'].dt.hour.max()
                unique_hours = sorted(df['ts'].dt.hour.unique())
                
                logger.info(f"📊 Available time range: {min_hour}:00 - {max_hour}:00")
                logger.info(f"📊 Available hours: {unique_hours}")
                logger.info(f"📊 Total data points: {len(df)}")
                logger.info(f"📊 Date range: {df['ts'].min()} to {df['ts'].max()}")
                
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
                        "message": f"No data available for {stock_code} {session} session on {date_str}",
                        "available_hours": [int(h) for h in unique_hours],
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
                        # 🔴 修正：正確轉換 tick 時間戳記
                        tick_df['ts'] = pd.to_datetime(tick_df['ts'], unit='ns')
                        tick_df['ts'] = tick_df['ts'].dt.tz_localize('UTC').dt.tz_convert(TW_TZ)
                        
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
                                
                                logger.debug(f"🎯 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張, 中性{neutral_volume}張")
                                
                            else:
                                # Use price judgment method (fallback)
                                interval_avg = (open_price + high_price + low_price + close_price) / 4
                                buy_ticks = interval_ticks[interval_ticks['close'] >= interval_avg]
                                sell_ticks = interval_ticks[interval_ticks['close'] < interval_avg]
                                
                                buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                
                                logger.debug(f"📊 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張 (價格判斷法)")
                            
                            # Ensure tick total volume doesn't exceed K-bar volume
                            total_tick_volume = buy_volume + sell_volume
                            if total_tick_volume > volume and volume > 0:
                                ratio = volume / total_tick_volume
                                buy_volume = int(buy_volume * ratio)
                                sell_volume = int(sell_volume * ratio)
                                logger.debug(f"🔧 調整: {row['ts'].strftime('%H:%M')} 買{buy_volume}張, 賣{sell_volume}張")
                                
                    # Fallback: estimate when no tick data available
                    if (buy_volume is None or sell_volume is None) and volume > 0:
                        buy_volume = volume // 2
                        sell_volume = volume - buy_volume
                        logger.debug(f"📊 {row['ts'].strftime('%H:%M')}: 買{buy_volume}張, 賣{sell_volume}張 (50/50估算)")

                    # Build 5-minute K-bar data (including buy/sell volumes)
                    bar_data = {
                        "time": row['ts'].strftime('%H:%M'),
                        "datetime": row['ts'].strftime('%Y-%m-%d %H:%M:%S'),
                        "date": row['ts'].strftime('%Y-%m-%d'),
                        "open": float(open_price),
                        "high": float(high_price),
                        "low": float(low_price),
                        "close": float(close_price),
                        "average_price": round(float(avg_price), 2),
                        "volume": int(volume),
                        "buy_volume": int(buy_volume) if buy_volume is not None else None,
                        "sell_volume": int(sell_volume) if sell_volume is not None else None,
                        "buy_sell_ratio": round(float(buy_volume) / float(sell_volume), 2) if buy_volume and sell_volume and sell_volume > 0 else None,
                        "net_flow": int(buy_volume) - int(sell_volume) if buy_volume is not None and sell_volume is not None else None
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
                    "available_hours": [int(h) for h in unique_hours],
                    "data_time_range": f"{df_session.index.min().strftime('%H:%M')} - {df_session.index.max().strftime('%H:%M')}",
                    "session_summary": {
                        "total_volume": int(total_volume_session),
                        "total_buy_volume": int(total_buy_volume),
                        "total_sell_volume": int(total_sell_volume),
                        "net_flow": int(total_buy_volume) - int(total_sell_volume) if total_buy_volume and total_sell_volume else None,
                        "buy_sell_ratio": round(float(total_buy_volume) / float(total_sell_volume), 2) if total_buy_volume and total_sell_volume and total_sell_volume > 0 else None
                    },
                    "intraday_bars": intraday_data,
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
            # 🔴 修正：日線資料需要指定日期範圍
            if target_date:
                # For specific date, get data around that date (50 days before to that date)
                end_date_str = target_date.strftime('%Y-%m-%d')
                start_date = target_date - timedelta(days=50)
                start_date_str = start_date.strftime('%Y-%m-%d')
            else:
                # For current date, get last 50 days
                now_tw = datetime.now(TW_TZ)
                end_date_str = now_tw.strftime('%Y-%m-%d')
                start_date = now_tw.date() - timedelta(days=50)
                start_date_str = start_date.strftime('%Y-%m-%d')
            
            logger.info(f"Getting daily kbars from {start_date_str} to {end_date_str}")
            
            try:
                # 🔴 修正：加入 start 和 end 參數
                kbars = api.kbars(
                    contract=contract,
                    start=start_date_str,  # 🔴 加入 start 參數
                    end=end_date_str,      # 🔴 加入 end 參數
                    timeout=30000
                )
                
                logger.info(f"Daily raw kbars object type: {type(kbars)}")
                
                if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
                    return {"success": False, "message": f"No historical data available for {stock_code} from {start_date_str} to {end_date_str}"}
                
                try:
                    df = pd.DataFrame({
                        'ts': kbars.ts,
                        'Open': kbars.Open,
                        'High': kbars.High,
                        'Low': kbars.Low,
                        'Close': kbars.Close,
                        'Volume': kbars.Volume
                    })
                    
                    # 🔴 修正：正確轉換時間戳記（nanosecond to datetime）
                    df['ts'] = pd.to_datetime(df['ts'], unit='ns')
                    df['ts'] = df['ts'].dt.tz_localize('UTC').dt.tz_convert(TW_TZ)
                    
                    logger.info(f"Daily DataFrame shape: {df.shape}")
                    logger.info(f"Daily DataFrame columns: {df.columns.tolist()}")
                    logger.info(f"Date range: {df['ts'].min()} to {df['ts'].max()}")
                    
                    if df.empty or len(df) < 5:
                        return {
                            "success": False, 
                            "message": f"Insufficient historical data for {stock_code}. Got {len(df)} days, need at least 5."
                        }
                    
                    # Extract price and volume data from DataFrame
                    close_prices = [float(price) for price in df['Close'].tolist()]
                    high_prices = [float(price) for price in df['High'].tolist()]
                    low_prices = [float(price) for price in df['Low'].tolist()]
                    volumes = [int(volume) for volume in df['Volume'].tolist()]
                    
                    logger.info(f"Extracted {len(close_prices)} days of data")
                    logger.info(f"Close prices - First: {close_prices[0]:.2f}, Last: {close_prices[-1]:.2f}")
                    
                    # 計算技術指標
                    def calculate_ma(prices, period):
                        if len(prices) < period:
                            return None
                        return sum(prices[-period:]) / period
                    
                    def calculate_bollinger_bands(prices, period=20, std_dev=2):
                        if len(prices) < period:
                            return None, None, None
                        
                        recent_prices = prices[-period:]
                        ma = sum(recent_prices) / period
                        variance = sum((price - ma) ** 2 for price in recent_prices) / period
                        std = variance ** 0.5
                        
                        upper_band = ma + (std_dev * std)
                        lower_band = ma - (std_dev * std)
                        
                        return ma, upper_band, lower_band
                    
                    # 計算技術指標
                    ma_20 = calculate_ma(close_prices, min(20, len(close_prices)))
                    ma_40 = calculate_ma(close_prices, min(40, len(close_prices))) if len(close_prices) >= 40 else None
                    bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(close_prices, min(20, len(close_prices)), 2)
                    
                    # 成交量分析
                    current_volume = int(volumes[-1]) if volumes else 0
                    avg_volume_5d = calculate_ma(volumes, min(5, len(volumes))) if len(volumes) >= 5 else None
                    avg_volume_20d = calculate_ma(volumes, min(20, len(volumes))) if len(volumes) >= 20 else None
                    
                    # Get current price
                    current_price = float(close_prices[-1]) if close_prices else None
                    
                    # Calculate trend signals
                    trend_signal = None
                    bb_signal = None
                    
                    if ma_20:
                        if ma_40:
                            if ma_20 > ma_40:
                                trend_signal = "bullish" if current_price > ma_20 else "neutral"
                            else:
                                trend_signal = "bearish" if current_price < ma_20 else "neutral"
                        else:
                            trend_signal = "bullish" if current_price > ma_20 else "bearish"
                    
                    if bb_upper and bb_lower and current_price:
                        if current_price > bb_upper:
                            bb_signal = "overbought"
                        elif current_price < bb_lower:
                            bb_signal = "oversold"
                        else:
                            bb_signal = "normal"
                    
                    # 嘗試獲取當日買賣量資料
                    daily_buy_volume = None
                    daily_sell_volume = None
                    
                    try:
                        if target_date:
                            tick_date = target_date
                        else:
                            tick_date = now_tw.date()
                        
                        ticks = api.ticks(
                            contract=contract,
                            date=tick_date.strftime('%Y-%m-%d'),
                            timeout=30000
                        )
                        
                        if ticks and hasattr(ticks, 'ts') and ticks.ts:
                            tick_data = {
                                'ts': ticks.ts, 
                                'close': ticks.close, 
                                'volume': ticks.volume
                            }
                            
                            if hasattr(ticks, 'tick_type') and ticks.tick_type:
                                tick_data['tick_type'] = ticks.tick_type
                            
                            tick_df = pd.DataFrame(tick_data)
                            # 🔴 修正：正確轉換 tick 時間戳記
                            tick_df['ts'] = pd.to_datetime(tick_df['ts'], unit='ns')
                            tick_df['ts'] = tick_df['ts'].dt.tz_localize('UTC').dt.tz_convert(TW_TZ)
                            
                            logger.info(f"Daily tick data shape: {tick_df.shape}")
                            
                            if 'tick_type' in tick_df.columns:
                                # Calculate daily buy/sell volume using tick_type
                                buy_ticks = tick_df[tick_df['tick_type'] == 1]  # 外盤
                                sell_ticks = tick_df[tick_df['tick_type'] == 2]  # 內盤
                                neutral_ticks = tick_df[tick_df['tick_type'] == 0]  # 無法判定
                                
                                daily_buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                daily_sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                neutral_volume = int(neutral_ticks['volume'].sum()) if not neutral_ticks.empty else 0
                                
                                # Distribute neutral volume 50/50
                                if neutral_volume > 0:
                                    daily_buy_volume += neutral_volume // 2
                                    daily_sell_volume += neutral_volume - (neutral_volume // 2)
                                    
                                logger.info(f"Daily buy/sell volume: buy={daily_buy_volume}, sell={daily_sell_volume}")
                        
                    except Exception as tick_error:
                        logger.error(f"Daily tick data error for {stock_code}: {tick_error}")
                        daily_buy_volume = None
                        daily_sell_volume = None
                    
                    # 返回完整的技術分析資料
                    return {
                        "success": True,
                        "query_date": base_date.strftime('%Y-%m-%d'),
                        "data_range": f"{df['ts'].min().strftime('%Y-%m-%d')} to {df['ts'].max().strftime('%Y-%m-%d')}",
                        "current_price": current_price,
                        "technical_indicators": {
                            "ma_20": round(ma_20, 2) if ma_20 else None,
                            "ma_40": round(ma_40, 2) if ma_40 else None,
                            "ma_40_available": ma_40 is not None,
                            "bollinger_bands": {
                                "middle": round(bb_ma, 2) if bb_ma else None,
                                "upper": round(bb_upper, 2) if bb_upper else None,
                                "lower": round(bb_lower, 2) if bb_lower else None,
                                "signal": bb_signal
                            }
                        },
                        "volume_analysis": {
                            "current_volume": current_volume,
                            "avg_volume_5d": round(avg_volume_5d, 0) if avg_volume_5d else None,
                            "avg_volume_20d": round(avg_volume_20d, 0) if avg_volume_20d else None,
                            "volume_ratio_5d": round(current_volume / avg_volume_5d, 2) if avg_volume_5d and avg_volume_5d > 0 else None,
                            "daily_buy_volume": daily_buy_volume,
                            "daily_sell_volume": daily_sell_volume,
                            "buy_sell_ratio": round(daily_buy_volume / daily_sell_volume, 2) if daily_buy_volume and daily_sell_volume and daily_sell_volume > 0 else None,
                            "net_flow": daily_buy_volume - daily_sell_volume if daily_buy_volume is not None and daily_sell_volume is not None else None
                        },
                        "signals": {
                            "trend": trend_signal,
                            "bollinger": bb_signal
                        },
                        "data_points": len(close_prices),
                        "last_update": df.iloc[-1]['ts'].strftime('%Y-%m-%d %H:%M:%S') if not df.empty else None,
                        "timezone": "Asia/Taipei (+8)"
                    }
                    
                except Exception as df_error:
                    logger.error(f"Error processing daily data for {stock_code}: {df_error}")
                    return {"success": False, "message": f"Daily data processing error: {str(df_error)}"}
        
            except Exception as e:
                logger.error(f"Error getting daily data for {stock_code}: {e}")
                return {"success": False, "message": f"Daily data error: {str(e)}"}
        
    except Exception as e:
        logger.error(f"Overall error for {stock_code}: {e}")
        return {"success": False, "message": f"Technical analysis error: {str(e)}"}
    
    finally:
        logger.info(f"=== RAW DATA LOGGING END for {stock_code} ===")

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
