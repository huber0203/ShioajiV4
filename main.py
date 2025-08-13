from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import shioaji as sj
import os
from typing import Optional, Dict, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shioaji Trading API",
    description="Official Shioaji Trading API for n8n integration with auto-login",
    version="1.0.0"
)

# Global Shioaji API instance
api = None
login_status = False

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
async def get_technical_indicators(stock_codes: str, timeframe: str = "daily"):
    """Get technical indicators for multiple stocks and timeframes
    
    Args:
        stock_codes: Single stock or comma-separated stocks (e.g., "2330" or "2454,2317")
        timeframe: Single timeframe or comma-separated timeframes (e.g., "daily" or "daily,5min")
    
    Examples:
        /technical/2330 - Single stock, daily data
        /technical/2330,2454,2317 - Multiple stocks, daily data
        /technical/2330?timeframe=5min - Single stock, 5min data
        /technical/2330?timeframe=daily,5min - Single stock, both timeframes
        /technical/2330,2454?timeframe=daily,5min - Multiple stocks, both timeframes
    """
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # Parse stock codes and timeframes
        stock_list = [code.strip() for code in stock_codes.split(',')]
        timeframe_list = [tf.strip() for tf in timeframe.split(',')]
        
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
                    # Get technical data for this stock and timeframe
                    tech_data = await get_single_stock_technical(stock_code, contract, tf)
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
                "total_combinations": len(stock_list) * len(timeframe_list)
            },
            "results": results,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        logger.error(f"Batch technical indicators error: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}

async def get_single_stock_technical(stock_code: str, contract, timeframe: str):
    """Helper function to get technical data for a single stock and timeframe"""
    global api
    
    from datetime import datetime, timedelta
    import statistics
    
    if timeframe == "5min":
        # Get intraday 5-minute data for today
        today = datetime.now()
        start_time = today.replace(hour=9, minute=0, second=0, microsecond=0)
        end_time = today.replace(hour=13, minute=30, second=0, microsecond=0)
        
        try:
            kbars = api.kbars(
                contract=contract,
                start=start_time.strftime('%Y-%m-%d'),
                end=end_time.strftime('%Y-%m-%d'),
                timeout=30000
            )
            
            kbars_list = list(kbars) if kbars else []
            
            if not kbars_list:
                return {"success": False, "message": f"No intraday data available for {stock_code}"}
            
            # Process 5-minute data - handle different data structures
            intraday_data = []
            total_volume_today = 0
            
            for kbar in kbars_list:
                try:
                    if hasattr(kbar, 'ts'):
                        # Object with attributes
                        bar_data = {
                            "time": kbar.ts.strftime('%H:%M'),
                            "open": float(kbar.Open),
                            "high": float(kbar.High),
                            "low": float(kbar.Low),
                            "close": float(kbar.Close),
                            "volume": int(kbar.Volume)
                        }
                    elif isinstance(kbar, (tuple, list)) and len(kbar) >= 6:
                        timestamp = kbar[0] if hasattr(kbar[0], 'strftime') else datetime.fromtimestamp(kbar[0])
                        bar_data = {
                            "time": timestamp.strftime('%H:%M'),
                            "open": float(kbar[1]),
                            "high": float(kbar[2]),
                            "low": float(kbar[3]),
                            "close": float(kbar[4]),
                            "volume": int(kbar[5])
                        }
                    else:
                        logger.warning(f"Unknown kbar structure for {stock_code}: {type(kbar)}")
                        continue
                    
                    intraday_data.append(bar_data)
                    total_volume_today += bar_data["volume"]
                    
                except Exception as bar_error:
                    logger.warning(f"Error processing kbar for {stock_code}: {bar_error}, kbar type: {type(kbar)}")
                    continue
            
            if not intraday_data:
                return {"success": False, "message": f"Could not process intraday data for {stock_code}"}
            
            # Find specific time data
            current_time = datetime.now()
            target_time = None
            if current_time.hour >= 13:
                target_time = "12:15"
            elif current_time.hour >= 12:
                target_time = "11:15"
            elif current_time.hour >= 11:
                target_time = "10:15"
            else:
                target_time = "09:15"
            
            target_bar = None
            for bar in intraday_data:
                if bar["time"] == target_time:
                    target_bar = bar
                    break
            
            return {
                "success": True,
                "current_time": current_time.strftime('%H:%M'),
                "target_time_data": target_bar,
                "today_total_volume": total_volume_today,
                "intraday_bars": intraday_data[-10:],  # Last 10 bars
                "total_bars_today": len(intraday_data)
            }
            
        except Exception as e:
            logger.error(f"Error getting 5min data for {stock_code}: {e}")
            return {"success": False, "message": f"5min data error: {str(e)}"}
        
    else:
        # Daily data with enhanced technical indicators
        end_date = datetime.now()
        start_date = end_date - timedelta(days=50)
        
        try:
            kbars = api.kbars(
                contract=contract,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d')
            )
            
            kbars_list = list(kbars) if kbars else []
            
            if not kbars_list or len(kbars_list) < 10:
                return {
                    "success": False, 
                    "message": f"Insufficient historical data for {stock_code}. Got {len(kbars_list) if kbars_list else 0} days, need at least 10."
                }
            
            # Extract price and volume data
            close_prices = [float(kbar.Close) for kbar in kbars_list]
            volumes = [int(kbar.Volume) for kbar in kbars_list]
            
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
            
            ma_20 = calculate_ma(close_prices, min(20, len(close_prices)))
            ma_40 = calculate_ma(close_prices, min(40, len(close_prices))) if len(close_prices) >= 10 else None
            bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(close_prices, min(20, len(close_prices)), 2)
            
            # Volume analysis
            current_volume = volumes[-1] if volumes else 0
            avg_volume_5d = calculate_ma(volumes, min(5, len(volumes))) if len(volumes) >= 3 else None
            avg_volume_20d = calculate_ma(volumes, min(20, len(volumes))) if len(volumes) >= 10 else None
            
            # Get current price
            current_price = close_prices[-1] if close_prices else None
            
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
                    # Use only MA20 for trend if MA40 is not available
                    trend_signal = "bullish" if current_price > ma_20 else "bearish"
            
            if bb_upper and bb_lower and current_price:
                if current_price > bb_upper:
                    bb_signal = "overbought"
                elif current_price < bb_lower:
                    bb_signal = "oversold"
                else:
                    bb_signal = "normal"
            
            return {
                "success": True,
                "current_price": current_price,
                "moving_averages": {
                    "ma_20": round(ma_20, 2) if ma_20 else None,
                    "ma_40": round(ma_40, 2) if ma_40 else None,
                    "ma_40_available": ma_40 is not None
                },
                "bollinger_bands": {
                    "middle": round(bb_ma, 2) if bb_ma else None,
                    "upper": round(bb_upper, 2) if bb_upper else None,
                    "lower": round(bb_lower, 2) if bb_lower else None,
                    "signal": bb_signal
                },
                "volume_analysis": {
                    "current_volume": current_volume,
                    "avg_volume_5d": round(avg_volume_5d, 0) if avg_volume_5d else None,
                    "avg_volume_20d": round(avg_volume_20d, 0) if avg_volume_20d else None,
                    "volume_ratio_5d": round(current_volume / avg_volume_5d, 2) if avg_volume_5d and avg_volume_5d > 0 else None
                },
                "signals": {
                    "trend": trend_signal,
                    "bollinger": bb_signal
                },
                "data_points": len(close_prices),
                "last_update": kbars_list[-1].ts.strftime('%Y-%m-%d %H:%M:%S') if kbars_list else None
            }
            
        except Exception as e:
            logger.error(f"Error getting daily data for {stock_code}: {e}")
            return {"success": False, "message": f"Daily data error: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    actual_connected = check_connection()
    return {
        "status": "healthy",
        "api_connected": actual_connected,
        "login_status_var": login_status,
        "auto_login_configured": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
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
</merged_code>
