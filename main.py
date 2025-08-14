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

@app.get("/realtime/{stock_code}")
async def get_realtime_quote(stock_code: str):
    """Get comprehensive real-time quote data with full parameter logging
    
    Returns all available Shioaji quote parameters with Chinese descriptions
    """
    global api
    try:
        if not ensure_login():
            return {"success": False, "message": "Unable to connect - please check environment variables"}
        
        # Get contract
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            return {"success": False, "message": f"Stock {stock_code} not found"}
        
        logger.info(f"=== 即時行情查詢開始 ===")
        logger.info(f"股票代碼: {stock_code}")
        logger.info(f"股票名稱: {contract.name}")
        
        # Try to get comprehensive quote data
        quote_data = None
        data_source = "unknown"
        
        try:
            # Method 1: Try snapshots (most comprehensive)
            snapshots = api.snapshots([contract])
            if snapshots and len(snapshots) > 0:
                quote_data = snapshots[0]
                data_source = "snapshots"
                logger.info("數據來源: snapshots API")
        except Exception as e:
            logger.warning(f"Snapshots failed: {e}")
        
        if not quote_data:
            try:
                # Method 2: Try quote subscription (if available)
                # Note: This would require setting up callbacks, so we'll skip for now
                pass
            except Exception as e:
                logger.warning(f"Quote subscription failed: {e}")
        
        if not quote_data:
            return {
                "success": False, 
                "message": f"Unable to get real-time data for {stock_code}",
                "contract_info": {
                    "code": stock_code,
                    "name": contract.name,
                    "exchange": str(contract.exchange) if hasattr(contract, 'exchange') else "unknown"
                }
            }
        
        # Log all raw attributes
        logger.info("=== 原始數據結構 ===")
        logger.info(f"數據類型: {type(quote_data)}")
        logger.info(f"所有屬性: {dir(quote_data)}")
        
        # Extract and log all available parameters with Chinese descriptions
        quote_params = {}
        
        # 基本資訊 (Basic Information)
        if hasattr(quote_data, 'code'):
            quote_params['code'] = str(quote_data.code)
            logger.info(f"商品代碼 (code): {quote_data.code}")
        
        if hasattr(quote_data, 'datetime'):
            quote_params['datetime'] = quote_data.datetime.strftime('%Y-%m-%d %H:%M:%S') if quote_data.datetime else None
            logger.info(f"時間 (datetime): {quote_data.datetime}")
        
        # 價格資訊 (Price Information)
        if hasattr(quote_data, 'open'):
            quote_params['open'] = float(quote_data.open) if quote_data.open else 0
            logger.info(f"開盤價 (open): {quote_data.open}")
        
        if hasattr(quote_data, 'close'):
            quote_params['close'] = float(quote_data.close) if quote_data.close else 0
            logger.info(f"成交價 (close): {quote_data.close}")
        
        if hasattr(quote_data, 'high'):
            quote_params['high'] = float(quote_data.high) if quote_data.high else 0
            logger.info(f"最高價 (high): {quote_data.high}")
        
        if hasattr(quote_data, 'low'):
            quote_params['low'] = float(quote_data.low) if quote_data.low else 0
            logger.info(f"最低價 (low): {quote_data.low}")
        
        if hasattr(quote_data, 'avg_price'):
            quote_params['avg_price'] = float(quote_data.avg_price) if quote_data.avg_price else 0
            logger.info(f"均價 (avg_price): {quote_data.avg_price}")
        
        # 成交量資訊 (Volume Information)
        if hasattr(quote_data, 'volume'):
            quote_params['volume'] = int(quote_data.volume) if quote_data.volume else 0
            logger.info(f"成交量 (volume): {quote_data.volume}")
        
        if hasattr(quote_data, 'total_volume'):
            quote_params['total_volume'] = int(quote_data.total_volume) if quote_data.total_volume else 0
            logger.info(f"總成交量 (total_volume): {quote_data.total_volume}")
        
        # 成交額資訊 (Amount Information)
        if hasattr(quote_data, 'amount'):
            quote_params['amount'] = float(quote_data.amount) if quote_data.amount else 0
            logger.info(f"成交額 (amount): {quote_data.amount}")
        
        if hasattr(quote_data, 'total_amount'):
            quote_params['total_amount'] = float(quote_data.total_amount) if quote_data.total_amount else 0
            logger.info(f"總成交額 (total_amount): {quote_data.total_amount}")
        
        # 內外盤資訊 (Buy/Sell Side Information)
        if hasattr(quote_data, 'tick_type'):
            quote_params['tick_type'] = int(quote_data.tick_type) if quote_data.tick_type is not None else 0
            tick_type_desc = {1: "外盤", 2: "內盤", 0: "無法判定"}.get(quote_params['tick_type'], "未知")
            logger.info(f"內外盤別 (tick_type): {quote_data.tick_type} ({tick_type_desc})")
        
        if hasattr(quote_data, 'bid_side_total_vol'):
            quote_params['bid_side_total_vol'] = int(quote_data.bid_side_total_vol) if quote_data.bid_side_total_vol else 0
            logger.info(f"買盤成交總量 (bid_side_total_vol): {quote_data.bid_side_total_vol}")
        
        if hasattr(quote_data, 'ask_side_total_vol'):
            quote_params['ask_side_total_vol'] = int(quote_data.ask_side_total_vol) if quote_data.ask_side_total_vol else 0
            logger.info(f"賣盤成交總量 (ask_side_total_vol): {quote_data.ask_side_total_vol}")
        
        if hasattr(quote_data, 'bid_side_total_cnt'):
            quote_params['bid_side_total_cnt'] = int(quote_data.bid_side_total_cnt) if quote_data.bid_side_total_cnt else 0
            logger.info(f"買盤成交筆數 (bid_side_total_cnt): {quote_data.bid_side_total_cnt}")
        
        if hasattr(quote_data, 'ask_side_total_cnt'):
            quote_params['ask_side_total_cnt'] = int(quote_data.ask_side_total_cnt) if quote_data.ask_side_total_cnt else 0
            logger.info(f"賣盤成交筆數 (ask_side_total_cnt): {quote_data.ask_side_total_cnt}")
        
        # 漲跌資訊 (Change Information)
        if hasattr(quote_data, 'chg_type'):
            quote_params['chg_type'] = int(quote_data.chg_type) if quote_data.chg_type is not None else 0
            chg_type_desc = {1: "漲停", 2: "漲", 3: "平盤", 4: "跌", 5: "跌停"}.get(quote_params['chg_type'], "未知")
            logger.info(f"漲跌註記 (chg_type): {quote_data.chg_type} ({chg_type_desc})")
        
        if hasattr(quote_data, 'price_chg'):
            quote_params['price_chg'] = float(quote_data.price_chg) if quote_data.price_chg else 0
            logger.info(f"漲跌價 (price_chg): {quote_data.price_chg}")
        
        if hasattr(quote_data, 'pct_chg'):
            quote_params['pct_chg'] = float(quote_data.pct_chg) if quote_data.pct_chg else 0
            logger.info(f"漲跌率 (pct_chg): {quote_data.pct_chg}%")
        
        # 委買委賣資訊 (Bid/Ask Information)
        if hasattr(quote_data, 'bid_price'):
            quote_params['bid_price'] = [float(p) for p in quote_data.bid_price] if quote_data.bid_price else []
            logger.info(f"買價 (bid_price): {quote_data.bid_price}")
        
        if hasattr(quote_data, 'bid_volume'):
            quote_params['bid_volume'] = [int(v) for v in quote_data.bid_volume] if quote_data.bid_volume else []
            logger.info(f"買量 (bid_volume): {quote_data.bid_volume}")
        
        if hasattr(quote_data, 'ask_price'):
            quote_params['ask_price'] = [float(p) for p in quote_data.ask_price] if quote_data.ask_price else []
            logger.info(f"賣價 (ask_price): {quote_data.ask_price}")
        
        if hasattr(quote_data, 'ask_volume'):
            quote_params['ask_volume'] = [int(v) for v in quote_data.ask_volume] if quote_data.ask_volume else []
            logger.info(f"賣量 (ask_volume): {quote_data.ask_volume}")
        
        if hasattr(quote_data, 'diff_bid_vol'):
            quote_params['diff_bid_vol'] = [int(v) for v in quote_data.diff_bid_vol] if quote_data.diff_bid_vol else []
            logger.info(f"買價增減量 (diff_bid_vol): {quote_data.diff_bid_vol}")
        
        if hasattr(quote_data, 'diff_ask_vol'):
            quote_params['diff_ask_vol'] = [int(v) for v in quote_data.diff_ask_vol] if quote_data.diff_ask_vol else []
            logger.info(f"賣價增減量 (diff_ask_vol): {quote_data.diff_ask_vol}")
        
        # 盤後零股資訊 (After-hours Odd Lot Information)
        if hasattr(quote_data, 'closing_oddlot_shares'):
            quote_params['closing_oddlot_shares'] = int(quote_data.closing_oddlot_shares) if quote_data.closing_oddlot_shares else 0
            logger.info(f"盤後零股成交股數 (closing_oddlot_shares): {quote_data.closing_oddlot_shares}")
        
        if hasattr(quote_data, 'closing_oddlot_close'):
            quote_params['closing_oddlot_close'] = float(quote_data.closing_oddlot_close) if quote_data.closing_oddlot_close else 0
            logger.info(f"盤後零股成交價 (closing_oddlot_close): {quote_data.closing_oddlot_close}")
        
        if hasattr(quote_data, 'closing_oddlot_amount'):
            quote_params['closing_oddlot_amount'] = float(quote_data.closing_oddlot_amount) if quote_data.closing_oddlot_amount else 0
            logger.info(f"盤後零股成交額 (closing_oddlot_amount): {quote_data.closing_oddlot_amount}")
        
        if hasattr(quote_data, 'closing_oddlot_bid_price'):
            quote_params['closing_oddlot_bid_price'] = float(quote_data.closing_oddlot_bid_price) if quote_data.closing_oddlot_bid_price else 0
            logger.info(f"盤後零股買價 (closing_oddlot_bid_price): {quote_data.closing_oddlot_bid_price}")
        
        if hasattr(quote_data, 'closing_oddlot_ask_price'):
            quote_params['closing_oddlot_ask_price'] = float(quote_data.closing_oddlot_ask_price) if quote_data.closing_oddlot_ask_price else 0
            logger.info(f"盤後零股賣價 (closing_oddlot_ask_price): {quote_data.closing_oddlot_ask_price}")
        
        # 定盤交易資訊 (Fixed Price Trading Information)
        if hasattr(quote_data, 'fixed_trade_vol'):
            quote_params['fixed_trade_vol'] = int(quote_data.fixed_trade_vol) if quote_data.fixed_trade_vol else 0
            logger.info(f"定盤成交量 (fixed_trade_vol): {quote_data.fixed_trade_vol}")
        
        if hasattr(quote_data, 'fixed_trade_amount'):
            quote_params['fixed_trade_amount'] = float(quote_data.fixed_trade_amount) if quote_data.fixed_trade_amount else 0
            logger.info(f"定盤成交額 (fixed_trade_amount): {quote_data.fixed_trade_amount}")
        
        # 其他資訊 (Other Information)
        if hasattr(quote_data, 'avail_borrowing'):
            quote_params['avail_borrowing'] = int(quote_data.avail_borrowing) if quote_data.avail_borrowing else 0
            logger.info(f"借券可用餘額 (avail_borrowing): {quote_data.avail_borrowing}")
        
        if hasattr(quote_data, 'suspend'):
            quote_params['suspend'] = bool(quote_data.suspend) if quote_data.suspend is not None else False
            logger.info(f"暫停交易 (suspend): {quote_data.suspend}")
        
        if hasattr(quote_data, 'simtrade'):
            quote_params['simtrade'] = bool(quote_data.simtrade) if quote_data.simtrade is not None else False
            logger.info(f"試撮 (simtrade): {quote_data.simtrade}")
        
        if hasattr(quote_data, 'intraday_odd'):
            quote_params['intraday_odd'] = int(quote_data.intraday_odd) if quote_data.intraday_odd is not None else 0
            intraday_odd_desc = {0: "整股", 1: "盤中零股"}.get(quote_params['intraday_odd'], "未知")
            logger.info(f"盤中零股 (intraday_odd): {quote_data.intraday_odd} ({intraday_odd_desc})")
        
        logger.info("=== 即時行情查詢完成 ===")
        
        # Return comprehensive response
        return {
            "success": True,
            "data_source": data_source,
            "stock_info": {
                "code": stock_code,
                "name": contract.name,
                "exchange": str(contract.exchange) if hasattr(contract, 'exchange') else "unknown"
            },
            "quote_data": quote_params,
            "parameter_descriptions": {
                "code": "商品代碼",
                "datetime": "時間",
                "open": "開盤價",
                "close": "成交價",
                "high": "最高價(自開盤)",
                "low": "最低價(自開盤)",
                "avg_price": "均價",
                "volume": "成交量",
                "total_volume": "總成交量",
                "amount": "成交額 (NTD)",
                "total_amount": "總成交額 (NTD)",
                "tick_type": "內外盤別 {1: 外盤, 2: 內盤, 0: 無法判定}",
                "chg_type": "漲跌註記 {1: 漲停, 2: 漲, 3: 平盤, 4: 跌, 5: 跌停}",
                "price_chg": "漲跌價",
                "pct_chg": "漲跌率",
                "bid_side_total_vol": "買盤成交總量 (張)",
                "ask_side_total_vol": "賣盤成交總量 (張)",
                "bid_side_total_cnt": "買盤成交筆數",
                "ask_side_total_cnt": "賣盤成交筆數",
                "bid_price": "買價",
                "bid_volume": "買量",
                "diff_bid_vol": "買價增減量",
                "ask_price": "賣價",
                "ask_volume": "賣量",
                "diff_ask_vol": "賣價增減量",
                "closing_oddlot_shares": "盤後零股成交股數",
                "closing_oddlot_close": "盤後零股成交價",
                "closing_oddlot_amount": "盤後零股成交額",
                "closing_oddlot_bid_price": "盤後零股買價",
                "closing_oddlot_ask_price": "盤後零股賣價",
                "fixed_trade_vol": "定盤成交量 (張)",
                "fixed_trade_amount": "定盤成交額",
                "avail_borrowing": "借券可用餘額",
                "suspend": "暫停交易",
                "simtrade": "試撮",
                "intraday_odd": "盤中零股 {0: 整股, 1: 盤中零股}"
            },
            "timestamp": datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "timezone": "Asia/Taipei (+8)"
        }
        
    except Exception as e:
        logger.error(f"Get realtime quote error: {e}")
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
    import statistics
    
    logger.info(f"=== RAW DATA LOGGING START for {stock_code} ===")
    logger.info(f"Contract info: {contract}")
    logger.info(f"Timeframe: {timeframe}, Session: {session}, Target date: {target_date}")
    
    # Use target_date if provided, otherwise use current date
    if target_date:
        base_date = target_date
        now_tw = TW_TZ.localize(datetime.combine(target_date, datetime.min.time().replace(hour=12, minute=0)))
    else:
        now_tw = datetime.now(TW_TZ)
        base_date = now_tw.date()
    
    if timeframe == "5min":
        # Get intraday 1-minute data - use broader time range to capture available data
        
        try:
            # First, get all available recent data
            kbars = api.kbars(
                contract=contract,
                timeout=30000
            )
            
            logger.info(f"Raw kbars object type: {type(kbars)}")
            logger.info(f"Raw kbars attributes: {dir(kbars) if kbars else 'None'}")
            
            if kbars:
                if hasattr(kbars, 'ts'):
                    logger.info(f"Raw kbars.ts type: {type(kbars.ts)}")
                    logger.info(f"Raw kbars.ts length: {len(kbars.ts) if kbars.ts else 0}")
                    logger.info(f"Raw kbars.ts first 5 values: {kbars.ts[:5] if kbars.ts else 'None'}")
                
                if hasattr(kbars, 'Open'):
                    logger.info(f"Raw kbars.Open type: {type(kbars.Open)}")
                    logger.info(f"Raw kbars.Open length: {len(kbars.Open) if kbars.Open else 0}")
                    logger.info(f"Raw kbars.Open first 5 values: {kbars.Open[:5] if kbars.Open else 'None'}")
                
                if hasattr(kbars, 'High'):
                    logger.info(f"Raw kbars.High first 5 values: {kbars.High[:5] if kbars.High else 'None'}")
                
                if hasattr(kbars, 'Low'):
                    logger.info(f"Raw kbars.Low first 5 values: {kbars.Low[:5] if kbars.Low else 'None'}")
                
                if hasattr(kbars, 'Close'):
                    logger.info(f"Raw kbars.Close first 5 values: {kbars.Close[:5] if kbars.Close else 'None'}")
                
                if hasattr(kbars, 'Volume'):
                    logger.info(f"Raw kbars.Volume type: {type(kbars.Volume)}")
                    logger.info(f"Raw kbars.Volume first 5 values: {kbars.Volume[:5] if kbars.Volume else 'None'}")
            
            if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
                logger.info(f"No kbars data available - kbars: {kbars}, has ts: {hasattr(kbars, 'ts') if kbars else False}")
                return {"success": False, "message": f"No intraday data available for {stock_code}"}
            
            try:
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
                logger.info(f"DataFrame columns: {df.columns.tolist()}")
                logger.info(f"DataFrame dtypes: {df.dtypes.to_dict()}")
                logger.info(f"DataFrame first 3 rows:\n{df.head(3)}")
                
                # Convert timestamps to Taiwan timezone
                df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_convert(TW_TZ)
                
                logger.info(f"After timezone conversion - first 3 timestamps: {df['ts'].head(3).tolist()}")
                logger.info(f"After timezone conversion - last 3 timestamps: {df['ts'].tail(3).tolist()}")
                
                if session == "morning":
                    # Morning session (早盤): 09:00 - 13:30 Taiwan time
                    session_name = "早盤 (Morning Session)"
                    # Filter for morning hours (9-13)
                    df_session = df[df['ts'].dt.hour.between(9, 13)]
                    # Also include 13:30 data
                    df_1330 = df[(df['ts'].dt.hour == 13) & (df['ts'].dt.minute <= 30)]
                    df_session = pd.concat([df_session, df_1330]).drop_duplicates().sort_values('ts')
                    
                elif session == "night":
                    # Night session (夜盤): 15:00 - 05:00 next day Taiwan time
                    session_name = "夜盤 (Night Session)"
                    # Filter for night hours (15-23 and 0-5)
                    df_night1 = df[df['ts'].dt.hour >= 15]  # 15:00-23:59
                    df_night2 = df[df['ts'].dt.hour <= 5]   # 00:00-05:00
                    df_session = pd.concat([df_night1, df_night2]).drop_duplicates().sort_values('ts')
                
                logger.info(f"After session filtering ({session}) - DataFrame shape: {df_session.shape}")
                
                if df_session.empty:
                    available_start = df['ts'].min().strftime('%H:%M') if not df.empty else "N/A"
                    available_end = df['ts'].max().strftime('%H:%M') if not df.empty else "N/A"
                    return {
                        "success": False, 
                        "message": f"No {session} session data available for {stock_code}. Available data time range: {available_start} - {available_end}",
                        "available_data_range": f"{available_start} - {available_end}",
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
                logger.info(f"5-minute bars first 3 rows:\n{df_5min.head(3)}")
                
                # Reset index to get timestamp back as column
                df_5min.reset_index(inplace=True)
                
                if df_5min.empty:
                    return {"success": False, "message": f"Could not create 5-minute bars for {stock_code} {session} session"}
                
                try:
                    if session == "night" and start_time.date() != end_time.date():
                        # For night session spanning two days, get ticks for both days
                        ticks_day1 = api.ticks(
                            contract=contract,
                            date=start_time.strftime('%Y-%m-%d'),
                            timeout=30000
                        )
                        ticks_day2 = api.ticks(
                            contract=contract,
                            date=end_time.strftime('%Y-%m-%d'),
                            timeout=30000
                        )
                        
                        logger.info(f"Ticks day 1 type: {type(ticks_day1)}")
                        logger.info(f"Ticks day 1 attributes: {dir(ticks_day1) if ticks_day1 else 'None'}")
                        logger.info(f"Ticks day 2 type: {type(ticks_day2)}")
                        logger.info(f"Ticks day 2 attributes: {dir(ticks_day2) if ticks_day2 else 'None'}")
                        
                        # Combine tick data from both days
                        if ticks_day1 and ticks_day2:
                            combined_ticks = {
                                'ts': list(ticks_day1.ts) + list(ticks_day2.ts),
                                'close': list(ticks_day1.close) + list(ticks_day2.close),
                                'volume': list(ticks_day1.volume) + list(ticks_day2.volume)
                            }
                            if hasattr(ticks_day1, 'tick_type') and hasattr(ticks_day2, 'tick_type'):
                                combined_ticks['tick_type'] = list(ticks_day1.tick_type) + list(ticks_day2.tick_type)
                            ticks = type('CombinedTicks', (), combined_ticks)()
                        else:
                            ticks = ticks_day1 or ticks_day2
                    else:
                        # For morning session or single-day night session
                        ticks = api.ticks(
                            contract=contract,
                            date=start_time.strftime('%Y-%m-%d'),
                            timeout=30000
                        )
                    
                    logger.info(f"Raw ticks object type: {type(ticks)}")
                    logger.info(f"Raw ticks attributes: {dir(ticks) if ticks else 'None'}")
                    
                    if ticks:
                        if hasattr(ticks, 'ts'):
                            logger.info(f"Raw ticks.ts type: {type(ticks.ts)}")
                            logger.info(f"Raw ticks.ts length: {len(ticks.ts) if ticks.ts else 0}")
                            logger.info(f"Raw ticks.ts first 3 values: {ticks.ts[:3] if ticks.ts else 'None'}")
                        
                        if hasattr(ticks, 'close'):
                            logger.info(f"Raw ticks.close type: {type(ticks.close)}")
                            logger.info(f"Raw ticks.close first 3 values: {ticks.close[:3] if ticks.close else 'None'}")
                        
                        if hasattr(ticks, 'volume'):
                            logger.info(f"Raw ticks.volume type: {type(ticks.volume)}")
                            logger.info(f"Raw ticks.volume first 3 values: {ticks.volume[:3] if ticks.volume else 'None'}")
                        
                        if hasattr(ticks, 'tick_type'):
                            logger.info(f"Raw ticks.tick_type type: {type(ticks.tick_type)}")
                            logger.info(f"Raw ticks.tick_type first 3 values: {ticks.tick_type[:3] if ticks.tick_type else 'None'}")
                        
                        if hasattr(ticks, 'bid_price'):
                            logger.info(f"Raw ticks.bid_price exists: {hasattr(ticks, 'bid_price')}")
                        
                        if hasattr(ticks, 'ask_price'):
                            logger.info(f"Raw ticks.ask_price exists: {hasattr(ticks, 'ask_price')}")
                    
                    # Process tick data to calculate buy/sell volume for each 5-minute interval
                    tick_df = None
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
                        
                        tick_df = tick_df[(tick_df['ts'] >= start_time) & (tick_df['ts'] <= end_time)]
                        
                        logger.info(f"Tick DataFrame shape: {tick_df.shape}")
                        logger.info(f"Tick DataFrame columns: {tick_df.columns.tolist()}")
                        logger.info(f"Tick DataFrame first 3 rows:\n{tick_df.head(3)}")
                        
                        logger.info(f"Processed {len(tick_df)} ticks for {stock_code} {session} session")
                        
                except Exception as tick_error:
                    logger.error(f"Tick data error for {stock_code} {session} session: {tick_error}")
                    tick_df = None
                
                intraday_data = []
                total_volume_session = 0
                
                for _, row in df_5min.iterrows():
                    # Calculate average price (OHLC average)
                    open_price = float(row['Open'])
                    high_price = float(row['High'])
                    low_price = float(row['Low'])
                    close_price = float(row['Close'])
                    volume = int(row['Volume'])
                    
                    avg_price = (open_price + high_price + low_price + close_price) / 4
                    
                    buy_volume = None
                    sell_volume = None
                    
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
                                buy_ticks = interval_ticks[interval_ticks['tick_type'] == 1]
                                sell_ticks = interval_ticks[interval_ticks['tick_type'] == 2]
                                neutral_ticks = interval_ticks[interval_ticks['tick_type'] == 0]
                                
                                buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                neutral_volume = int(neutral_ticks['volume'].sum()) if not neutral_ticks.empty else 0
                                
                                logger.info(f"5min interval {row['ts'].strftime('%H:%M')}: buy={buy_volume}, sell={sell_volume}, neutral={neutral_volume}, total_ticks={len(interval_ticks)}")
                                
                                # Distribute neutral volume 50/50 instead of proportionally
                                if neutral_volume > 0:
                                    buy_volume += neutral_volume // 2
                                    sell_volume += neutral_volume - (neutral_volume // 2)
                                
                            else:
                                interval_avg = (open_price + high_price + low_price + close_price) / 4
                                buy_ticks = interval_ticks[interval_ticks['close'] >= interval_avg]
                                sell_ticks = interval_ticks[interval_ticks['close'] < interval_avg]
                                
                                buy_volume = int(buy_ticks['volume'].sum()) if not buy_ticks.empty else 0
                                sell_volume = int(sell_ticks['volume'].sum()) if not sell_ticks.empty else 0
                                
                                logger.info(f"5min interval {row['ts'].strftime('%H:%M')} (fallback): buy={buy_volume}, sell={sell_volume}")
                            
                            total_tick_volume = buy_volume + sell_volume
                            if total_tick_volume > volume and volume > 0:
                                # Proportionally adjust if tick volume exceeds K-bar volume
                                ratio = volume / total_tick_volume
                                buy_volume = int(buy_volume * ratio)
                                sell_volume = int(sell_volume * ratio)
                                logger.info(f"Adjusted volumes for {row['ts'].strftime('%H:%M')}: buy={buy_volume}, sell={sell_volume} (ratio={ratio:.2f})")
                            elif total_tick_volume == 0 and volume > 0:
                                # If no tick data, estimate 50/50
                                buy_volume = volume // 2
                                sell_volume = volume - buy_volume
                                logger.info(f"Estimated 50/50 for {row['ts'].strftime('%H:%M')}: buy={buy_volume}, sell={sell_volume}")

                    bar_data = {
                        "time": row['ts'].strftime('%H:%M'),
                        "open": open_price,
                        "high": high_price,
                        "low": low_price,
                        "close": close_price,
                        "average_price": round(avg_price, 2),
                        "volume": volume,
                        "buy_volume": buy_volume,
                        "sell_volume": sell_volume
                    }
                    
                    intraday_data.append(bar_data)
                    total_volume_session += volume

                if not intraday_data:
                    return {"success": False, "message": f"Could not process 5-minute data for {stock_code} {session} session"}
                
                # Sort by time to ensure proper order
                intraday_data.sort(key=lambda x: x["time"])
                
                current_hour = now_tw.hour
                current_minute = now_tw.minute
                
                if session == "morning":
                    market_status = "open" if 9 <= current_hour < 13 or (current_hour == 13 and current_minute <= 30) else "closed"
                else:  # night session
                    market_status = "open" if (current_hour >= 15) or (current_hour < 5) else "closed"
                
                return {
                    "success": True,
                    "session": session_name,
                    "query_date": base_date.strftime('%Y-%m-%d'),
                    "session_time_range": f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}",
                    "current_time": now_tw.strftime('%H:%M'),
                    "current_date": now_tw.strftime('%Y-%m-%d'),
                    "session_total_volume": total_volume_session,
                    "intraday_bars": intraday_data,  # All 5-minute bars for the specified session
                    "total_bars_session": len(intraday_data),
                    "market_status": market_status,
                    "timezone": "Asia/Taipei (+8)",
                    "data_range": f"{intraday_data[0]['time']} - {intraday_data[-1]['time']}" if intraday_data else "No data",
                    "buy_sell_data_available": tick_df is not None,
                    "aggregation_method": "1min -> 5min resampling"
                }
                
            except Exception as df_error:
                logger.error(f"Error processing 5-minute data for {stock_code} {session} session on {base_date}: {df_error}")
                return {"success": False, "message": f"Data processing error: {str(df_error)}"}
            
        except Exception as e:
            logger.error(f"Error getting 5min data for {stock_code} {session} session on {base_date}: {e}")
            return {"success": False, "message": f"5min data error: {str(e)}"}
        
    else:
        # Daily data with specific date support
        if target_date:
            # For specific date, get data around that date
            end_date = TW_TZ.localize(datetime.combine(target_date, datetime.min.time().replace(hour=23, minute=59)))
            start_date = end_date - timedelta(days=50)
        else:
            # For current date, use existing logic
            now_tw = datetime.now(TW_TZ)
            end_date = now_tw
            start_date = end_date - timedelta(days=50)
        
        logger.info(f"Daily data - Start: {start_date}, End: {end_date}")
        
        try:
            kbars = api.kbars(
                contract=contract
            )
            
            logger.info(f"Daily raw kbars object type: {type(kbars)}")
            logger.info(f"Daily raw kbars attributes: {dir(kbars) if kbars else 'None'}")
            
            if kbars:
                if hasattr(kbars, 'ts'):
                    logger.info(f"Daily kbars.ts length: {len(kbars.ts) if kbars.ts else 0}")
                    logger.info(f"Daily kbars.ts first 3: {kbars.ts[:3] if kbars.ts else 'None'}")
                    logger.info(f"Daily kbars.ts last 3: {kbars.ts[-3:] if kbars.ts else 'None'}")
                
                if hasattr(kbars, 'Close'):
                    logger.info(f"Daily kbars.Close first 3: {kbars.Close[:3] if kbars.Close else 'None'}")
                    logger.info(f"Daily kbars.Close last 3: {kbars.Close[-3:] if kbars.Close else 'None'}")
                
                if hasattr(kbars, 'Volume'):
                    logger.info(f"Daily kbars.Volume first 3: {kbars.Volume[:3] if kbars.Volume else 'None'}")
                    logger.info(f"Daily kbars.Volume last 3: {kbars.Volume[-3:] if kbars.Volume else 'None'}")
            
            if not kbars or not hasattr(kbars, 'ts') or not kbars.ts:
                return {"success": False, "message": f"No historical data available for {stock_code} around {base_date}"}
            
            try:
                df = pd.DataFrame({**kbars})
                df.ts = pd.to_datetime(df.ts, utc=True).dt.tz_convert(TW_TZ)
                
                logger.info(f"Daily DataFrame shape: {df.shape}")
                logger.info(f"Daily DataFrame columns: {df.columns.tolist()}")
                logger.info(f"Daily DataFrame first 3 rows:\n{df.head(3)}")
                logger.info(f"Daily DataFrame last 3 rows:\n{df.tail(3)}")
                
                if df.empty or len(df) < 5:
                    return {
                        "success": False, 
                        "message": f"Insufficient historical data for {stock_code} around {base_date}. Got {len(df)} days, need at least 5."
                    }
                
                # Extract price and volume data from DataFrame
                close_prices = df['Close'].tolist()
                volumes = df['Volume'].tolist()
                
                logger.info(f"Extracted close_prices length: {len(close_prices)}")
                logger.info(f"Close prices first 5: {close_prices[:5]}")
                logger.info(f"Close prices last 5: {close_prices[-5:]}")
                logger.info(f"Extracted volumes length: {len(volumes)}")
                logger.info(f"Volumes first 5: {volumes[:5]}")
                logger.info(f"Volumes last 5: {volumes[-5:]}")
                
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
                ma_40 = calculate_ma(close_prices, min(40, len(close_prices))) if len(close_prices) >= 5 else None
                bb_ma, bb_upper, bb_lower = calculate_bollinger_bands(close_prices, min(20, len(close_prices)), 2)
                
                # Volume analysis
                current_volume = volumes[-1] if volumes else 0
                avg_volume_5d = calculate_ma(volumes, min(5, len(volumes))) if len(volumes) >= 3 else None
                avg_volume_20d = calculate_ma(volumes, min(20, len(volumes))) if len(volumes) >= 5 else None
                
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
                
                daily_buy_volume = None
                daily_sell_volume = None
                
                # Try to get tick data for buy/sell volume analysis
                try:
                    if target_date:
                        tick_date = target_date
                    else:
                        tick_date = now_tw.date()
                    
                    ticks = api.ticks(
                        contract=contract,
                        date=tick_date.strftime('%Y-%m-%d')
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
                        tick_df['ts'] = pd.to_datetime(tick_df['ts'], utc=True).dt.tz_convert(TW_TZ)
                        
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
                                
                            logger.info(f"Daily buy/sell volume calculated: buy={daily_buy_volume}, sell={daily_sell_volume}")
                        else:
                            # Fallback: estimate buy/sell based on price movement
                            if len(close_prices) >= 2:
                                price_up_volume = 0
                                price_down_volume = 0
                                
                                for i in range(1, len(close_prices)):
                                    if close_prices[i] > close_prices[i-1]:
                                        price_up_volume += volumes[i]
                                    elif close_prices[i] < close_prices[i-1]:
                                        price_down_volume += volumes[i]
                                
                                daily_buy_volume = price_up_volume
                                daily_sell_volume = price_down_volume
                                
                                logger.info(f"Daily buy/sell volume estimated from price: buy={daily_buy_volume}, sell={daily_sell_volume}")
                        
                except Exception as tick_error:
                    logger.error(f"Daily tick data error for {stock_code}: {tick_error}")
                    daily_buy_volume = None
                    daily_sell_volume = None

                return {
                    "success": True,
                    "query_date": base_date.strftime('%Y-%m-%d'),
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
                        "volume_ratio_5d": round(current_volume / avg_volume_5d, 2) if avg_volume_5d and avg_volume_5d > 0 else None,
                        "daily_buy_volume": daily_buy_volume,
                        "daily_sell_volume": daily_sell_volume,
                        "buy_sell_ratio": round(daily_buy_volume / daily_sell_volume, 2) if daily_buy_volume and daily_sell_volume and daily_sell_volume > 0 else None
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
