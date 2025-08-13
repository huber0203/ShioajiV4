from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

security = HTTPBearer()

# Global Shioaji API instance
api = None
login_status = False

# Pydantic models
class LoginRequest(BaseModel):
    api_key: str
    secret_key: str
    person_id: Optional[str] = None

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
        
        if api_key and secret_key:
            try:
                logger.info("Attempting auto-login with environment variables...")
                accounts = api.login(
                    api_key=api_key,
                    secret_key=secret_key,
                    fetch_contract=True,
                    subscribe_trade=True
                )
                
                if accounts:
                    login_status = True
                    logger.info(f"Auto-login successful! Connected accounts: {[acc.account_id for acc in accounts]}")
                    logger.info(f"Stock account: {api.stock_account}")
                    logger.info(f"Future account: {api.futopt_account}")
                else:
                    login_status = False
                    logger.error("Auto-login failed: No accounts returned")
                    
            except Exception as e:
                login_status = False
                logger.error(f"Auto-login error: {e}")
                # Don't raise the exception - let the service start anyway
        else:
            login_status = False
            logger.warning("Auto-login skipped: Missing SHIOAJI_API_KEY or SHIOAJI_SECRET_KEY environment variables")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Initialize api as None if there's an error, but don't crash the service
        api = None
        login_status = False

# Shutdown event
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
        "auto_login": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY"))
    }

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Manual login to Shioaji API (optional if auto-login is configured)"""
    global api, login_status
    try:
        accounts = api.login(
            api_key=request.api_key,
            secret_key=request.secret_key,
            fetch_contract=True,
            subscribe_trade=True
        )
        
        if accounts:
            login_status = True
            # Get account info
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
        raise HTTPException(status_code=400, detail=str(e))

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
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/accounts")
async def get_accounts():
    """Get account information"""
    global api
    try:
        if not api or not login_status:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
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
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    global api
    try:
        if not api or not login_status:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
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
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/order", response_model=OrderResponse)
async def place_order(request: OrderRequest):
    """Place a stock order"""
    global api
    try:
        if not api or not login_status:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
        # Get contract
        contract = api.Contracts.Stocks.get(request.code)
        if not contract:
            raise HTTPException(status_code=404, detail=f"Stock {request.code} not found")
        
        if not api.stock_account:
            raise HTTPException(status_code=400, detail="No stock account available")
        
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
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/quote/{stock_code}")
async def get_quote(stock_code: str):
    """Get real-time quote for a stock"""
    global api
    try:
        if not api or not login_status:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
        # Get contract
        contract = api.Contracts.Stocks.get(stock_code)
        if not contract:
            raise HTTPException(status_code=404, detail=f"Stock {stock_code} not found")
        
        # Get quote
        quote = api.quote.get_quote(contract)
        
        return {
            "success": True,
            "code": stock_code,
            "name": contract.name,
            "price": quote.get('Close', 0),
            "change": quote.get('Change', 0),
            "change_percent": quote.get('ChangePercent', 0),
            "volume": quote.get('Volume', 0),
            "high": quote.get('High', 0),
            "low": quote.get('Low', 0),
            "open": quote.get('Open', 0)
        }
        
    except Exception as e:
        logger.error(f"Get quote error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "api_connected": login_status,
        "auto_login_configured": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
        "timestamp": str(datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
