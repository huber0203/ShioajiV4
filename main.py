from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import shioaji as sj
import os
from typing import Optional, Dict, Any
import logging

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
    global api
    api = sj.Shioaji()
    logger.info("Shioaji API initialized")
    
    # Auto-login using environment variables
    api_key = os.getenv("SHIOAJI_API_KEY")
    secret_key = os.getenv("SHIOAJI_SECRET_KEY")
    person_id = os.getenv("SHIOAJI_PERSON_ID")
    
    if api_key and secret_key:
        try:
            logger.info("Attempting auto-login with environment variables...")
            result = api.login(
                api_key=api_key,
                secret_key=secret_key,
                person_id=person_id
            )
            
            if result:
                accounts = api.list_accounts()
                logger.info(f"Auto-login successful! Connected accounts: {[acc.account_id for acc in accounts]}")
            else:
                logger.error("Auto-login failed: Invalid credentials")
                
        except Exception as e:
            logger.error(f"Auto-login error: {e}")
    else:
        logger.warning("Auto-login skipped: Missing SHIOAJI_API_KEY or SHIOAJI_SECRET_KEY environment variables")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    global api
    if api and api.login:
        try:
            api.logout()
            logger.info("Logged out from Shioaji API")
        except Exception as e:
            logger.error(f"Error during logout: {e}")

@app.get("/")
async def root():
    return {
        "message": "Shioaji Trading API with Auto-Login",
        "version": "1.0.0",
        "status": "running",
        "connected": api.login if api else False,
        "auto_login": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY"))
    }

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Manual login to Shioaji API (optional if auto-login is configured)"""
    global api
    try:
        # Login to Shioaji
        result = api.login(
            api_key=request.api_key,
            secret_key=request.secret_key,
            person_id=request.person_id
        )
        
        if result:
            # Get account info
            accounts = api.list_accounts()
            account_info = {
                "accounts": [acc.account_id for acc in accounts],
                "login_time": str(result)
            }
            
            return LoginResponse(
                success=True,
                message="Manual login successful",
                account_info=account_info
            )
        else:
            return LoginResponse(
                success=False,
                message="Login failed"
            )
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/logout")
async def logout():
    """Logout from Shioaji API"""
    global api
    try:
        if api and api.login:
            api.logout()
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
        if not api or not api.login:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
        accounts = api.list_accounts()
        return {
            "success": True,
            "accounts": [
                {
                    "account_id": acc.account_id,
                    "broker_id": acc.broker_id,
                    "account_type": acc.account_type,
                    "signed": acc.signed
                }
                for acc in accounts
            ]
        }
    except Exception as e:
        logger.error(f"Get accounts error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/positions")
async def get_positions():
    """Get current positions"""
    global api
    try:
        if not api or not api.login:
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
        if not api or not api.login:
            raise HTTPException(status_code=401, detail="Not logged in - please check environment variables")
        
        # Get contract
        contract = api.Contracts.Stocks[request.code]
        if not contract:
            raise HTTPException(status_code=404, detail=f"Stock {request.code} not found")
        
        # Create order
        order = api.Order(
            action=request.action,
            price=request.price or 0,  # 0 for market order
            quantity=request.quantity,
            price_type=sj.constant.StockPriceType.LMT if request.price else sj.constant.StockPriceType.MKT,
            order_type=getattr(sj.constant.OrderType, request.order_type),
            account=api.stock_account
        )
        
        # Place order
        trade = api.place_order(contract, order)
        
        return OrderResponse(
            success=True,
            message="Order placed successfully",
            order_id=trade.order.id if trade else None
        )
        
    except Exception as e:
        logger.error(f"Place order error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/quote/{stock_code}")
async def get_quote(stock_code: str):
    """Get real-time quote for a stock"""
    global api
    try:
        if not api or not api.login:
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
        "api_connected": api.login if api else False,
        "auto_login_configured": bool(os.getenv("SHIOAJI_API_KEY") and os.getenv("SHIOAJI_SECRET_KEY")),
        "timestamp": str(sj.datetime.now())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
