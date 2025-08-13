# Shioaji Trading API

基於官方 Shioaji SDK 的 FastAPI 交易服務，專為 n8n 自動化工作流程設計。

## 功能特色

- ✅ **官方 SDK** - 使用永豐金證券官方 Shioaji Python SDK
- ✅ **REST API** - 標準 HTTP API，完美支援 n8n 整合
- ✅ **完整功能** - 登入、下單、查詢持倉、即時報價
- ✅ **Docker 部署** - 支援 Zeabur、Vercel 等雲端平台
- ✅ **錯誤處理** - 完整的錯誤處理和日誌記錄

## 快速開始

### 1. 環境設定

複製環境變數範例檔案：
\`\`\`bash
cp .env.example .env
\`\`\`

編輯 `.env` 檔案，填入您的永豐金 API 金鑰：
\`\`\`env
SHIOAJI_API_KEY=your_api_key_here
SHIOAJI_SECRET_KEY=your_secret_key_here
SHIOAJI_PERSON_ID=your_person_id_here
\`\`\`

### 2. 本地開發

安裝依賴：
\`\`\`bash
pip install -r requirements.txt
\`\`\`

啟動服務：
\`\`\`bash
python main.py
\`\`\`

API 文檔：http://localhost:8000/docs

### 3. Docker 部署

建置映像：
\`\`\`bash
docker build -t shioaji-api .
\`\`\`

執行容器：
\`\`\`bash
docker run -p 8000:8000 --env-file .env shioaji-api
\`\`\`

### 4. Zeabur 部署

1. 將專案推送到 GitHub
2. 在 Zeabur 中連接 GitHub 儲存庫
3. 設定環境變數（API 金鑰）
4. 一鍵部署完成

## API 端點

### 基本操作
- `GET /` - 服務狀態
- `GET /health` - 健康檢查
- `POST /login` - 登入 Shioaji API
- `POST /logout` - 登出

### 帳戶管理
- `GET /accounts` - 取得帳戶資訊
- `GET /positions` - 取得持倉資訊

### 交易功能
- `POST /order` - 下單
- `GET /quote/{stock_code}` - 取得即時報價

## n8n 整合範例

### 登入
\`\`\`json
{
  "method": "POST",
  "url": "https://your-api.zeabur.app/login",
  "body": {
    "api_key": "your_api_key",
    "secret_key": "your_secret_key",
    "person_id": "your_person_id"
  }
}
\`\`\`

### 下單
\`\`\`json
{
  "method": "POST",
  "url": "https://your-api.zeabur.app/order",
  "body": {
    "action": "Buy",
    "code": "2330",
    "quantity": 1000,
    "price": 500.0,
    "order_type": "ROD"
  }
}
\`\`\`

### 查詢報價
\`\`\`json
{
  "method": "GET",
  "url": "https://your-api.zeabur.app/quote/2330"
}
\`\`\`

## 注意事項

1. **API 金鑰安全** - 請妥善保管您的 API 金鑰，不要提交到版本控制
2. **交易風險** - 請謹慎使用自動化交易功能，建議先在測試環境驗證
3. **連線穩定** - 確保網路連線穩定，避免交易中斷
4. **法規遵循** - 請遵守相關金融法規和交易規則

## 技術支援

- [Shioaji 官方文檔](https://sinotrade.github.io/shioaji/)
- [FastAPI 文檔](https://fastapi.tiangolo.com/)
- [n8n 文檔](https://docs.n8n.io/)

## 授權條款

MIT License
