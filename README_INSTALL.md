# LiteLLM Proxy 安裝與部署說明文件

本文件說明如何在這台 Linux 主機上安裝、部署與初始化 LiteLLM Proxy 服務。

---

## 🌐 環境規劃與部署工作流 (Environments & Workflow)

本系統採雙機架構：
* **本機 spark1 (10.115.140.130)**：**實驗與開發驗證環境** (Dev/Experimental)。
  * 所有的配置修改、鏡像更新或環境參數調整，都必須先在 `spark1` 本機進行測試。
* **遠端 spark3 (10.115.140.188)**：**正式服務環境** (Production/Prod)。
  * 只有在 `spark1` 本機驗證無誤、QC Check 通過後，才可以同步並部署到 `spark3`。

### 🔄 從 spark1 同步並部署至 spark3 的指令
當本機 `spark1` 的修改驗證通過後，請於 `spark1` 執行以下指令同步至 `spark3`：
```bash
# 進入本機部署目錄
cd /opt/litellm

# 同步部署檔案與 Git 歷史到 spark3
rsync -avz /opt/litellm/ spark3:/opt/litellm/

# 登入 spark3 執行服務重新加載
ssh spark3 "cd /opt/litellm && ./restart.sh"
```

---

## 📋 系統前提需求
* **作業系統**：Linux
* **容器引擎**：Docker Engine (已驗證且已安裝)
* **編排工具**：Docker Compose plugin (已驗證且已安裝)
* **網頁工具**：`curl` 與 `jq` (用於 QC 驗證)

---

## 🛠️ 安裝與部署步驟

目前本主機已完整部署於 `/opt/litellm/` 目錄下。若未來需於其他機器重新部署，請遵循以下步驟：

### 1. 建立部署目錄與權限設定
```bash
sudo mkdir -p /opt/litellm
sudo chown -R "$USER:$USER" /opt/litellm
cd /opt/litellm
```

### 2. 配置環境變數檔 (.env)
建立 `.env` 檔案以儲存主密鑰（Master Key），確保其檔案權限為 600：
```bash
cat > /opt/litellm/.env <<'INNER_EOF'
LITELLM_MASTER_KEY=wpd-local-llm
INNER_EOF
chmod 600 /opt/litellm/.env
```

### 3. 設定服務配置文件 (config.yaml)
根據 upstream API 端點配置模型路由。詳細參數設定說明請參閱 `README_USAGE.md`。

### 4. 建立 Docker Compose 編排檔 (docker-compose.yml)
使用 `berriai/litellm:main-stable` 官方穩定鏡像。

### 5. 首次拉取與啟動
直接使用工作目錄下的管理腳本：
```bash
# 拉取最新穩定鏡像
docker compose pull

# 執行啟動腳本
./start.sh
```

---

## 🔄 系統服務開機自啟動驗證

本服務已完美整合開機自啟動：
1. **Systemd 層面**：Docker 系統服務已啟用開機自啟動（已執行 `systemctl enable docker`）。
2. **Container 層面**：`docker-compose.yml` 中配置了 `restart: unless-stopped` 策略。
只要主機開機，LiteLLM 容器便會於背景自動啟動，無需手動干預。
