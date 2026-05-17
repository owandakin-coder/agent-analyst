#!/bin/bash
# ============================================================
# cloud_setup.sh
# Oracle Cloud Free VM – Agent Analyst setup script
# Run this ONCE after SSHing into your VM:
#   bash cloud_setup.sh
# ============================================================
set -e

echo "============================================"
echo " Agent Analyst – Cloud Setup"
echo "============================================"

# ── 1. System packages ───────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git unzip curl

# ── 2. Project directory ─────────────────────────────────────
echo "[2/6] Setting up project directory..."
mkdir -p ~/agent_analyst
cd ~/agent_analyst

# ── 3. Python virtual environment ────────────────────────────
echo "[3/6] Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# ── 4. Install Python dependencies ───────────────────────────
echo "[4/6] Installing Python packages (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -q \
    stable-baselines3 \
    gymnasium \
    alpaca-py \
    yfinance \
    pandas \
    numpy \
    optuna \
    python-dotenv \
    matplotlib \
    seaborn \
    scipy \
    torch --index-url https://download.pytorch.org/whl/cpu

echo "[5/6] Dependencies installed."

# ── 5. .env reminder ─────────────────────────────────────────
if [ ! -f .env ]; then
    cat > .env << 'EOF'
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
EOF
    echo ""
    echo "⚠️  Edit .env with your credentials:"
    echo "    nano ~/agent_analyst/.env"
fi

# ── 6. systemd service ───────────────────────────────────────
echo "[6/6] Installing systemd service..."
sudo bash -c "cat > /etc/systemd/system/agent-analyst.service" << EOF
[Unit]
Description=Agent Analyst RL Trading Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/home/$(whoami)/agent_analyst
ExecStart=/home/$(whoami)/agent_analyst/venv/bin/python main.py --mode live_paper --auto-approve
Restart=always
RestartSec=30
StandardOutput=append:/home/$(whoami)/agent_analyst/agent_analyst.log
StandardError=append:/home/$(whoami)/agent_analyst/agent_analyst.log
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable agent-analyst
sudo systemctl start agent-analyst

echo ""
echo "============================================"
echo " ✅ Setup complete!"
echo "============================================"
echo " Check status:  sudo systemctl status agent-analyst"
echo " View logs:     tail -f ~/agent_analyst/agent_analyst.log"
echo " Stop agent:    sudo systemctl stop agent-analyst"
echo " Restart:       sudo systemctl restart agent-analyst"
echo "============================================"
