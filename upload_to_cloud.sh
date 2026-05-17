#!/bin/bash
# ============================================================
# upload_to_cloud.sh
# Run this on YOUR WINDOWS machine (Git Bash) to upload
# all project files to the Oracle Cloud VM.
#
# Usage:
#   bash upload_to_cloud.sh <VM_IP>
#
# Example:
#   bash upload_to_cloud.sh 129.154.55.210
# ============================================================

VM_IP="${1:?Please provide VM IP: bash upload_to_cloud.sh <IP>}"
VM_USER="ubuntu"   # default Oracle Cloud user (change if different)
KEY_FILE="$HOME/.ssh/oracle_cloud_key"   # path to your SSH private key
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Uploading project to $VM_USER@$VM_IP ..."

# Files to upload (exclude cache, models, logs, venv)
rsync -avz --progress \
  --exclude='cache/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='venv/' \
  --exclude='*.log' \
  --exclude='results/' \
  --exclude='.env' \
  -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
  "$PROJECT_DIR/" \
  "$VM_USER@$VM_IP:~/agent_analyst/"

# Upload model separately (large file)
echo "Uploading model files..."
rsync -avz --progress \
  -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
  "$PROJECT_DIR/models/" \
  "$VM_USER@$VM_IP:~/agent_analyst/models/"

# Upload .env securely
echo "Uploading .env ..."
scp -i "$KEY_FILE" "$PROJECT_DIR/.env" "$VM_USER@$VM_IP:~/agent_analyst/.env"

echo ""
echo "✅ Upload complete."
echo ""
echo "Next: SSH into the VM and run setup:"
echo "  ssh -i $KEY_FILE $VM_USER@$VM_IP"
echo "  bash ~/agent_analyst/cloud_setup.sh"
