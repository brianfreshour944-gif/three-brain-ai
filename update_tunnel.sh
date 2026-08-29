#!/bin/bash
# Usage: ./update_tunnel.sh <ministral_tunnel_url> <deepseek_tunnel_url>

MINISTRAL_URL="$1"
DEEPSEEK_URL="$2"

if [ -z "$MINISTRAL_URL" ] || [ -z "$DEEPSEEK_URL" ]; then
    echo "❌ Please provide both tunnel URLs"
    echo "Usage: ./update_tunnel.sh https://ministral-tunnel.trycloudflare.com https://deepseek-tunnel.trycloudflare.com"
    exit 1
fi

# Remove trailing slashes
MINISTRAL_URL="${MINISTRAL_URL%/}"
DEEPSEEK_URL="${DEEPSEEK_URL%/}"

cd /home/ubuntu/three-brain-ai

# Backup existing .env
cp .env .env.backup.$(date +%s)

# Update the URLs in .env
sed -i "s|MINISTRAL_BASE_URL=.*|MINISTRAL_BASE_URL=${MINISTRAL_URL}/v1|" .env
sed -i "s|DEEPSEEK_BASE_URL=.*|DEEPSEEK_BASE_URL=${DEEPSEEK_URL}/v1|" .env

echo "✅ Updated Ministral tunnel: $MINISTRAL_URL"
echo "✅ Updated DeepSeek tunnel: $DEEPSEEK_URL"

# Show current config
echo ""
echo "Current .env:"
grep -E "MINISTRAL_BASE_URL|DEEPSEEK_BASE_URL|OPENROUTER" .env

# Run health check
echo ""
echo "Running health check..."
source venv/bin/activate
three-brain health