#!/bin/bash
# rcloneのセットアップとGoogle Drive設定スクリプト

set -e

echo "=== rclone Setup for StreamVC ==="
echo ""

# rcloneがインストールされているか確認
if ! command -v rclone &> /dev/null; then
    echo "rclone is not installed. Installing via Homebrew..."
    brew install rclone
    echo "✓ rclone installed"
else
    echo "✓ rclone is already installed ($(rclone version | head -1))"
fi

echo ""
echo "=== Google Drive Configuration ==="
echo ""
echo "次のステップでrcloneとGoogle Driveを連携します:"
echo "1. ブラウザが開くのでGoogleアカウントでログイン"
echo "2. rcloneへのアクセスを許可"
echo "3. リモート名を 'gdrive' として設定"
echo ""
read -p "Press Enter to start configuration..."

# rclone config
# ユーザーに対話的に設定させる
if rclone listremotes | grep -q "^gdrive:"; then
    echo "✓ 'gdrive' remote is already configured"
    echo ""
    read -p "Reconfigure? (y/N): " reconfigure
    if [[ "$reconfigure" =~ ^[Yy]$ ]]; then
        rclone config reconnect gdrive:
    fi
else
    echo "Configuring new remote 'gdrive'..."
    echo ""
    echo "===== Configuration Instructions ====="
    echo "1. Type: 'n' (New remote)"
    echo "2. Name: 'gdrive'"
    echo "3. Storage: Select number for 'Google Drive' (usually 15 or 16)"
    echo "4. client_id: (press Enter to skip)"
    echo "5. client_secret: (press Enter to skip)"
    echo "6. scope: Select '1' (Full access)"
    echo "7. root_folder_id: (press Enter to skip)"
    echo "8. service_account_file: (press Enter to skip)"
    echo "9. Edit advanced config: 'n'"
    echo "10. Use auto config: 'y' (browser will open)"
    echo "11. Configure as team drive: 'n'"
    echo "12. Confirm: 'y'"
    echo "13. Quit: 'q'"
    echo "======================================"
    echo ""
    rclone config
fi

echo ""
echo "=== Testing Connection ==="
rclone lsd gdrive: --max-depth 1

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Run: ./scripts/upload_dataset_to_gdrive.sh"
echo "2. This will upload your local data/ to Google Drive"
