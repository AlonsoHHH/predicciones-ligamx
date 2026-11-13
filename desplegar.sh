#!/usr/bin/env bash
# Instala la interfaz como servicio en un VPS Ubuntu/Debian, en el puerto 8501:
#     sudo bash desplegar.sh
# Para actualizar después: git pull && sudo bash desplegar.sh
set -euo pipefail
cd "$(dirname "$0")"
DIR=$(pwd)
USUARIO=${SUDO_USER:-root}

apt-get update -qq
apt-get install -y -qq python3-venv
python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt

if [ ! -f .env ]; then
    read -rsp "Token de Sportmonks (Enter para dejarlo vacío): " TOKEN; echo
    read -rsp "Contraseña para entrar a la app: " CLAVE; echo
    [ -n "$CLAVE" ] || { echo "La contraseña no puede quedar vacía: la app queda abierta a internet."; exit 1; }
    printf 'SPORTMONKS_API_TOKEN=%s\nCLAVE_APP=%s\n' "$TOKEN" "$CLAVE" > .env
    chmod 600 .env
fi
chown -R "$USUARIO" "$DIR"

cat > /etc/systemd/system/ligamx.service <<UNIT
[Unit]
Description=Predicciones Liga MX
After=network-online.target

[Service]
User=$USUARIO
WorkingDirectory=$DIR
EnvironmentFile=$DIR/.env
ExecStart=$DIR/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable ligamx
systemctl restart ligamx
command -v ufw >/dev/null && ufw status | grep -q "Status: active" && ufw allow 8501/tcp
echo "Lista: http://$(hostname -I | awk '{print $1}'):8501"
