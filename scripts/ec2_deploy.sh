#!/usr/bin/env bash
# Runs ON the EC2 instance (sent by the CD workflow through SSM Run Command).
#
# Syncs the deploy bundle (compose file + nginx/monitoring config) from S3, points the
# stack to the new API image in ECR and restarts only what changed.
#
# Required env: API_IMAGE (ECR image with tag), AWS_REGION, BUNDLE_URI (s3://.../deploy/<sha>/)
set -euo pipefail
: "${API_IMAGE:?}" "${AWS_REGION:?}" "${BUNDLE_URI:?}"

APP_DIR=/opt/triage
COMPOSE="docker compose -f docker-compose.prod.yml"
mkdir -p "$APP_DIR" && cd "$APP_DIR"

aws s3 sync --delete --exclude ".env" --exclude ".grafana_password" "$BUNDLE_URI" .

# Grafana admin password: generated once on the instance, kept across deploys.
# Read it with: sudo cat /opt/triage/.grafana_password (SSM Session Manager)
[ -f .grafana_password ] || (umask 077 && openssl rand -base64 18 > .grafana_password)

# Public IP (IMDSv2) for the links Prometheus generates behind nginx
TOKEN=$(curl -s -X PUT http://169.254.169.254/latest/api/token \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4 || true)

cat > .env <<EOF
API_IMAGE=$API_IMAGE
PUBLIC_URL=http://${PUBLIC_IP:-localhost}
GRAFANA_ADMIN_PASSWORD=$(cat .grafana_password)
EOF
chmod 600 .env

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${API_IMAGE%%/*}"

$COMPOSE pull --quiet
$COMPOSE up -d --remove-orphans
docker image prune -f >/dev/null

for _ in $(seq 1 30); do
  if curl -sf http://localhost/health; then
    echo
    echo "Deployed $API_IMAGE"
    exit 0
  fi
  sleep 2
done
echo "API did not become healthy" >&2
$COMPOSE logs --tail 50 api >&2
exit 1
