#!/usr/bin/env bash
# deploy_azure.sh: deploy the app to Azure Container Apps.
#
# One-time setup:
#   1. An Azure for Students account (azure.microsoft.com/free/students)
#   2. Azure CLI installed and signed in:
#        curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#        az login --use-device-code
#   3. Docker running (GitHub Codespaces has it already)
#
# Usage (from the repo root):
#   bash scripts/deploy_azure.sh <registry-name> [version]
#   e.g. bash scripts/deploy_azure.sh ca7dcf67ebb3acr v2
#
# The image is built locally and pushed, because Azure for Students
# subscriptions do not allow building images in the cloud (ACR Tasks).
set -euo pipefail

ACR="${1:?Usage: bash scripts/deploy_azure.sh <registry-name> [version]}"
VERSION="${2:-v1}"
APP=cassava-leaf-doctor
RG=cassava-rg
LOCATION=southafricanorth
IMAGE="$ACR.azurecr.io/$APP:$VERSION"

echo "== Enabling Container Apps (safe to repeat)"
az extension add --name containerapp --upgrade --only-show-errors
az provider register -n Microsoft.App --wait
az provider register -n Microsoft.OperationalInsights --wait
az provider register -n Microsoft.ContainerRegistry --wait

echo "== Creating the resource group and registry if they do not exist"
az group create --name "$RG" --location "$LOCATION" --output none
az acr show --name "$ACR" --output none 2>/dev/null || \
  az acr create --name "$ACR" --resource-group "$RG" --sku Basic --output none
az acr update --name "$ACR" --admin-enabled true --output none

echo "== Building and pushing $IMAGE"
az acr login --name "$ACR"
docker build -t "$IMAGE" .
docker push "$IMAGE"

echo "== Deploying the container"
ACR_PW=$(az acr credential show --name "$ACR" --query "passwords[0].value" -o tsv)
az containerapp up --name "$APP" --resource-group "$RG" --location "$LOCATION" \
  --image "$IMAGE" --ingress external --target-port 7860 \
  --registry-server "$ACR.azurecr.io" --registry-username "$ACR" --registry-password "$ACR_PW"

echo "== Setting resources: 1 vCPU, 2 GB RAM, scale to zero when idle, max 1 replica"
az containerapp update --name "$APP" --resource-group "$RG" \
  --cpu 1 --memory 2Gi --min-replicas 0 --max-replicas 1 --output none

URL=$(az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)
echo
echo "Live app: https://$URL"
echo "API docs: https://$URL/docs"
