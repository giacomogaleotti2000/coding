# Foundry Standard Private Networking Lab (Terraform)

This folder now follows the private standard-agent architecture from Microsoft guidance and the Terraform sample family (BYO VNet pattern):
- VNet with delegated Agent subnet (`Microsoft.App/environments`)
- Private endpoint subnet
- Private DNS zones for Foundry + dependent data resources
- Private endpoints for Foundry account, Storage, Search, and Cosmos DB
- Foundry account and optional Foundry project
- Cheap Linux jumpbox VM (`Standard_B1s`) for DNS/connectivity tests

## Files
- `providers.tf`: provider and version requirements (`azurerm`, `azapi`, `random`)
- `variables.tf`: inputs
- `main.tf`: infrastructure resources
- `outputs.tf`: outputs
- `terraform.tfvars.example`: starter variable file

## Deploy

```powershell
cd foundry-architecture
terraform init
Copy-Item terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars
terraform plan -out main.tfplan
terraform apply main.tfplan
```

## Connect to the jumpbox

```powershell
ssh azureuser@$(terraform output -raw jumpbox_public_ip)
```

## Test private DNS from inside the VM

```bash
nslookup <foundry-account-name>.cognitiveservices.azure.com
nslookup <foundry-account-name>.openai.azure.com
nslookup <foundry-account-name>.services.ai.azure.com

nslookup <storage-account-name>.blob.core.windows.net
nslookup <search-name>.search.windows.net
nslookup <cosmos-account-name>.documents.azure.com
```

Expected:
- Hostnames resolve to private IPs from the PE subnet.
- Public network access is disabled on these resources.

## Optional model deployment

Model deployment is disabled by default (`deploy_model = false`) to avoid failures when model quota is missing.
Set `deploy_model = true` in `terraform.tfvars` when your subscription/region has quota for that model.

## Notes
- The classic secured flow is currently the supported flow for end-to-end private network isolation.
- This template gives you the infra baseline to practice private connectivity and troubleshooting from a VM.

## Cleanup

```powershell
terraform destroy
```
