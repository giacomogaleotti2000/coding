variable "location" {
  description = "Azure region for all resources. Must support Foundry Agent private networking."
  type        = string
  default     = "italynorth"
}

variable "resource_group_name" {
  description = "Resource group name."
  type        = string
  default     = "rg-foundry-architecture-lab"
}

variable "name_prefix" {
  description = "Prefix used to generate resource names. Use lowercase letters and numbers."
  type        = string
  default     = "fdryl"
}

variable "tags" {
  description = "Common tags."
  type        = map(string)
  default = {
    project = "foundry-architecture"
    env     = "lab"
  }
}

variable "vnet_cidr" {
  description = "Address space for the lab VNet."
  type        = string
  default     = "192.168.0.0/16"
}

variable "agent_subnet_cidr" {
  description = "Delegated subnet for Agent compute."
  type        = string
  default     = "192.168.0.0/24"
}

variable "private_endpoint_subnet_cidr" {
  description = "Subnet for private endpoints."
  type        = string
  default     = "192.168.1.0/24"
}

variable "jumpbox_subnet_cidr" {
  description = "Subnet for test VM (jumpbox)."
  type        = string
  default     = "192.168.2.0/24"
}

variable "ssh_allowed_cidr" {
  description = "Public IP CIDR allowed to SSH to the jumpbox VM (for example x.x.x.x/32)."
  type        = string
}

variable "admin_username" {
  description = "Linux admin username for jumpbox VM."
  type        = string
  default     = "azureuser"
}

variable "ssh_public_key" {
  description = "SSH public key content for VM login."
  type        = string
}

variable "vm_size" {
  description = "Low cost VM size for testing."
  type        = string
  default     = "Standard_B1s"
}

variable "search_sku" {
  description = "Azure AI Search SKU."
  type        = string
  default     = "basic"
}

variable "ai_services_sku" {
  description = "Foundry account SKU."
  type        = string
  default     = "S0"
}

variable "create_foundry_project" {
  description = "Create a Foundry project inside the Foundry account."
  type        = bool
  default     = true
}

variable "foundry_project_name" {
  description = "Foundry project resource name."
  type        = string
  default     = "project"
}

variable "foundry_project_display_name" {
  description = "Foundry project display name."
  type        = string
  default     = "project"
}

variable "foundry_project_description" {
  description = "Foundry project description."
  type        = string
  default     = "Private networking lab project"
}

variable "deploy_model" {
  description = "Deploy a model in the Foundry account. Disable if your subscription has no quota yet."
  type        = bool
  default     = false
}

variable "model_deployment_name" {
  description = "Model deployment name."
  type        = string
  default     = "gpt-4o"
}

variable "model_name" {
  description = "Model name."
  type        = string
  default     = "gpt-4o"
}

variable "model_version" {
  description = "Model version."
  type        = string
  default     = "2024-11-20"
}

variable "model_sku_name" {
  description = "Model deployment SKU name."
  type        = string
  default     = "GlobalStandard"
}

variable "model_capacity" {
  description = "Model deployment capacity."
  type        = number
  default     = 1
}
