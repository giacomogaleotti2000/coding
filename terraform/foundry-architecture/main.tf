resource "random_string" "suffix" {
  length  = 5
  upper   = false
  special = false
}

locals {
  suffix               = random_string.suffix.result
  base_name            = "${var.name_prefix}-${local.suffix}"
  storage_account_name = substr(replace("${var.name_prefix}${local.suffix}st", "-", ""), 0, 24)
  search_name          = "${var.name_prefix}-${local.suffix}-srch"
  cosmos_name          = "${var.name_prefix}-${local.suffix}-cosmos"
  foundry_name         = "${var.name_prefix}-${local.suffix}-fdry"
  vm_name              = "${var.name_prefix}-${local.suffix}-vm"
}

resource "azurerm_resource_group" "this" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

resource "azurerm_virtual_network" "this" {
  name                = "${local.base_name}-vnet"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  address_space       = [var.vnet_cidr]
  tags                = var.tags
}

resource "azurerm_subnet" "agent" {
  name                 = "snet-agent"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [var.agent_subnet_cidr]

  delegation {
    name = "agent-delegation"

    service_delegation {
      name = "Microsoft.App/environments"
      actions = [
        "Microsoft.Network/virtualNetworks/subnets/join/action"
      ]
    }
  }
}

resource "azurerm_subnet" "private_endpoints" {
  name                 = "snet-private-endpoints"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [var.private_endpoint_subnet_cidr]

  private_endpoint_network_policies = "Disabled"
}

resource "azurerm_subnet" "jumpbox" {
  name                 = "snet-jumpbox"
  resource_group_name  = azurerm_resource_group.this.name
  virtual_network_name = azurerm_virtual_network.this.name
  address_prefixes     = [var.jumpbox_subnet_cidr]
}

resource "azurerm_private_dns_zone" "zones" {
  for_each = {
    blob            = "privatelink.blob.core.windows.net"
    file            = "privatelink.file.core.windows.net"
    cognitive       = "privatelink.cognitiveservices.azure.com"
    documents       = "privatelink.documents.azure.com"
    search          = "privatelink.search.windows.net"
    openai          = "privatelink.openai.azure.com"
    foundryservices = "privatelink.services.ai.azure.com"
  }

  name                = each.value
  resource_group_name = azurerm_resource_group.this.name
  tags                = var.tags
}

resource "azurerm_private_dns_zone_virtual_network_link" "zone_links" {
  for_each = azurerm_private_dns_zone.zones

  name                  = "${each.key}-vnet-link"
  resource_group_name   = azurerm_resource_group.this.name
  private_dns_zone_name = each.value.name
  virtual_network_id    = azurerm_virtual_network.this.id
}

resource "azurerm_storage_account" "this" {
  name                          = local.storage_account_name
  resource_group_name           = azurerm_resource_group.this.name
  location                      = azurerm_resource_group.this.location
  account_tier                  = "Standard"
  account_replication_type      = "LRS"
  public_network_access_enabled = false
  min_tls_version               = "TLS1_2"

  network_rules {
    default_action = "Deny"
    bypass         = ["None"]
  }

  tags = var.tags
}

resource "azurerm_search_service" "this" {
  name                          = local.search_name
  resource_group_name           = azurerm_resource_group.this.name
  location                      = azurerm_resource_group.this.location
  sku                           = var.search_sku
  public_network_access_enabled = false
  local_authentication_enabled  = true
  tags                          = var.tags
}

resource "azurerm_cosmosdb_account" "this" {
  name                          = local.cosmos_name
  location                      = azurerm_resource_group.this.location
  resource_group_name           = azurerm_resource_group.this.name
  offer_type                    = "Standard"
  kind                          = "GlobalDocumentDB"
  public_network_access_enabled = false

  consistency_policy {
    consistency_level = "Session"
  }

  geo_location {
    location          = azurerm_resource_group.this.location
    failover_priority = 0
  }

  capabilities {
    name = "EnableServerless"
  }

  tags = var.tags
}

resource "azapi_resource" "foundry" {
  type                      = "Microsoft.CognitiveServices/accounts@2025-06-01"
  name                      = local.foundry_name
  parent_id                 = azurerm_resource_group.this.id
  location                  = azurerm_resource_group.this.location
  schema_validation_enabled = false
  tags                      = var.tags

  body = {
    kind = "AIServices"
    sku = {
      name = var.ai_services_sku
    }
    identity = {
      type = "SystemAssigned"
    }
    properties = {
      allowProjectManagement = true
      customSubDomainName    = "${var.name_prefix}-${local.suffix}-ais"
      publicNetworkAccess    = "Disabled"
    }
  }
}

resource "azapi_resource" "foundry_project" {
  count                     = var.create_foundry_project ? 1 : 0
  type                      = "Microsoft.CognitiveServices/accounts/projects@2025-06-01"
  name                      = var.foundry_project_name
  parent_id                 = azapi_resource.foundry.id
  location                  = azurerm_resource_group.this.location
  schema_validation_enabled = false

  body = {
    sku = {
      name = "S0"
    }
    identity = {
      type = "SystemAssigned"
    }
    properties = {
      displayName = var.foundry_project_display_name
      description = var.foundry_project_description
    }
  }
}

resource "azapi_resource" "foundry_model" {
  count                     = var.deploy_model ? 1 : 0
  type                      = "Microsoft.CognitiveServices/accounts/deployments@2023-05-01"
  name                      = var.model_deployment_name
  parent_id                 = azapi_resource.foundry.id
  schema_validation_enabled = false

  body = {
    sku = {
      name     = var.model_sku_name
      capacity = var.model_capacity
    }
    properties = {
      model = {
        format  = "OpenAI"
        name    = var.model_name
        version = var.model_version
      }
    }
  }
}

resource "azurerm_private_endpoint" "storage_blob" {
  name                = "${local.base_name}-pe-st-blob"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = var.tags

  private_service_connection {
    name                           = "storage-blob-connection"
    private_connection_resource_id = azurerm_storage_account.this.id
    subresource_names              = ["blob"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "blob-dns"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["blob"].id]
  }
}

resource "azurerm_private_endpoint" "storage_file" {
  name                = "${local.base_name}-pe-st-file"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = var.tags

  private_service_connection {
    name                           = "storage-file-connection"
    private_connection_resource_id = azurerm_storage_account.this.id
    subresource_names              = ["file"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "file-dns"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["file"].id]
  }
}

resource "azurerm_private_endpoint" "search" {
  name                = "${local.base_name}-pe-search"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = var.tags

  private_service_connection {
    name                           = "search-connection"
    private_connection_resource_id = azurerm_search_service.this.id
    subresource_names              = ["searchService"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "search-dns"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["search"].id]
  }
}

resource "azurerm_private_endpoint" "cosmos_sql" {
  name                = "${local.base_name}-pe-cosmos"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = var.tags

  private_service_connection {
    name                           = "cosmos-connection"
    private_connection_resource_id = azurerm_cosmosdb_account.this.id
    subresource_names              = ["Sql"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name                 = "cosmos-dns"
    private_dns_zone_ids = [azurerm_private_dns_zone.zones["documents"].id]
  }
}

resource "azurerm_private_endpoint" "foundry" {
  name                = "${local.base_name}-pe-foundry"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  subnet_id           = azurerm_subnet.private_endpoints.id
  tags                = var.tags

  private_service_connection {
    name                           = "foundry-connection"
    private_connection_resource_id = azapi_resource.foundry.id
    subresource_names              = ["account"]
    is_manual_connection           = false
  }

  private_dns_zone_group {
    name = "foundry-dns"
    private_dns_zone_ids = [
      azurerm_private_dns_zone.zones["cognitive"].id,
      azurerm_private_dns_zone.zones["openai"].id,
      azurerm_private_dns_zone.zones["foundryservices"].id
    ]
  }
}

resource "azurerm_network_security_group" "jumpbox" {
  name                = "${local.base_name}-nsg-jumpbox"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  tags                = var.tags
}

resource "azurerm_network_security_rule" "allow_ssh" {
  name                        = "Allow-SSH-Inbound"
  priority                    = 100
  direction                   = "Inbound"
  access                      = "Allow"
  protocol                    = "Tcp"
  source_port_range           = "*"
  destination_port_range      = "22"
  source_address_prefix       = var.ssh_allowed_cidr
  destination_address_prefix  = "*"
  resource_group_name         = azurerm_resource_group.this.name
  network_security_group_name = azurerm_network_security_group.jumpbox.name
}

resource "azurerm_subnet_network_security_group_association" "jumpbox" {
  subnet_id                 = azurerm_subnet.jumpbox.id
  network_security_group_id = azurerm_network_security_group.jumpbox.id
}

resource "azurerm_public_ip" "jumpbox" {
  name                = "${local.base_name}-pip-jumpbox"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  allocation_method   = "Static"
  sku                 = "Standard"
  tags                = var.tags
}

resource "azurerm_network_interface" "jumpbox" {
  name                = "${local.base_name}-nic-jumpbox"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  tags                = var.tags

  ip_configuration {
    name                          = "ipconfig1"
    subnet_id                     = azurerm_subnet.jumpbox.id
    private_ip_address_allocation = "Dynamic"
    public_ip_address_id          = azurerm_public_ip.jumpbox.id
  }
}

resource "azurerm_linux_virtual_machine" "jumpbox" {
  name                = local.vm_name
  resource_group_name = azurerm_resource_group.this.name
  location            = azurerm_resource_group.this.location
  size                = var.vm_size
  admin_username      = var.admin_username

  network_interface_ids = [
    azurerm_network_interface.jumpbox.id
  ]

  admin_ssh_key {
    username   = var.admin_username
    public_key = var.ssh_public_key
  }

  os_disk {
    caching              = "ReadWrite"
    storage_account_type = "Standard_LRS"
  }

  source_image_reference {
    publisher = "Canonical"
    offer     = "0001-com-ubuntu-server-jammy"
    sku       = "22_04-lts"
    version   = "latest"
  }

  disable_password_authentication = true
  tags                            = var.tags
}
