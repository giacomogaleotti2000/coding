output "resource_group_name" {
  value = azurerm_resource_group.this.name
}

output "virtual_network_id" {
  value = azurerm_virtual_network.this.id
}

output "agent_subnet_id" {
  value = azurerm_subnet.agent.id
}

output "private_endpoint_subnet_id" {
  value = azurerm_subnet.private_endpoints.id
}

output "jumpbox_subnet_id" {
  value = azurerm_subnet.jumpbox.id
}

output "jumpbox_public_ip" {
  value = azurerm_public_ip.jumpbox.ip_address
}

output "jumpbox_private_ip" {
  value = azurerm_network_interface.jumpbox.private_ip_address
}

output "foundry_account_name" {
  value = azapi_resource.foundry.name
}

output "foundry_account_id" {
  value = azapi_resource.foundry.id
}

output "foundry_project_id" {
  value = var.create_foundry_project ? azapi_resource.foundry_project[0].id : null
}

output "storage_account_name" {
  value = azurerm_storage_account.this.name
}

output "search_service_name" {
  value = azurerm_search_service.this.name
}

output "cosmos_account_name" {
  value = azurerm_cosmosdb_account.this.name
}

output "private_dns_zones" {
  value = { for k, z in azurerm_private_dns_zone.zones : k => z.name }
}
