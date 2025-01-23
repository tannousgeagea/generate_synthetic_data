from django.contrib import admin
from unfold.admin import ModelAdmin, TabularInline, StackedInline
from tenants.models import Tenant

@admin.register(Tenant)
class TenantAdmin(ModelAdmin):
    list_display = ('tenant_id', 'tenant_name', 'location', 'domain', 'is_active', 'created_at')
    search_fields = ('tenant_name', 'location', 'domain')
    list_filter = ('is_active',)