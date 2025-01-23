from django.db import models

# Create your models here.
# Create your models here.
class Tenant(models.Model):
    tenant_id = models.CharField(max_length=255, unique=True)
    tenant_name = models.CharField(max_length=255)
    location = models.CharField(max_length=100)
    domain = models.CharField(max_length=50)
    default_language = models.CharField(max_length=100, choices=(('en', 'English'), ('de', 'German'), ('es', 'Spanish'), ('fr', 'French')), null=True, blank=True)
    is_active = models.BooleanField(default=True, help_text="Indicates if the filter is currently active.")
    created_at = models.DateTimeField(auto_now_add=True)
    meta_info = models.JSONField(null=True, blank=True)
    
    class Meta:
        db_table = "wa_tenant"
        verbose_name_plural = "Tenants"
        
    def __str__(self):
        return f"{self.tenant_name}"