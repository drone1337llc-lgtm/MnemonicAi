"""Create special accounts (admin / business) directly in the tenant store.
Usage: python create_account.py <role: admin|business> <email> [display_name]
Prints the raw API key ONCE — save it.
"""
import sys, os
from tenants import TenantStore
role = sys.argv[1] if len(sys.argv) > 1 else "admin"
email = sys.argv[2] if len(sys.argv) > 2 else f"{role}@mnemonicai.org"
name  = sys.argv[3] if len(sys.argv) > 3 else role.capitalize()
assert role in ("admin", "business"), "role must be admin or business"
store = TenantStore(root=os.environ.get("MNEM_TENANTS", "/workspace/tenants"))
raw, t = store.create(email=email, tier=role, display_name=name, role=role)
print(f"role={t.role} tier={t.tier} tenant_id={t.tenant_id} email={t.email}")
print(f"API KEY (save this, shown once): {raw}")
