import hmac
import json

from fastapi import Header, HTTPException, Request


class TenantRegistry:
    def __init__(self, tenants):
        self.tenants = tenants

    @classmethod
    def from_file(cls, path):
        return cls(json.loads(path.read_text(encoding="utf-8"))["tenants"])

    def resolve(self, api_key):
        for tenant_id, info in self.tenants.items():
            if hmac.compare_digest(api_key.encode(), info["api_key"].encode()):  # constant-time compare
                return tenant_id
        return None

    def foreign_terms(self, tenant_id):
        terms = []
        for tid, info in self.tenants.items():
            if tid != tenant_id:
                terms += [tid, info.get("display_name", "")]
        return [t for t in terms if t]


def current_tenant(request: Request, x_api_key: str | None = Header(default=None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    tenant = request.app.state.tenants.resolve(x_api_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return tenant