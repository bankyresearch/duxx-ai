"""Role-Based Access Control for enterprise agent governance."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Permission(BaseModel):
    resource: str  # e.g., "tool:web_search", "agent:researcher", "finetune:*"
    actions: list[str] = Field(default_factory=lambda: ["read"])  # read, write, execute, admin


class RoleDefinition(BaseModel):
    name: str
    description: str = ""
    permissions: list[Permission] = Field(default_factory=list)
    inherits: list[str] = Field(default_factory=list)  # inherit from other roles


class User(BaseModel):
    id: str
    name: str = ""
    roles: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Default enterprise roles
DEFAULT_ROLES = {
    "viewer": RoleDefinition(
        name="viewer",
        description="Read-only access to agents and traces",
        permissions=[
            Permission(resource="agent:*", actions=["read"]),
            Permission(resource="trace:*", actions=["read"]),
        ],
    ),
    "operator": RoleDefinition(
        name="operator",
        description="Can run agents and view traces",
        permissions=[
            Permission(resource="agent:*", actions=["read", "execute"]),
            Permission(resource="tool:*", actions=["read", "execute"]),
            Permission(resource="trace:*", actions=["read"]),
        ],
        inherits=["viewer"],
    ),
    "developer": RoleDefinition(
        name="developer",
        description="Can create and modify agents, tools, and run fine-tuning",
        permissions=[
            Permission(resource="agent:*", actions=["read", "write", "execute"]),
            Permission(resource="tool:*", actions=["read", "write", "execute"]),
            Permission(resource="finetune:*", actions=["read", "write", "execute"]),
            Permission(resource="trace:*", actions=["read", "write"]),
        ],
        inherits=["operator"],
    ),
    "admin": RoleDefinition(
        name="admin",
        description="Full access to all resources",
        permissions=[Permission(resource="*", actions=["read", "write", "execute", "admin"])],
    ),
}


class RBACManager:
    """Manages roles, users, and permission checks."""

    def __init__(self) -> None:
        self.roles: dict[str, RoleDefinition] = dict(DEFAULT_ROLES)
        self.users: dict[str, User] = {}

    def add_role(self, role: RoleDefinition) -> None:
        self.roles[role.name] = role

    def add_user(self, user: User) -> None:
        self.users[user.id] = user

    def assign_role(self, user_id: str, role_name: str) -> None:
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        user = self.users.get(user_id)
        if user is None:
            raise ValueError(f"User '{user_id}' not found")
        if role_name not in user.roles:
            user.roles.append(role_name)

    def check_permission(self, user_id: str, resource: str, action: str) -> bool:
        user = self.users.get(user_id)
        if user is None:
            return False

        all_roles = self._resolve_roles(user.roles)
        for role in all_roles:
            for perm in role.permissions:
                if self._matches_resource(perm.resource, resource) and action in perm.actions:
                    return True
        return False

    def _resolve_roles(self, role_names: list[str]) -> list[RoleDefinition]:
        resolved: list[RoleDefinition] = []
        visited: set[str] = set()

        def _collect(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            role = self.roles.get(name)
            if role is None:
                return
            for parent in role.inherits:
                _collect(parent)
            resolved.append(role)

        for rn in role_names:
            _collect(rn)
        return resolved

    @staticmethod
    def _matches_resource(pattern: str, resource: str) -> bool:
        if pattern == "*":
            return True
        if pattern.endswith(":*"):
            prefix = pattern[:-1]
            return resource.startswith(prefix) or resource == pattern[:-2]
        return pattern == resource
