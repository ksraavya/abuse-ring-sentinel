from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InfrastructureState:
    """Exact event-time account/device/IP state used by Baseline B.

    Only account lifecycle events mutate this state. Transactions are read-only
    from the infrastructure perspective.
    """

    account_to_device: dict[str, str] = field(default_factory=dict)
    account_to_ip: dict[str, str] = field(default_factory=dict)
    device_to_accounts: dict[str, set[str]] = field(default_factory=dict)
    ip_to_accounts: dict[str, set[str]] = field(default_factory=dict)

    def add_or_update(self, account_id: str, device_id: str, ip_prefix: str) -> None:
        old_device = self.account_to_device.get(account_id)
        old_ip = self.account_to_ip.get(account_id)

        if old_device is not None and old_device != device_id:
            accounts = self.device_to_accounts.get(old_device)
            if accounts is not None:
                accounts.discard(account_id)
                if not accounts:
                    self.device_to_accounts.pop(old_device, None)

        if old_ip is not None and old_ip != ip_prefix:
            accounts = self.ip_to_accounts.get(old_ip)
            if accounts is not None:
                accounts.discard(account_id)
                if not accounts:
                    self.ip_to_accounts.pop(old_ip, None)

        self.account_to_device[account_id] = device_id
        self.account_to_ip[account_id] = ip_prefix
        self.device_to_accounts.setdefault(device_id, set()).add(account_id)
        self.ip_to_accounts.setdefault(ip_prefix, set()).add(account_id)

    def features(self, account_id: str) -> dict[str, int]:
        device = self.account_to_device.get(account_id)
        ip_prefix = self.account_to_ip.get(account_id)

        device_accounts = self.device_to_accounts.get(device, set()) if device else set()
        ip_accounts = self.ip_to_accounts.get(ip_prefix, set()) if ip_prefix else set()

        device_sharing = len(device_accounts)
        ip_sharing = len(ip_accounts)

        return {
            "degree": int((1 if device else 0) + (1 if ip_prefix else 0)),
            "device_degree": int(1 if device else 0),
            "ip_degree": int(1 if ip_prefix else 0),
            "shared_device_accounts": max(0, device_sharing - 1),
            "shared_ip_accounts": max(0, ip_sharing - 1),
            "max_device_sharing": int(device_sharing),
            "max_ip_sharing": int(ip_sharing),
        }
