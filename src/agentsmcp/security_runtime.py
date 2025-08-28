"""
Legacy full SecurityManager implementation (kept for future use).

Note: The default import path `agentsmcp.security` now points to the
package-based stub in `agentsmcp/security/manager.py` to avoid name
collisions and simplify Phase 1 bring-up.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set, Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from .runtime_config import Config

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    pass


class KeyValidationError(SecurityError):
    pass


class SignatureValidationError(SecurityError):
    pass


class TrustStoreError(SecurityError):
    pass


class KeyManager:
    def __init__(self, config: Config):
        self.config = config
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.key_loaded_time: Optional[float] = None

    def load_or_generate_keys(self) -> None:
        try:
            self.load_keys()
            logger.info("Loaded existing RSA key pair")
        except (FileNotFoundError, KeyValidationError):
            logger.info("Generating new RSA key pair")
            self.generate_keys()
            if self.config.private_key_path and self.config.public_key_path:
                self.save_keys()
                logger.info("Saved new RSA key pair")

    def generate_keys(self, key_size: int = 2048) -> None:
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend(),
        )
        self.public_key = self.private_key.public_key()
        self.key_loaded_time = time.time()

    def load_keys(self) -> None:
        if not self.config.private_key_path or not self.config.public_key_path:
            raise KeyValidationError("Key paths not configured")

        private_path = Path(self.config.private_key_path)
        public_path = Path(self.config.public_key_path)

        if not private_path.exists():
            raise FileNotFoundError(f"Private key not found: {private_path}")
        if not public_path.exists():
            raise FileNotFoundError(f"Public key not found: {public_path}")

        with open(private_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        with open(public_path, "rb") as f:
            self.public_key = serialization.load_pem_public_key(
                f.read(), backend=default_backend()
            )

        self.key_loaded_time = time.time()

        if not self._validate_key_pair():
            raise KeyValidationError("Private and public keys do not match")

    def save_keys(self) -> None:
        if not self.private_key or not self.public_key:
            raise KeyValidationError("No keys loaded to save")
        if not self.config.private_key_path or not self.config.public_key_path:
            raise KeyValidationError("Key paths not configured")

        private_path = Path(self.config.private_key_path)
        public_path = Path(self.config.public_key_path)

        private_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)

        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(private_path, "wb") as f:
            f.write(private_pem)
        os.chmod(private_path, 0o600)

        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with open(public_path, "wb") as f:
            f.write(public_pem)

    def _validate_key_pair(self) -> bool:
        if not self.private_key or not self.public_key:
            return False
        test_data = b"test message for key validation"
        try:
            signature = self.sign_data(test_data)
            return self.verify_signature(test_data, signature, self.public_key)
        except Exception:
            return False

    def should_rotate_keys(self) -> bool:
        if not self.key_loaded_time:
            return True
        age_hours = (time.time() - self.key_loaded_time) / 3600
        return age_hours >= self.config.key_rotation_interval_hours

    def sign_data(self, data: bytes) -> bytes:
        if not self.private_key:
            raise KeyValidationError("No private key loaded")
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return signature

    def verify_signature(
        self, data: bytes, signature: bytes, public_key: rsa.RSAPublicKey
    ) -> bool:
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except InvalidSignature:
            return False

    def get_public_key_pem(self) -> str:
        if not self.public_key:
            raise KeyValidationError("No public key loaded")
        pem_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return pem_bytes.decode("utf-8")


class MessageSigner:
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager

    def sign_agent_message(self, agent_dict: Dict[str, Any]) -> str:
        canonical_json = json.dumps(agent_dict, separators=(",", ":"), sort_keys=True)
        canonical_bytes = canonical_json.encode("utf-8")
        signature_bytes = self.key_manager.sign_data(canonical_bytes)
        return base64.b64encode(signature_bytes).decode("utf-8")

    def verify_agent_message(
        self, agent_dict: Dict[str, Any], signature_b64: str, public_key_pem: str
    ) -> bool:
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )
            canonical_json = json.dumps(agent_dict, separators=(",", ":"), sort_keys=True)
            canonical_bytes = canonical_json.encode("utf-8")
            signature_bytes = base64.b64decode(signature_b64)
            return self.key_manager.verify_signature(
                canonical_bytes, signature_bytes, public_key
            )
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False


class TrustStore:
    def __init__(self, config: Config):
        self.config = config
        self.trusted_keys: Dict[str, rsa.RSAPublicKey] = {}
        self.allowlist: Set[str] = set(config.discovery_allowlist)
        self.trust_store_path = (
            Path(config.trust_store_path) if config.trust_store_path else None
        )
        if self.trust_store_path:
            self.load_trusted_keys()

    def load_trusted_keys(self) -> None:
        if not self.trust_store_path or not self.trust_store_path.exists():
            logger.info("Trust store path not configured or doesn't exist")
            return
        for key_file in self.trust_store_path.glob("*.pem"):
            try:
                agent_id = key_file.stem
                with open(key_file, "rb") as f:
                    public_key = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
                self.trusted_keys[agent_id] = public_key
            except Exception as e:
                logger.warning(f"Failed to load trusted key {key_file}: {e}")

    def add_trusted_key(self, agent_id: str, public_key_pem: str) -> None:
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8"), backend=default_backend()
            )
            self.trusted_keys[agent_id] = public_key
            if self.trust_store_path:
                self.trust_store_path.mkdir(parents=True, exist_ok=True)
                key_file = self.trust_store_path / f"{agent_id}.pem"
                with open(key_file, "w") as f:
                    f.write(public_key_pem)
        except Exception as e:
            raise TrustStoreError(f"Failed to add trusted key for {agent_id}: {e}")

    def is_agent_trusted(self, agent_id: str) -> bool:
        if self.allowlist and agent_id not in self.allowlist:
            return False
        if not self.trusted_keys:
            return not self.allowlist or agent_id in self.allowlist
        return agent_id in self.trusted_keys

    def get_trusted_key(self, agent_id: str) -> Optional[rsa.RSAPublicKey]:
        return self.trusted_keys.get(agent_id)

    def verify_agent_signature(
        self, agent_id: str, agent_dict: Dict[str, Any], signature_b64: str
    ) -> bool:
        if not self.is_agent_trusted(agent_id):
            logger.warning(f"Agent {agent_id} is not trusted")
            return False

        trusted_key = self.get_trusted_key(agent_id)
        if not trusted_key:
            embedded_key_pem = agent_dict.get("public_key")
            if not embedded_key_pem:
                logger.warning(f"No public key in message from agent {agent_id}")
                return False
            try:
                public_key = serialization.load_pem_public_key(
                    embedded_key_pem.encode("utf-8"), backend=default_backend()
                )
                canonical_json = json.dumps(agent_dict, separators=(",", ":"), sort_keys=True)
                signature_bytes = base64.b64decode(signature_b64)
                key_manager = KeyManager(self.config)
                result = key_manager.verify_signature(
                    canonical_json.encode("utf-8"), signature_bytes, public_key
                )
                if result:
                    self.add_trusted_key(agent_id, embedded_key_pem)
                return result
            except Exception as e:
                logger.warning(
                    f"Failed to verify embedded key for agent {agent_id}: {e}"
                )
                return False

        try:
            canonical_json = json.dumps(agent_dict, separators=(",", ":"), sort_keys=True)
            signature_bytes = base64.b64decode(signature_b64)
            key_manager = KeyManager(self.config)
            return key_manager.verify_signature(
                canonical_json.encode("utf-8"), signature_bytes, trusted_key
            )
        except Exception as e:
            logger.warning(f"Signature verification failed for agent {agent_id}: {e}")
            return False


class SecurityManager:
    def __init__(self, config: Config):
        self.config = config
        self.key_manager = KeyManager(config)
        self.message_signer = MessageSigner(self.key_manager)
        self.trust_store = TrustStore(config)
        if config.security_enabled:
            self.key_manager.load_or_generate_keys()

    def is_enabled(self) -> bool:
        return self.config.security_enabled

    def sign_announcement(self, agent_dict: Dict[str, Any]) -> Optional[str]:
        if not self.is_enabled():
            return None
        try:
            return self.message_signer.sign_agent_message(agent_dict)
        except Exception as e:
            logger.error(f"Failed to sign announcement: {e}")
            raise SignatureValidationError(f"Signature creation failed: {e}")

    def validate_announcement(
        self, agent_id: str, agent_dict: Dict[str, Any], signature_b64: Optional[str]
    ) -> bool:
        if not self.is_enabled():
            return True
        if not signature_b64:
            logger.warning(f"Missing signature in announcement from {agent_id}")
            return False
        return self.trust_store.verify_agent_signature(agent_id, agent_dict, signature_b64)

    def get_public_key_pem(self) -> Optional[str]:
        if not self.is_enabled():
            return None
        return self.key_manager.get_public_key_pem()

    def check_key_rotation(self) -> bool:
        if not self.is_enabled():
            return False
        if self.key_manager.should_rotate_keys():
            logger.info("Performing key rotation")
            self.key_manager.generate_keys()
            if self.config.private_key_path and self.config.public_key_path:
                self.key_manager.save_keys()
            return True
        return False

    def add_trusted_agent(self, agent_id: str, public_key_pem: str) -> None:
        if self.is_enabled():
            self.trust_store.add_trusted_key(agent_id, public_key_pem)

    def is_agent_allowed(self, agent_id: str) -> bool:
        if not self.is_enabled():
            return True
        return self.trust_store.is_agent_trusted(agent_id)

