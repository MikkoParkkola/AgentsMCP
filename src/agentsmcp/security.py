"""
Security utilities for the AgentsMCP discovery protocol (AD5).

Provides digital signatures, key management, trust store, and signature validation
following the Agent Discovery Protocol specification.
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

from .config import Config

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass


class KeyValidationError(SecurityError):
    """Raised when key validation fails."""
    pass


class SignatureValidationError(SecurityError):
    """Raised when signature validation fails."""
    pass


class TrustStoreError(SecurityError):
    """Raised when trust store operations fail."""
    pass


class KeyManager:
    """
    Manages RSA key pairs for agent identity and message signing.
    
    Handles key generation, loading, rotation, and provides signing/verification.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.private_key: Optional[rsa.RSAPrivateKey] = None
        self.public_key: Optional[rsa.RSAPublicKey] = None
        self.key_loaded_time: Optional[float] = None
        
    def load_or_generate_keys(self) -> None:
        """Load existing keys or generate new ones if not found."""
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
        """Generate a new RSA key pair."""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.key_loaded_time = time.time()
        logger.info(f"Generated {key_size}-bit RSA key pair")
    
    def load_keys(self) -> None:
        """Load keys from configured file paths."""
        if not self.config.private_key_path or not self.config.public_key_path:
            raise KeyValidationError("Key paths not configured")
        
        private_path = Path(self.config.private_key_path)
        public_path = Path(self.config.public_key_path)
        
        if not private_path.exists():
            raise FileNotFoundError(f"Private key not found: {private_path}")
        if not public_path.exists():
            raise FileNotFoundError(f"Public key not found: {public_path}")
        
        # Load private key
        with open(private_path, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        
        # Load public key
        with open(public_path, "rb") as f:
            self.public_key = serialization.load_pem_public_key(
                f.read(), backend=default_backend()
            )
        
        self.key_loaded_time = time.time()
        
        # Validate that the keys match
        if not self._validate_key_pair():
            raise KeyValidationError("Private and public keys do not match")
    
    def save_keys(self) -> None:
        """Save keys to configured file paths."""
        if not self.private_key or not self.public_key:
            raise KeyValidationError("No keys loaded to save")
        
        if not self.config.private_key_path or not self.config.public_key_path:
            raise KeyValidationError("Key paths not configured")
        
        private_path = Path(self.config.private_key_path)
        public_path = Path(self.config.public_key_path)
        
        # Create directories if they don't exist
        private_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save private key
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        with open(private_path, "wb") as f:
            f.write(private_pem)
        os.chmod(private_path, 0o600)  # Restrict access to private key
        
        # Save public key
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(public_path, "wb") as f:
            f.write(public_pem)
        
        logger.info(f"Saved keys to {private_path} and {public_path}")
    
    def _validate_key_pair(self) -> bool:
        """Validate that private and public keys match."""
        if not self.private_key or not self.public_key:
            return False
        
        # Test sign/verify operation
        test_data = b"test message for key validation"
        try:
            signature = self.sign_data(test_data)
            return self.verify_signature(test_data, signature, self.public_key)
        except Exception:
            return False
    
    def should_rotate_keys(self) -> bool:
        """Check if keys should be rotated based on age."""
        if not self.key_loaded_time:
            return True
        
        age_hours = (time.time() - self.key_loaded_time) / 3600
        return age_hours >= self.config.key_rotation_interval_hours
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data using the private key."""
        if not self.private_key:
            raise KeyValidationError("No private key loaded")
        
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes, public_key: rsa.RSAPublicKey) -> bool:
        """Verify a signature against data using a public key."""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False
    
    def get_public_key_pem(self) -> str:
        """Get the public key in PEM format as a string."""
        if not self.public_key:
            raise KeyValidationError("No public key loaded")
        
        pem_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        return pem_bytes.decode('utf-8')


class MessageSigner:
    """
    Signs and validates agent discovery protocol messages.
    
    Implements canonical JSON signing as specified in the protocol.
    """
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def sign_agent_message(self, agent_dict: Dict[str, Any]) -> str:
        """
        Sign an agent dictionary and return Base64-encoded signature.
        
        Uses canonical JSON (RFC 8785) for consistent signing.
        """
        # Create canonical JSON representation
        canonical_json = json.dumps(agent_dict, separators=(',', ':'), sort_keys=True)
        canonical_bytes = canonical_json.encode('utf-8')
        
        # Sign the canonical bytes
        signature_bytes = self.key_manager.sign_data(canonical_bytes)
        
        # Return Base64-encoded signature
        return base64.b64encode(signature_bytes).decode('utf-8')
    
    def verify_agent_message(
        self, 
        agent_dict: Dict[str, Any], 
        signature_b64: str, 
        public_key_pem: str
    ) -> bool:
        """
        Verify an agent message signature.
        
        Returns True if signature is valid, False otherwise.
        """
        try:
            # Parse public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=default_backend()
            )
            
            # Create canonical JSON representation
            canonical_json = json.dumps(agent_dict, separators=(',', ':'), sort_keys=True)
            canonical_bytes = canonical_json.encode('utf-8')
            
            # Decode signature
            signature_bytes = base64.b64decode(signature_b64)
            
            # Verify signature
            return self.key_manager.verify_signature(canonical_bytes, signature_bytes, public_key)
            
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False


class TrustStore:
    """
    Manages trusted public keys for agent verification.
    
    Supports allowlist enforcement and trusted key storage.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.trusted_keys: Dict[str, rsa.RSAPublicKey] = {}  # agent_id -> public_key
        self.allowlist: Set[str] = set(config.discovery_allowlist)
        self.trust_store_path = Path(config.trust_store_path) if config.trust_store_path else None
        
        if self.trust_store_path:
            self.load_trusted_keys()
    
    def load_trusted_keys(self) -> None:
        """Load trusted public keys from the trust store directory."""
        if not self.trust_store_path or not self.trust_store_path.exists():
            logger.info("Trust store path not configured or doesn't exist")
            return
        
        for key_file in self.trust_store_path.glob("*.pem"):
            try:
                agent_id = key_file.stem  # filename without extension
                with open(key_file, "rb") as f:
                    public_key = serialization.load_pem_public_key(
                        f.read(), backend=default_backend()
                    )
                self.trusted_keys[agent_id] = public_key
                logger.debug(f"Loaded trusted key for agent {agent_id}")
            except Exception as e:
                logger.warning(f"Failed to load trusted key {key_file}: {e}")
        
        logger.info(f"Loaded {len(self.trusted_keys)} trusted keys from {self.trust_store_path}")
    
    def add_trusted_key(self, agent_id: str, public_key_pem: str) -> None:
        """Add a trusted public key for an agent."""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode('utf-8'),
                backend=default_backend()
            )
            self.trusted_keys[agent_id] = public_key
            
            # Save to trust store if path configured
            if self.trust_store_path:
                self.trust_store_path.mkdir(parents=True, exist_ok=True)
                key_file = self.trust_store_path / f"{agent_id}.pem"
                with open(key_file, "w") as f:
                    f.write(public_key_pem)
                logger.info(f"Added trusted key for agent {agent_id}")
            
        except Exception as e:
            raise TrustStoreError(f"Failed to add trusted key for {agent_id}: {e}")
    
    def is_agent_trusted(self, agent_id: str) -> bool:
        """Check if an agent is trusted (has key in trust store or is in allowlist)."""
        # Check allowlist first
        if self.allowlist and agent_id not in self.allowlist:
            return False
        
        # If no keys loaded, allow if in allowlist or allowlist is empty
        if not self.trusted_keys:
            return not self.allowlist or agent_id in self.allowlist
        
        # Check if we have a trusted key
        return agent_id in self.trusted_keys
    
    def get_trusted_key(self, agent_id: str) -> Optional[rsa.RSAPublicKey]:
        """Get the trusted public key for an agent."""
        return self.trusted_keys.get(agent_id)
    
    def verify_agent_signature(
        self, 
        agent_id: str, 
        agent_dict: Dict[str, Any], 
        signature_b64: str
    ) -> bool:
        """
        Verify an agent message signature using trust store.
        
        Returns True if signature is valid and agent is trusted.
        """
        if not self.is_agent_trusted(agent_id):
            logger.warning(f"Agent {agent_id} is not trusted")
            return False
        
        trusted_key = self.get_trusted_key(agent_id)
        if not trusted_key:
            # Agent is in allowlist but no key stored - accept if signature matches embedded key
            embedded_key_pem = agent_dict.get("public_key")
            if not embedded_key_pem:
                logger.warning(f"No public key in message from agent {agent_id}")
                return False
            
            try:
                # Use embedded key for verification (bootstrap case)
                public_key = serialization.load_pem_public_key(
                    embedded_key_pem.encode('utf-8'),
                    backend=default_backend()
                )
                
                # Create canonical JSON and verify
                canonical_json = json.dumps(agent_dict, separators=(',', ':'), sort_keys=True)
                signature_bytes = base64.b64decode(signature_b64)
                
                key_manager = KeyManager(self.config)  # Temporary instance for verification
                result = key_manager.verify_signature(canonical_json.encode('utf-8'), signature_bytes, public_key)
                
                # Add to trust store if verification succeeds
                if result:
                    self.add_trusted_key(agent_id, embedded_key_pem)
                
                return result
                
            except Exception as e:
                logger.warning(f"Failed to verify embedded key for agent {agent_id}: {e}")
                return False
        
        # Use stored trusted key for verification
        try:
            canonical_json = json.dumps(agent_dict, separators=(',', ':'), sort_keys=True)
            signature_bytes = base64.b64decode(signature_b64)
            
            key_manager = KeyManager(self.config)  # Temporary instance for verification
            return key_manager.verify_signature(canonical_json.encode('utf-8'), signature_bytes, trusted_key)
            
        except Exception as e:
            logger.warning(f"Signature verification failed for agent {agent_id}: {e}")
            return False


class SecurityManager:
    """
    High-level security manager that coordinates key management, signing, and trust store.
    
    Provides a unified interface for all AD5 security features.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.key_manager = KeyManager(config)
        self.message_signer = MessageSigner(self.key_manager)
        self.trust_store = TrustStore(config)
        
        # Initialize keys if security is enabled
        if config.security_enabled:
            self.key_manager.load_or_generate_keys()
    
    def is_enabled(self) -> bool:
        """Check if security features are enabled."""
        return self.config.security_enabled
    
    def sign_announcement(self, agent_dict: Dict[str, Any]) -> Optional[str]:
        """
        Sign an agent announcement message.
        
        Returns Base64-encoded signature if security is enabled, None otherwise.
        """
        if not self.is_enabled():
            return None
        
        try:
            return self.message_signer.sign_agent_message(agent_dict)
        except Exception as e:
            logger.error(f"Failed to sign announcement: {e}")
            raise SignatureValidationError(f"Signature creation failed: {e}")
    
    def validate_announcement(
        self, 
        agent_id: str, 
        agent_dict: Dict[str, Any], 
        signature_b64: Optional[str]
    ) -> bool:
        """
        Validate an incoming agent announcement.
        
        Returns True if valid (or security disabled), False otherwise.
        """
        if not self.is_enabled():
            return True  # Security disabled, accept all
        
        if not signature_b64:
            logger.warning(f"Missing signature in announcement from {agent_id}")
            return False
        
        return self.trust_store.verify_agent_signature(agent_id, agent_dict, signature_b64)
    
    def get_public_key_pem(self) -> Optional[str]:
        """Get local agent's public key in PEM format."""
        if not self.is_enabled():
            return None
        
        return self.key_manager.get_public_key_pem()
    
    def check_key_rotation(self) -> bool:
        """Check if keys need rotation and perform if necessary."""
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
        """Add an agent to the trust store."""
        if self.is_enabled():
            self.trust_store.add_trusted_key(agent_id, public_key_pem)
    
    def is_agent_allowed(self, agent_id: str) -> bool:
        """Check if agent is allowed by allowlist and trust policies."""
        if not self.is_enabled():
            return True
        
        return self.trust_store.is_agent_trusted(agent_id)