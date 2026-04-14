"""RSA-PSS request signing for Kalshi API v2."""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class KalshiAuth:
    """Generate authentication headers for Kalshi API requests.

    Kalshi uses RSA-PSS SHA256 to sign: timestamp_ms + method + path
    """

    def __init__(self, api_key_id: str, private_key_path: str):
        self._api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)

    def sign_request(self, method: str, path: str) -> dict[str, str]:
        """Generate auth headers for a request.

        Returns:
            Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE,
            KALSHI-ACCESS-TIMESTAMP headers.
        """
        timestamp_ms = str(int(time.time() * 1000))
        message = f"{timestamp_ms}{method.upper()}{path}"

        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )

        sig_b64 = base64.b64encode(signature).decode("utf-8")

        return {
            "KALSHI-ACCESS-KEY": self._api_key_id,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }

    @staticmethod
    def _load_private_key(path: str) -> rsa.RSAPrivateKey:
        key_path = Path(path).expanduser()
        key_data = key_path.read_bytes()

        private_key = serialization.load_pem_private_key(key_data, password=None)
        if not isinstance(private_key, rsa.RSAPrivateKey):
            raise TypeError(f"Expected RSA private key, got {type(private_key)}")
        return private_key
