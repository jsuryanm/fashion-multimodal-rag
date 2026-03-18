import base64
import logging
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(source: str | Path) -> Image.Image:
    """
    Load a PIL Image from a URL or local file path.

    Args:
        source: HTTP/HTTPS URL or local filesystem path.

    Returns:
        RGB PIL Image.

    Raises:
        FileNotFoundError: If a local path does not exist.
        requests.HTTPError: If a URL request fails.
    """
    source = str(source)

    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=15)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        image = Image.open(path)

    return image.convert("RGB")


def image_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """
    Encode a PIL Image to a base64 string.

    Args:
        image: RGB PIL Image.
        fmt:   "JPEG" (smaller, lossy) or "PNG" (lossless).

    Returns:
        Base64-encoded string without data URI prefix.
    """
    buffer = BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pil_to_data_uri(image: Image.Image, fmt: str = "JPEG") -> str:
    """
    Convert a PIL Image to a data URI suitable for vision LLM APIs.

    Returns:
        String in the form "data:image/jpeg;base64,<encoded>"
    """
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{image_to_base64(image, fmt)}"