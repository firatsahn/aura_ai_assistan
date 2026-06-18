"""Generation provider — turns retrieved chunks into a grounded answer.

The rest of the system calls `generate()` only through this module, so the LLM
provider can be swapped in one place (mirrors the embedding module).
"""

from .client import GENERATION_MODEL, generate

__all__ = ["generate", "GENERATION_MODEL"]
