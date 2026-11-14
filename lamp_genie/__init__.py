"""Genie, the tool-calling agent of the LAMP materials database."""
from .api import ask, build_index, configure, embed_records, load

__all__ = ["ask", "build_index", "configure", "embed_records", "load"]
