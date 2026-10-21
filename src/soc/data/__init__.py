"""Datasets, frozen splits and the answering prompt template."""

from soc.data.prompts import TEMPLATE_HASH, TEMPLATE_VERSION, messages
from soc.data.sources import SOURCES, Question, load_source
from soc.data.splits import SPECS, load_split

__all__ = [
    "SOURCES",
    "SPECS",
    "TEMPLATE_HASH",
    "TEMPLATE_VERSION",
    "Question",
    "load_source",
    "load_split",
    "messages",
]
