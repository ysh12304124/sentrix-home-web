"""Multi-retriever package (Phase R §5.1).

Exposes the retriever factory used by the Application Composition Root.  Each
channel can be toggled individually via ``config.channel_enabled(name)``
(env ``SENTRIX_RETRIEVER_<NAME>`` or config defaults).
"""

from __future__ import annotations

from .base import CandidateHit, HardFilterContext, Retriever, RetrievalQuery
from .config import RetrievalConfig
from .fusion import fuse
from .near_duplicate import NearDuplicateGrouper
from .probes import NeutralProbe
from .ranking import rank

_RETRIEVER_FACTORIES = {}


def _register(name):
    def decorator(factory):
        _RETRIEVER_FACTORIES[name] = factory
        return factory
    return decorator


@_register("metadata")
def _metadata(store, **kwargs):
    from .metadata import MetadataRetriever
    return MetadataRetriever(store)


@_register("entity")
def _entity(store, **kwargs):
    from .entity import EntityRetriever
    return EntityRetriever(store)


@_register("lexical")
def _lexical(store, **kwargs):
    from .lexical import LexicalRetriever
    return LexicalRetriever(store)


@_register("visual_ann")
def _visual_ann(store, **kwargs):
    from .visual_ann import VisualAnnRetriever
    return VisualAnnRetriever(store, embedding_router=kwargs.get("embedding_router"),
                              ann_dir=kwargs.get("ann_dir"))


@_register("text_ann")
def _text_ann(store, **kwargs):
    from .text_ann import TextAnnRetriever
    return TextAnnRetriever(store, embedding_router=kwargs.get("embedding_router"),
                            ann_dir=kwargs.get("ann_dir"))


@_register("adjacency")
def _adjacency(store, **kwargs):
    from .adjacency import AdjacencyRetriever
    return AdjacencyRetriever(store, config=kwargs.get("config"))


def build_default_retrievers(store, *, embedding_router=None, config=None, ann_dir=None):
    """Instantiate the enabled retriever set for a Kernel.

    Adjacency is built but the Kernel gates it behind the seed-quality pass
    (R3B); when disabled it simply contributes nothing.
    """
    config = config or RetrievalConfig()
    # config keys use short names (visual/text) while the retriever factories
    # use the concrete names (visual_ann/text_ann) — map them explicitly.
    retriever_config_key = {
        "metadata": "metadata", "entity": "entity", "lexical": "lexical",
        "visual_ann": "visual", "text_ann": "text", "adjacency": "adjacency",
    }
    retrievers = []
    for name in ("metadata", "entity", "lexical", "visual_ann", "text_ann", "adjacency"):
        if not config.channel_enabled(retriever_config_key[name]):
            continue
        factory = _RETRIEVER_FACTORIES[name]
        retrievers.append(factory(store, embedding_router=embedding_router,
                                  config=config, ann_dir=ann_dir))
    return retrievers


__all__ = [
    "CandidateHit", "HardFilterContext", "Retriever", "RetrievalQuery",
    "RetrievalConfig", "fuse", "NeutralProbe", "NearDuplicateGrouper", "rank",
    "build_default_retrievers",
]
