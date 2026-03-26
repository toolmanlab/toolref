"""Milvus connection manager and collection bootstrapper."""

import logging

from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from app.config import settings

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None

# ── Collection schemas ───────────────────────────────────────────────────────

CHILD_CHUNKS_COLLECTION = "toolref_child_chunks"
LONG_TERM_MEMORY_COLLECTION = "toolref_long_term_memory"


def _child_chunks_schema() -> CollectionSchema:
    """Schema for the child-chunks retrieval collection."""
    fields = [
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="parent_chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(
            name="dense_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.embedding_dim,
        ),
        FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    return CollectionSchema(fields=fields, description="Child chunks for precise retrieval")


def _long_term_memory_schema() -> CollectionSchema:
    """Schema for the long-term memory collection."""
    fields = [
        FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(
            name="dense_embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=settings.embedding_dim,
        ),
    ]
    return CollectionSchema(
        fields=fields, description="Long-term conversation memory summaries"
    )


# ── HNSW index parameters ────────────────────────────────────────────────────

HNSW_INDEX_PARAMS = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256},
}


# ── Connection helpers ────────────────────────────────────────────────────────


def connect_milvus() -> None:
    """Establish a connection to Milvus and ensure collections exist."""
    global _client
    logger.info("Connecting to Milvus at %s:%s", settings.milvus_host, settings.milvus_port)

    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )
    _client = MilvusClient(
        uri=f"http://{settings.milvus_host}:{settings.milvus_port}"
    )

    _ensure_collection(CHILD_CHUNKS_COLLECTION, _child_chunks_schema(), "dense_embedding")
    _ensure_collection(LONG_TERM_MEMORY_COLLECTION, _long_term_memory_schema(), "dense_embedding")
    logger.info("Milvus collections ready")


def _ensure_collection(name: str, schema: CollectionSchema, vector_field: str) -> None:
    """Create a collection with HNSW index if it does not exist."""
    if utility.has_collection(name):
        logger.info("Collection '%s' already exists", name)
        return

    from pymilvus import Collection

    collection = Collection(name=name, schema=schema)
    collection.create_index(field_name=vector_field, index_params=HNSW_INDEX_PARAMS)
    collection.load()
    logger.info("Created and loaded collection '%s'", name)


def disconnect_milvus() -> None:
    """Disconnect from Milvus."""
    global _client
    connections.disconnect(alias="default")
    _client = None
    logger.info("Milvus connection closed")


def check_milvus() -> bool:
    """Return True if Milvus is reachable."""
    try:
        connections.connect(
            alias="health_check",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        utility.list_collections(using="health_check")
        connections.disconnect(alias="health_check")
        return True
    except Exception:
        return False
