"""KyroDB VectorDBBench client backed by the official KyroDB SDK."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from kyrodb import (
    InsertItem,
    KyroDBClient,
    KyroDBError,
    MetadataFilter,
    RetryPolicy,
    TLSConfig,
    exact,
    range_match,
)

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import IngestRPC, KyroDBIndexConfig

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

log = logging.getLogger(__name__)


class KyroDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]
    name = "KyroDB"
    thread_safe = True

    _ID_METADATA_FIELD = "id"
    _LABEL_METADATA_FIELD = "labels"
    _NAMESPACE_METADATA_FIELD = "__namespace__"
    _READY_COMPONENT = "cold_tier"
    _MAX_KYRO_DOC_ID = 0xFFFFFFFFFFFFFFFF
    _MAX_BENCHMARK_DOC_ID = _MAX_KYRO_DOC_ID - 1

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: KyroDBIndexConfig,
        collection_name: str = "KyroDBCollection",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ) -> None:
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.namespace = collection_name
        self.with_scalar_labels = with_scalar_labels
        self.search_parameter = self.case_config.search_param()
        self._query_filter: MetadataFilter | None = None
        self._client: KyroDBClient | None = None
        self._endpoint = f"{self.db_config['host']}:{self.db_config['port']}"

        log.info("KyroDB case config: %s", self.case_config.index_param())
        log.info("KyroDB search parameter: %s", self.search_parameter)

        with self._temporary_client() as client:
            client.wait_for_ready(timeout_s=self.db_config["connect_timeout"])
            self._ensure_server_reachable(client)
            self._validate_server_config(client)
            if drop_old:
                self._purge_namespace(client)

    @contextmanager
    def init(self):
        with self._make_sdk_client() as client:
            client.wait_for_ready(timeout_s=self.db_config["connect_timeout"])
            self._ensure_server_reachable(client)
            self._client = client
            try:
                yield
            finally:
                self._client = None

    def need_normalize_cosine(self) -> bool:
        return False

    def optimize(self, data_size: int | None = None):
        assert self._client is not None, "Please call self.init() before optimize()"
        self._ensure_server_reachable(self._client)

    def prepare_filter(self, filters: Filter):
        self._query_filter = self._translate_filter(filters)

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        assert self._client is not None, "Please call self.init() before insert_embeddings()"

        embeddings_list = list(embeddings)
        self._validate_insert_batch(embeddings_list, metadata, labels_data)

        try:
            if self.case_config.ingest_rpc == IngestRPC.BULK_LOAD_HNSW:
                return self._bulk_load_embeddings(embeddings_list, metadata, labels_data)

            if self.case_config.ingest_rpc == IngestRPC.BULK_INSERT:
                return self._bulk_insert_embeddings(embeddings_list, metadata, labels_data)

            return self._insert_embeddings_unary(embeddings_list, metadata, labels_data)
        except (KyroDBError, RuntimeError, TypeError, ValueError) as exc:
            return 0, exc

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
    ) -> list[int]:
        assert self._client is not None, "Please call self.init() before search_embedding()"

        response = self._client.search(
            query_embedding=query,
            k=k,
            namespace=self.namespace,
            include_embeddings=False,
            filter=self._query_filter,
            ef_search=self.search_parameter.get("ef_search", 0),
        )
        if response.error:
            raise RuntimeError(response.error)
        return [self._from_kyro_doc_id(hit.doc_id) for hit in response.results]

    @contextmanager
    def _temporary_client(self) -> Iterator[KyroDBClient]:
        with self._make_sdk_client() as client:
            yield client

    def _make_sdk_client(self) -> KyroDBClient:
        tls_config = self._build_tls_config()
        channel_options = [
            ("grpc.max_send_message_length", self.db_config["max_message_length_bytes"]),
            ("grpc.max_receive_message_length", self.db_config["max_message_length_bytes"]),
        ]
        return KyroDBClient(
            target=self._endpoint,
            api_key=self.db_config.get("api_key"),
            tls=tls_config,
            default_namespace=self.namespace,
            default_timeout_s=self.db_config["rpc_timeout"],
            retry_policy=RetryPolicy.no_retry(),
            channel_options=channel_options,
        )

    def _build_tls_config(self) -> TLSConfig | None:
        if not self.db_config["use_tls"]:
            return None

        root_certificates = self._read_tls_bytes(self.db_config.get("tls_root_cert_path"))
        certificate_chain = self._read_tls_bytes(self.db_config.get("tls_client_cert_path"))
        private_key = self._read_tls_bytes(self.db_config.get("tls_client_key_path"))
        return TLSConfig(
            root_certificates=root_certificates,
            certificate_chain=certificate_chain,
            private_key=private_key,
        )

    @staticmethod
    def _read_tls_bytes(path: str | None) -> bytes | None:
        if not path:
            return None
        return Path(path).read_bytes()

    def _validate_server_config(self, client: KyroDBClient) -> None:
        config = client.get_config(timeout_s=self.db_config["connect_timeout"])

        if config.embedding_dimension and config.embedding_dimension != self.dim:
            msg = (
                f"KyroDB server dimension mismatch: dataset dim={self.dim}, "
                f"server dim={config.embedding_dimension}"
            )
            raise RuntimeError(msg)

        expected_metric = self._normalize_metric_name(self.case_config.parse_metric())
        actual_metric = self._normalize_metric_name(config.hnsw_distance)
        if actual_metric and actual_metric != expected_metric:
            msg = (
                f"KyroDB server metric mismatch: case metric={expected_metric}, "
                f"server metric={actual_metric}"
            )
            raise RuntimeError(msg)

        if config.hnsw_m and config.hnsw_m != self.case_config.m:
            msg = f"KyroDB server m mismatch: case m={self.case_config.m}, server m={config.hnsw_m}"
            raise RuntimeError(msg)

        if config.hnsw_ef_construction and config.hnsw_ef_construction != self.case_config.ef_construct:
            msg = (
                "KyroDB server ef_construction mismatch: "
                f"case ef_construction={self.case_config.ef_construct}, "
                f"server ef_construction={config.hnsw_ef_construction}"
            )
            raise RuntimeError(msg)

    def _purge_namespace(self, client: KyroDBClient) -> None:
        if not self.namespace:
            raise RuntimeError("KyroDB benchmark drop_old requires a non-empty namespace")

        response = client.batch_delete_filter(
            filter=exact(self._NAMESPACE_METADATA_FIELD, self.namespace),
            namespace=self.namespace,
            timeout_s=self.db_config["connect_timeout"],
        )
        if not response.success:
            error = response.error or "batch_delete_filter namespace purge failed"
            raise RuntimeError(error)

    def _ensure_server_reachable(self, client: KyroDBClient) -> None:
        response = client.health(
            component=self._READY_COMPONENT,
            timeout_s=self.db_config["connect_timeout"],
        )
        if response.status != "HEALTHY":
            msg = (
                "KyroDB benchmark requires "
                f"{self._READY_COMPONENT} status HEALTHY, got {response.status}"
            )
            raise RuntimeError(msg)

    def _iter_insert_items(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> Iterator[InsertItem]:
        for index, (embedding, benchmark_doc_id) in enumerate(zip(embeddings, metadata, strict=True)):
            metadata_payload = {
                self._ID_METADATA_FIELD: str(int(benchmark_doc_id)),
            }
            if labels_data is not None:
                metadata_payload[self._LABEL_METADATA_FIELD] = labels_data[index]

            yield InsertItem.from_parts(
                doc_id=self._to_kyro_doc_id(int(benchmark_doc_id)),
                embedding=embedding,
                metadata=metadata_payload,
                namespace=self.namespace,
            )

    def _validate_insert_batch(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> None:
        if len(embeddings) != len(metadata):
            msg = f"Embeddings ({len(embeddings)}) and metadata ({len(metadata)}) length mismatch"
            raise ValueError(msg)
        if labels_data is not None and len(labels_data) != len(metadata):
            msg = f"labels_data ({len(labels_data)}) and metadata ({len(metadata)}) length mismatch"
            raise ValueError(msg)
        for benchmark_doc_id in metadata:
            self._to_kyro_doc_id(int(benchmark_doc_id))

    def _bulk_load_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> tuple[int, Exception | None]:
        assert self._client is not None
        response = self._client.bulk_load_hnsw(self._iter_insert_items(embeddings, metadata, labels_data))
        if not response.success or response.total_failed:
            error = response.error or "BulkLoadHnsw reported failures"
            return int(response.total_loaded), RuntimeError(error)
        return int(response.total_loaded), None

    def _bulk_insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> tuple[int, Exception | None]:
        assert self._client is not None
        response = self._client.bulk_insert(self._iter_insert_items(embeddings, metadata, labels_data))
        if not response.success or response.total_failed:
            error = response.error or "BulkInsert reported failures"
            return int(response.total_inserted), RuntimeError(error)
        return int(response.total_inserted), None

    def _insert_embeddings_unary(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None,
    ) -> tuple[int, Exception | None]:
        assert self._client is not None
        inserted = 0
        for item in self._iter_insert_items(embeddings, metadata, labels_data):
            response = self._client.insert(
                doc_id=item.doc_id,
                embedding=item.embedding,
                metadata=item.metadata,
                namespace=item.namespace,
            )
            if not response.success:
                error = response.error or "Insert reported failure"
                return inserted, RuntimeError(error)
            inserted += 1
        return inserted, None

    @classmethod
    def _to_kyro_doc_id(cls, benchmark_doc_id: int) -> int:
        if benchmark_doc_id < 0:
            msg = f"KyroDB benchmark adapter requires non-negative benchmark IDs, got {benchmark_doc_id}"
            raise ValueError(msg)
        if benchmark_doc_id > cls._MAX_BENCHMARK_DOC_ID:
            msg = (
                "KyroDB benchmark adapter cannot translate doc_id beyond uint64-1 after "
                f"positive-ID normalization, got {benchmark_doc_id}"
            )
            raise ValueError(msg)
        return benchmark_doc_id + 1

    @staticmethod
    def _from_kyro_doc_id(kyro_doc_id: int) -> int:
        if kyro_doc_id <= 0:
            msg = f"KyroDB returned invalid doc_id {kyro_doc_id}; expected positive uint64"
            raise RuntimeError(msg)
        return kyro_doc_id - 1

    @staticmethod
    def _normalize_metric_name(metric_name: str) -> str:
        normalized = metric_name.replace("-", "_").replace(" ", "_").lower()
        if normalized in {"ip", "dp"}:
            return "inner_product"
        return normalized

    @classmethod
    def _translate_filter(cls, filters: Filter) -> MetadataFilter | None:
        if filters.type == FilterOp.NonFilter:
            return None
        if filters.type == FilterOp.NumGE:
            return range_match(filters.int_field, gte=str(filters.int_value))
        if filters.type == FilterOp.StrEqual:
            return exact(filters.label_field, filters.label_value)
        msg = f"Unsupported filter for KyroDB: {filters.type}"
        raise ValueError(msg)
