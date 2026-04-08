from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Self

import pytest
from click.testing import CliRunner
from kyrodb import (
    BatchDeleteResult,
    BulkLoadResult,
    InsertAck,
    RetryPolicy,
    SearchHit,
    SearchResponse,
    TLSConfig,
)

from vectordb_bench.backend.cases import CaseLabel, CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import MetricType
from vectordb_bench.backend.clients.kyrodb import kyrodb as kyrodb_module
from vectordb_bench.backend.clients.kyrodb.cli import KyroDB as KyroDBCli
from vectordb_bench.backend.clients.kyrodb.config import IngestRPC, KyroDBConfig, KyroDBIndexConfig
from vectordb_bench.backend.clients.kyrodb.kyrodb import KyroDB
from vectordb_bench.backend.filter import FilterOp, IntFilter, LabelFilter, non_filter
from vectordb_bench.frontend.components.run_test.generateTasks import generate_tasks
from vectordb_bench.frontend.config.dbCaseConfigs import get_case_config_inputs
from vectordb_bench.models import CaseConfig, CaseConfigParamType

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch
    from kyrodb import InsertItem


def build_client(
    *,
    ingest_rpc: IngestRPC = IngestRPC.BULK_LOAD_HNSW,
    namespace: str = "bench_ns",
    filter_type: FilterOp = FilterOp.NonFilter,
) -> KyroDB:
    client = KyroDB.__new__(KyroDB)
    client.dim = 768
    client.db_config = {
        "host": "127.0.0.1",
        "port": 51051,
        "api_key": "bench-key",
        "use_tls": False,
        "tls_root_cert_path": None,
        "tls_client_cert_path": None,
        "tls_client_key_path": None,
        "connect_timeout": 15.0,
        "rpc_timeout": 30.0,
        "max_message_length_bytes": 64 * 1024 * 1024,
    }
    client.case_config = KyroDBIndexConfig(
        metric_type=MetricType.COSINE,
        m=16,
        ef_construct=200,
        ef_search=96,
        ingest_rpc=ingest_rpc,
    )
    client.collection_name = namespace
    client.namespace = namespace
    client.with_scalar_labels = True
    client.filter_type = filter_type
    client.search_parameter = client.case_config.search_param()
    client._query_filter = None
    client._client = None
    client._endpoint = "127.0.0.1:51051"
    return client


class SearchClient:
    def __init__(self) -> None:
        self.search_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return SearchResponse(
            results=(
                SearchHit(doc_id=42, score=0.99),
                SearchHit(doc_id=43, score=0.98),
            ),
            total_found=2,
            search_latency_ms=1.2,
            search_path="COLD_TIER_ONLY",
        )


class BulkLoadClient:
    def __init__(self) -> None:
        self.bulk_load_items = None

    def bulk_load_hnsw(self, items: Iterable[InsertItem]):
        self.bulk_load_items = list(items)
        return BulkLoadResult(
            success=True,
            total_loaded=len(self.bulk_load_items),
            total_failed=0,
        )


class BulkInsertClient:
    def __init__(self) -> None:
        self.bulk_insert_items = None

    def bulk_insert(self, items: Iterable[InsertItem]):
        self.bulk_insert_items = list(items)
        return InsertAck(
            success=True,
            inserted_at=123,
            total_inserted=len(self.bulk_insert_items),
            total_failed=0,
            tier="HOT_TIER",
        )


class UnaryInsertClient:
    def __init__(self) -> None:
        self.insert_calls: list[dict] = []

    def insert(self, **kwargs):
        self.insert_calls.append(kwargs)
        return InsertAck(
            success=True,
            inserted_at=456,
            total_inserted=len(self.insert_calls),
            total_failed=0,
            tier="HOT_TIER",
        )


class RecordingSDKClient:
    instances: list[RecordingSDKClient] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.wait_calls: list[float | None] = []
        self.health_calls: list[dict] = []
        self.batch_delete_filter_calls: list[dict] = []
        self.closed = False
        self.config = SimpleNamespace(
            embedding_dimension=768,
            hnsw_distance="cosine",
            hnsw_m=16,
            hnsw_ef_construction=200,
        )
        RecordingSDKClient.instances.append(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.closed = True

    def wait_for_ready(self, *, timeout_s: float | None = None) -> None:
        self.wait_calls.append(timeout_s)

    def health(self, *, component: str = "", timeout_s: float | None = None):
        self.health_calls.append({"component": component, "timeout_s": timeout_s})
        return SimpleNamespace(status="HEALTHY")

    def get_config(self, *, timeout_s: float | None = None):
        return self.config

    def batch_delete_filter(self, **kwargs):
        self.batch_delete_filter_calls.append(kwargs)
        return BatchDeleteResult(success=True, deleted_count=3)


def test_kyrodb_config_to_dict_serializes_optional_secret_and_tls_paths():
    config = KyroDBConfig(
        host="127.0.0.1",
        port=51051,
        api_key="bench-key",
        use_tls=True,
        tls_root_cert_path="/tmp/ca.pem",
        tls_client_cert_path="/tmp/client.crt",
        tls_client_key_path="/tmp/client.key",
    )

    payload = config.to_dict()
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 51051
    assert payload["api_key"] == "bench-key"
    assert payload["tls_root_cert_path"] == "/tmp/ca.pem"
    assert payload["tls_client_cert_path"] == "/tmp/client.crt"
    assert payload["tls_client_key_path"] == "/tmp/client.key"


def test_registry_exposes_kyrodb_backend():
    assert DB.KyroDB.value == "KyroDB"
    assert DB.KyroDB.init_cls.__name__ == "KyroDB"
    assert DB.KyroDB.config_cls.__name__ == "KyroDBConfig"
    assert DB.KyroDB.case_config_cls().__name__ == "KyroDBIndexConfig"


def test_kyrodb_index_config_accepts_frontend_ef_construction_alias():
    config = KyroDBIndexConfig(
        m=16,
        ef_construction=200,
    )

    assert config.ef_construct == 200


def test_parse_metric_requires_supported_metric_type():
    missing_metric = KyroDBIndexConfig(metric_type=None, m=16, ef_construction=200)
    unsupported_metric = KyroDBIndexConfig(metric_type=MetricType.HAMMING, m=16, ef_construction=200)

    with pytest.raises(ValueError, match="metric_type must be set"):
        missing_metric.parse_metric()

    with pytest.raises(ValueError, match="Unsupported metric_type"):
        unsupported_metric.parse_metric()


def test_prepare_filter_translates_supported_filters():
    client = build_client()

    client.prepare_filter(non_filter)
    assert client._query_filter is None

    client.prepare_filter(IntFilter(filter_rate=0.01, int_value=123))
    int_filter = client._query_filter
    assert int_filter is not None
    int_proto = int_filter.to_proto()
    assert int_proto.range.key == "id"
    assert int_proto.range.gte == "123"

    client.prepare_filter(LabelFilter(label_percentage=0.05))
    label_filter = client._query_filter
    assert label_filter is not None
    label_proto = label_filter.to_proto()
    assert label_proto.exact.key == "labels"
    assert label_proto.exact.value == "label_5p"


def test_search_embedding_uses_sdk_search_and_translates_doc_ids_back_to_benchmark_space():
    client = build_client()
    sdk_client = SearchClient()
    client._client = sdk_client
    client.prepare_filter(LabelFilter(label_percentage=0.05))

    results = client.search_embedding([0.1, 0.2, 0.3], k=25)

    assert results == [41, 42]
    assert len(sdk_client.search_calls) == 1
    search_call = sdk_client.search_calls[0]
    assert search_call["namespace"] == "bench_ns"
    assert search_call["k"] == 25
    assert search_call["ef_search"] == 96
    filter_proto = search_call["filter"].to_proto()
    assert filter_proto.exact.key == "labels"
    assert filter_proto.exact.value == "label_5p"


def test_insert_embeddings_bulk_load_uses_sdk_items_and_positive_wire_doc_ids():
    client = build_client(ingest_rpc=IngestRPC.BULK_LOAD_HNSW)
    sdk_client = BulkLoadClient()
    client._client = sdk_client

    inserted, error = client.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[7, 8],
        labels_data=["label_1p", "label_5p"],
    )

    assert inserted == 2
    assert error is None
    assert sdk_client.bulk_load_items is not None
    assert [item.doc_id for item in sdk_client.bulk_load_items] == [8, 9]
    assert all(item.namespace == "bench_ns" for item in sdk_client.bulk_load_items)
    assert sdk_client.bulk_load_items[0].metadata == {}
    assert sdk_client.bulk_load_items[1].metadata == {}


def test_insert_embeddings_bulk_insert_uses_sdk_streaming_path():
    client = build_client(ingest_rpc=IngestRPC.BULK_INSERT)
    sdk_client = BulkInsertClient()
    client._client = sdk_client

    inserted, error = client.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[0, 3],
    )

    assert inserted == 2
    assert error is None
    assert sdk_client.bulk_insert_items is not None
    assert [item.doc_id for item in sdk_client.bulk_insert_items] == [1, 4]
    assert sdk_client.bulk_insert_items[0].metadata == {}
    assert sdk_client.bulk_insert_items[1].metadata == {}


def test_insert_embeddings_unary_path_uses_positive_wire_ids():
    client = build_client(ingest_rpc=IngestRPC.INSERT)
    sdk_client = UnaryInsertClient()
    client._client = sdk_client

    inserted, error = client.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[5, 6],
    )

    assert inserted == 2
    assert error is None
    assert [call["doc_id"] for call in sdk_client.insert_calls] == [6, 7]
    assert all(call["namespace"] == "bench_ns" for call in sdk_client.insert_calls)
    assert sdk_client.insert_calls[0]["metadata"] == {}
    assert sdk_client.insert_calls[1]["metadata"] == {}


def test_insert_embeddings_numeric_filter_cases_only_emit_id_metadata():
    client = build_client(filter_type=FilterOp.NumGE)
    sdk_client = BulkLoadClient()
    client._client = sdk_client

    inserted, error = client.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[7, 8],
    )

    assert inserted == 2
    assert error is None
    assert sdk_client.bulk_load_items is not None
    assert sdk_client.bulk_load_items[0].metadata == {"id": "7"}
    assert sdk_client.bulk_load_items[1].metadata == {"id": "8"}


def test_insert_embeddings_label_filter_cases_only_emit_label_metadata():
    client = build_client(filter_type=FilterOp.StrEqual)
    sdk_client = BulkLoadClient()
    client._client = sdk_client

    inserted, error = client.insert_embeddings(
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadata=[7, 8],
        labels_data=["label_1p", "label_5p"],
    )

    assert inserted == 2
    assert error is None
    assert sdk_client.bulk_load_items is not None
    assert sdk_client.bulk_load_items[0].metadata == {"labels": "label_1p"}
    assert sdk_client.bulk_load_items[1].metadata == {"labels": "label_5p"}


def test_insert_embeddings_label_filter_cases_require_label_metadata():
    client = build_client(filter_type=FilterOp.StrEqual)
    client._client = BulkLoadClient()

    with pytest.raises(ValueError, match="labels_data is required"):
        client.insert_embeddings(
            embeddings=[[0.1, 0.2]],
            metadata=[7],
            labels_data=None,
        )


def test_adapter_rejects_negative_benchmark_doc_ids():
    client = build_client()

    with pytest.raises(ValueError, match="non-negative benchmark IDs"):
        client._to_kyro_doc_id(-1)


def test_validate_server_config_rejects_mismatched_server_index_parameters():
    client = build_client()

    def get_config(*, timeout_s: float | None = None) -> SimpleNamespace:
        _ = timeout_s
        return SimpleNamespace(
            embedding_dimension=768,
            hnsw_distance="cosine",
            hnsw_m=24,
            hnsw_ef_construction=200,
        )

    sdk_client = SimpleNamespace(
        get_config=get_config
    )

    with pytest.raises(RuntimeError, match="m mismatch"):
        client._validate_server_config(sdk_client)


def test_constructor_rejects_non_healthy_server_status(monkeypatch: MonkeyPatch):
    class DegradedSDKClient(RecordingSDKClient):
        def health(self, *, component: str = "", timeout_s: float | None = None):
            self.health_calls.append({"component": component, "timeout_s": timeout_s})
            return SimpleNamespace(status="DEGRADED")

    RecordingSDKClient.instances.clear()
    monkeypatch.setattr(kyrodb_module, "KyroDBClient", DegradedSDKClient)

    with pytest.raises(RuntimeError, match="status HEALTHY, got DEGRADED"):
        KyroDB(
            dim=768,
            db_config={
                "host": "127.0.0.1",
                "port": 51051,
                "api_key": None,
                "use_tls": False,
                "tls_root_cert_path": None,
                "tls_client_cert_path": None,
                "tls_client_key_path": None,
                "connect_timeout": 9.0,
                "rpc_timeout": 30.0,
                "max_message_length_bytes": 64 * 1024 * 1024,
            },
            db_case_config=KyroDBIndexConfig(metric_type=MetricType.COSINE, m=16, ef_construction=200, ef_search=128),
            collection_name="bench_ns",
        )


def test_constructor_builds_no_retry_sdk_client_and_uses_tls_material(tmp_path: Path, monkeypatch: MonkeyPatch):
    RecordingSDKClient.instances.clear()
    root_cert = tmp_path / "ca.pem"
    client_cert = tmp_path / "client.crt"
    client_key = tmp_path / "client.key"
    root_cert.write_bytes(b"ca")
    client_cert.write_bytes(b"cert")
    client_key.write_bytes(b"key")

    monkeypatch.setattr(kyrodb_module, "KyroDBClient", RecordingSDKClient)

    client = KyroDB(
        dim=768,
        db_config={
            "host": "127.0.0.1",
            "port": 51051,
            "api_key": "bench-key",
            "use_tls": True,
            "tls_root_cert_path": str(root_cert),
            "tls_client_cert_path": str(client_cert),
            "tls_client_key_path": str(client_key),
            "connect_timeout": 9.0,
            "rpc_timeout": 30.0,
            "max_message_length_bytes": 64 * 1024 * 1024,
        },
        db_case_config=KyroDBIndexConfig(metric_type=MetricType.COSINE, m=16, ef_construct=200, ef_search=128),
        collection_name="bench_ns",
        drop_old=True,
    )

    assert client.namespace == "bench_ns"
    assert len(RecordingSDKClient.instances) == 1
    sdk_client = RecordingSDKClient.instances[0]
    assert sdk_client.kwargs["target"] == "127.0.0.1:51051"
    assert sdk_client.kwargs["api_key"] == "bench-key"
    assert sdk_client.kwargs["default_namespace"] == "bench_ns"
    assert sdk_client.kwargs["default_timeout_s"] == 30.0
    assert sdk_client.kwargs["retry_policy"] == RetryPolicy.no_retry()
    assert sdk_client.kwargs["channel_options"] == [
        ("grpc.max_send_message_length", 64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ]
    assert isinstance(sdk_client.kwargs["tls"], TLSConfig)
    assert sdk_client.kwargs["tls"].root_certificates == b"ca"
    assert sdk_client.kwargs["tls"].certificate_chain == b"cert"
    assert sdk_client.kwargs["tls"].private_key == b"key"
    assert sdk_client.wait_calls == [9.0]
    assert sdk_client.health_calls[0]["component"] == "cold_tier"
    delete_call = sdk_client.batch_delete_filter_calls[0]
    assert delete_call["namespace"] == "bench_ns"
    assert delete_call["timeout_s"] == 9.0
    delete_filter_proto = delete_call["filter"].to_proto()
    assert delete_filter_proto.exact.key == "__namespace__"
    assert delete_filter_proto.exact.value == "bench_ns"
    assert sdk_client.closed is True


def test_kyrodb_cli_rejects_partial_mtls_config(tmp_path: Path):
    runner = CliRunner()
    client_cert = tmp_path / "client.crt"
    client_cert.write_text("cert")
    result = runner.invoke(
        KyroDBCli,
        [
            "--host",
            "127.0.0.1",
            "--port",
            "51051",
            "--m",
            "16",
            "--ef-construction",
            "200",
            "--ef-search",
            "128",
            "--tls-client-cert-path",
            str(client_cert),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "must be provided together for mTLS" in result.output


def test_kyrodb_cli_rejects_tls_material_without_use_tls(tmp_path: Path):
    runner = CliRunner()
    root_cert = tmp_path / "ca.pem"
    root_cert.write_text("ca")
    result = runner.invoke(
        KyroDBCli,
        [
            "--host",
            "127.0.0.1",
            "--port",
            "51051",
            "--m",
            "16",
            "--ef-construction",
            "200",
            "--ef-search",
            "128",
            "--tls-root-cert-path",
            str(root_cert),
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "--use-tls is required" in result.output


def test_kyrodb_cli_dry_run_builds_task():
    runner = CliRunner()
    result = runner.invoke(
        KyroDBCli,
        [
            "--host",
            "127.0.0.1",
            "--port",
            "51051",
            "--m",
            "16",
            "--ef-construction",
            "200",
            "--ef-search",
            "128",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output


def test_kyrodb_frontend_case_configs_use_honest_load_and_streaming_defaults():
    load_inputs = {item.label: item for item in get_case_config_inputs(DB.KyroDB, CaseLabel.Performance)}
    streaming_inputs = {item.label: item for item in get_case_config_inputs(DB.KyroDB, CaseLabel.Streaming)}

    assert load_inputs[CaseConfigParamType.m].inputConfig["value"] == 16
    assert load_inputs[CaseConfigParamType.ef_construction].inputConfig["value"] == 200
    assert load_inputs[CaseConfigParamType.ef_search].inputConfig["value"] == 128
    assert load_inputs[CaseConfigParamType.ingest_rpc].inputConfig["value"] == "bulk_load_hnsw"
    assert streaming_inputs[CaseConfigParamType.ingest_rpc].inputConfig["value"] == "bulk_insert"


def test_generate_tasks_builds_kyrodb_streaming_case_config_from_frontend_defaults():
    streaming_case = CaseConfig(case_id=CaseType.StreamingPerformanceCase)
    streaming_defaults = {
        item.label: item.inputConfig["value"]
        for item in get_case_config_inputs(DB.KyroDB, CaseLabel.Streaming)
        if "value" in item.inputConfig
    }

    tasks = generate_tasks(
        activedDbList=[DB.KyroDB],
        dbConfigs={
            DB.KyroDB: KyroDBConfig(
                host="127.0.0.1",
                port=51051,
            )
        },
        activedCaseList=[streaming_case],
        allCaseConfigs={DB.KyroDB: {streaming_case: streaming_defaults}},
    )

    assert len(tasks) == 1
    assert isinstance(tasks[0].db_case_config, KyroDBIndexConfig)
    assert tasks[0].db_case_config.ingest_rpc == IngestRPC.BULK_INSERT
    assert tasks[0].db_case_config.m == 16
    assert tasks[0].db_case_config.ef_construct == 200
    assert tasks[0].db_case_config.ef_search == 128
