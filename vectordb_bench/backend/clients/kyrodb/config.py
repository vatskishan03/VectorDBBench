from enum import StrEnum

from pydantic import BaseModel, Field, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class IngestRPC(StrEnum):
    BULK_LOAD_HNSW = "bulk_load_hnsw"
    BULK_INSERT = "bulk_insert"
    INSERT = "insert"


class KyroDBConfig(DBConfig):
    host: str
    port: int
    api_key: SecretStr | None = None
    use_tls: bool = False
    tls_root_cert_path: str | None = None
    tls_client_cert_path: str | None = None
    tls_client_key_path: str | None = None
    connect_timeout: float = 15.0
    rpc_timeout: float | None = None
    max_message_length_bytes: int = 64 * 1024 * 1024

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "use_tls": self.use_tls,
            "tls_root_cert_path": self.tls_root_cert_path,
            "tls_client_cert_path": self.tls_client_cert_path,
            "tls_client_key_path": self.tls_client_key_path,
            "connect_timeout": self.connect_timeout,
            "rpc_timeout": self.rpc_timeout,
            "max_message_length_bytes": self.max_message_length_bytes,
        }


class KyroDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    m: int
    ef_construct: int = Field(alias="ef_construction")
    ef_search: int | None = 0
    ingest_rpc: IngestRPC = IngestRPC.BULK_LOAD_HNSW

    class Config:
        allow_population_by_field_name = True

    def parse_metric(self) -> str:
        if self.metric_type is None:
            raise ValueError("KyroDB metric_type must be set before benchmarking")
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type in (MetricType.IP, MetricType.DP):
            return "inner_product"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        msg = f"Unsupported metric_type for KyroDB: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "distance": self.parse_metric(),
            "m": self.m,
            "ef_construct": self.ef_construct,
            "ingest_rpc": self.ingest_rpc.value,
        }

    def search_param(self) -> dict:
        params = {}
        if self.ef_search and self.ef_search > 0:
            params["ef_search"] = self.ef_search
        return params
