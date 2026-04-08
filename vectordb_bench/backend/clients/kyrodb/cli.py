from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB
from vectordb_bench.cli.cli import (
    CommonTypedDict,
    HNSWFlavor3,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)

DBTYPE = DB.KyroDB


class KyroDBTypedDict(CommonTypedDict, HNSWFlavor3):
    host: Annotated[
        str,
        click.option("--host", type=str, required=True, help="KyroDB gRPC host"),
    ]
    port: Annotated[
        int,
        click.option("--port", type=int, required=True, help="KyroDB gRPC port"),
    ]
    api_key: Annotated[
        str | None,
        click.option("--api-key", type=str, required=False, help="KyroDB API key for auth-enabled clusters"),
    ]
    use_tls: Annotated[
        bool,
        click.option(
            "--use-tls/--no-use-tls",
            type=bool,
            default=False,
            show_default=True,
            help="Use TLS when connecting to the KyroDB gRPC endpoint",
        ),
    ]
    tls_root_cert_path: Annotated[
        str | None,
        click.option(
            "--tls-root-cert-path",
            type=click.Path(exists=True, dir_okay=False, path_type=str),
            required=False,
            help="Optional PEM bundle for custom CA trust when --use-tls is enabled",
        ),
    ]
    tls_client_cert_path: Annotated[
        str | None,
        click.option(
            "--tls-client-cert-path",
            type=click.Path(exists=True, dir_okay=False, path_type=str),
            required=False,
            help="Optional client certificate PEM for mTLS benchmarks",
        ),
    ]
    tls_client_key_path: Annotated[
        str | None,
        click.option(
            "--tls-client-key-path",
            type=click.Path(exists=True, dir_okay=False, path_type=str),
            required=False,
            help="Optional client private key PEM for mTLS benchmarks",
        ),
    ]
    connect_timeout: Annotated[
        float,
        click.option(
            "--connect-timeout",
            type=float,
            default=15.0,
            show_default=True,
            help="Seconds to wait for initial gRPC channel readiness",
        ),
    ]
    rpc_timeout: Annotated[
        float | None,
        click.option(
            "--rpc-timeout",
            type=float,
            required=False,
            help="Optional per-RPC timeout in seconds; unset means no explicit deadline",
        ),
    ]
    ingest_rpc: Annotated[
        str,
        click.option(
            "--ingest-rpc",
            type=click.Choice(["bulk_load_hnsw", "bulk_insert", "insert"], case_sensitive=False),
            default="bulk_load_hnsw",
            show_default=True,
            help="KyroDB write path to benchmark",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(KyroDBTypedDict)
def KyroDB(**parameters: Unpack[KyroDBTypedDict]):
    from .config import IngestRPC, KyroDBConfig, KyroDBIndexConfig

    has_client_cert = bool(parameters["tls_client_cert_path"])
    has_client_key = bool(parameters["tls_client_key_path"])
    if has_client_cert != has_client_key:
        raise click.UsageError(
            "--tls-client-cert-path and --tls-client-key-path must be provided together for mTLS"
        )
    if (
        parameters["tls_root_cert_path"]
        or parameters["tls_client_cert_path"]
        or parameters["tls_client_key_path"]
    ) and not parameters["use_tls"]:
        raise click.UsageError("--use-tls is required when TLS certificate paths are provided")

    run(
        db=DBTYPE,
        db_config=KyroDBConfig(
            db_label=parameters["db_label"],
            host=parameters["host"],
            port=parameters["port"],
            api_key=SecretStr(parameters["api_key"]) if parameters["api_key"] else None,
            use_tls=parameters["use_tls"],
            tls_root_cert_path=parameters["tls_root_cert_path"],
            tls_client_cert_path=parameters["tls_client_cert_path"],
            tls_client_key_path=parameters["tls_client_key_path"],
            connect_timeout=parameters["connect_timeout"],
            rpc_timeout=parameters["rpc_timeout"],
        ),
        db_case_config=KyroDBIndexConfig(
            m=parameters["m"],
            ef_construct=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
            ingest_rpc=IngestRPC(parameters["ingest_rpc"].lower()),
        ),
        **parameters,
    )
