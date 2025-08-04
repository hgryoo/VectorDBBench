from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)


class CubridTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            required=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            required=True,
        ),
    ]

    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            help="Db host",
            default="127.0.0.1",
        ),
    ]

    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=33000,
            help="Db Port",
        ),
    ]


class CubridHNSWTypedDict(CubridTypedDict):
    m: Annotated[
        int | None,
        click.option(
            "--m",
            type=int,
            help="M parameter in MHNSW vector indexing",
            required=False,
        ),
    ]

    ef_search: Annotated[
        int | None,
        click.option(
            "--ef-search",
            type=int,
            help="Cubrid system variable mhnsw_min_limit",
            required=False,
        ),
    ]

@cli.command()
@click_parameter_decorators_from_typed_dict(CubridHNSWTypedDict)
def CubridHNSW(
    **parameters: Unpack[CubridHNSWTypedDict],
):
    from .config import CubridConfig, CubridHNSWConfig

    run(
        db=DB.Cubrid,
        db_config=CubridConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
        ),
        db_case_config=CubridHNSWConfig(
            M=parameters["m"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
