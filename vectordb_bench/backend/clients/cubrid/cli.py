from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)


class CubridTypedDict(CommonTypedDict):
    user_name: Annotated[
        str,
        click.option("--user-name", type=str, help="Db username", required=True),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            help="Password",
            required=False,
        ),
    ]

    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            help="Postgres database port",
            default=5432,
            show_default=True,
            required=False,
        ),
    ]
    db_name: Annotated[str, click.option("--db-name", type=str, help="Db name", required=True)]

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

    ef_construction: Annotated[
        int | None,
        click.option(
            "--ef-construction",
            type=int,
            help="Cubrid system variable mhnsw_min_limit",
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

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.Cubrid,
        db_config=CubridConfig(
            db_label=parameters["db_label"],
            user_name=parameters["user_name"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
        ),
        db_case_config=CubridHNSWConfig(
            m=parameters["m"],
            ef_construction=parameters["ef_construction"],
            ef_search=parameters["ef_search"],
        ),
        **parameters,
    )
