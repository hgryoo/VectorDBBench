from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


class CubridConfigDict(TypedDict):
    """These keys will be directly used as kwargs in Cubrid connection string,
    so the names must match exactly Cubrid API"""

    user: str
    password: str
    host: str
    port: int
    dbname: str

class CubridConfig(DBConfig):
    user_name: str = "dba"
    password: SecretStr
    host: str = "localhost"
    port: int = 33000
    db_name: str = "ann"

    def to_dict(self) -> CubridConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.db_name,
            "user": self.user_name,
            "password": pwd_str,
        }


class CubridIndexConfig(BaseModel, DBCaseConfig):
    """Base config for Cubrid"""

    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return "euclidean"
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        msg = f"Metric type {self.metric_type} is not supported!"
        raise ValueError(msg)

    def parse_metric_fun_op(self) -> str:
        if self.metric_type == MetricType.L2:
            return "<->"
        if self.metric_type == MetricType.IP:
            return "<#>"
        if self.metric_type == MetricType.COSINE:
            return "<c>"
        return "<>"

class CubridHNSWConfig(CubridIndexConfig):
    
    m: int | None
    ef_construction: int | None
    ef_search: int | None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric": self.parse_metric(),
            "index_type": self.index.value,
            "m": self.m,
            "ef_construction": self.ef_construction,
        }

    def search_param(self) -> dict:
        return {
            "metric_fun_op": self.parse_metric_fun_op(),
            "ef_search": self.ef_search,
        }


_Cubrid_case_config = {
    IndexType.HNSW: CubridHNSWConfig,
}
