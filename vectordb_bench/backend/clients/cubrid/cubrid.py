import logging
from contextlib import contextmanager

import CUBRIDdb
import numpy as np

from ..api import VectorDB
from .config import CubridConfigDict, CubridIndexConfig

log = logging.getLogger(__name__)

METRIC_PROPERTIES = {
    "angular": {
        "distance_operator": "<c>",
        "ops_type": "COSINE",
    },
    "euclidean": {
        "distance_operator": "<->",
        "ops_type": "EUCLIDEAN",
    }
}

class Cubrid(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: CubridConfigDict,
        db_case_config: CubridIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "CUBRID"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        if drop_old:
            self._drop_table()
            self._create_table(dim)

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None
        self.new_init = False

    @staticmethod
    def _create_connection(**kwargs) -> tuple[CUBRIDdb.Connection, CUBRIDdb.Cursor]:
        conn = CUBRIDdb.connect(**kwargs)
        cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"

        return conn, cursor

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cursor.execute(
            f"DROP TABLE IF EXISTS {self.table_name}"
        )
        self.conn.commit()

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()

        try:
            self.cursor.execute(
                f"""
                CREATE TABLE {self.table_name} (
                    id BIGINT PRIMARY KEY,
                    embedding VECTOR({self.dim}) NOT NULL
                )
                """
            )
            idx_stmt = (
                "CREATE VECTOR INDEX idx_v ON %s(embedding %s) "
                "WITH (m = %d, ef_construction = %d);" % (
                    self.table_name,
                    METRIC_PROPERTIES[index_param["metric_type"]]["ops_type"],
                    index_param["M"],
                    index_param["ef_search"]
                )
            )
            self.cursor.execute(idx_stmt)
            self.conn.commit()

        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise e from None

    @contextmanager
    def init(self):
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn, self.cursor = self._create_connection(**self.db_config)

        search_param = self.case_config.search_param()

        self.cursor.execute(f"SET SYSTEM PARAMETERS 'hnsw_ef_search = {search_param['ef_search']}'")
        self.cursor.execute("COMMIT")

        self.insert_sql = f"INSERT INTO {self.table_name} (id, embedding) VALUES (%s, %s)"
        self.select_sql = (
            f"SELECT id FROM {self.table_name}"
            f"ORDER BY embedding {search_param['metric_type']} %s LIMIT %d"
        )
        self.select_sql_with_filter = (
            f"SELECT id FROM {self.table_name} WHERE id >= %d "
            f"ORDER BY embedding {search_param['metric_type']} %s LIMIT %d"
        )

        self.new_init = True

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def ready_to_load(self) -> bool:
        pass

    def optimize(self) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            batch_data = []
            for i, row in enumerate(metadata_arr):
                batch_data.append((int(row), "[" + ",".join(map(str, embeddings_arr[i])) + "]"))

            self.cursor.executemany(self.insert_sql, batch_data)
            self.cursor.execute("COMMIT")

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Vector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        vector_str = "[" + ",".join(map(str, query)) + "]"

        if self.new_init:
            if filters:
                self.cursor._cs.prepare(self.select_sql_with_filter)
            else:
                self.cursor._cs.prepare(self.select_sql)
            self.new_init = False

        set_type = None
        if filters:
            args = (filters.get("id"), vector_str, k)
            self.cursor._bind_params(args, set_type)
            r = self.cursor._cs.execute()
            self.cursor.rowcount = self.cursor._cs.rowcount
            self.cursor.description = self.cursor._cs.description
        else:
            args = (vector_str, k)
            self.cursor._bind_params(args, set_type)
            r = self.cursor._cs.execute()
            self.cursor.rowcount = self.cursor._cs.rowcount
            self.cursor.description = self.cursor._cs.description

        return [id for (id,) in self.cursor.fetchall()]
