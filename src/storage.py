"""https://elasticsearch-py.readthedocs.io/en/latest/async.html"""
from __future__ import annotations

import os
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from src.config import logger
from pydantic_settings import BaseSettings
import asyncio



class Settings(BaseSettings):
    """Base settings object to inherit from."""

    class Config:
        env_file = os.path.join(os.getcwd(), ".env")
        env_file_encoding = "utf-8"



class ElasticClient(AsyncElasticsearch):
    """Handling with AsyncElasticsearch"""
    def __init__(self, *args, **kwargs):
        self.max_hits = 300
        self.chunk_size = 500
        self.loop = asyncio.new_event_loop()
        super().__init__(
            #"http://elasticsearch.dev.nlp.aservices.tech:9200/"
            # hosts="http://srv01.nlp.dev.msk2.sl.amedia.tech:9200",
            hosts="http://elasticsearch.dev.nlp.aservices.tech:9200/",
            basic_auth=("elastic", "changeme"),
            request_timeout=100,
            max_retries=50,
            retry_on_timeout=True,
            *args,
            **kwargs,
        )


    def create_index(self, index_name: str = None) -> None:
        """
        :param index_name:
        """

        async def create(index: str = None) -> None:
            """Creates the index if one does not exist."""
            try:
                await self.indices.create(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible create index with name {}".format(index_name))

        self.loop.run_until_complete(create(index_name))
        self.loop.close()

    def delete_index(self, index_name) -> None:
        """Deletes the index if one exists."""

        async def delete(index: str):
            """
            :param index:
            """
            try:
                await self.indices.delete(index=index)
                await self.close()
            except:
                await self.close()
                logger.info("impossible delete index with name {}".format(index_name))

        self.loop.run_until_complete(delete(index_name))

    async def search_by_query(self, index: str, query: dict):
        """
        :param query:
        :return:
        """
        resp = await self.search(
            allow_partial_search_results=True,
            min_score=0,
            index=index,
            query=query,
            size=self.max_hits)
        await self.close()
        return resp

    async def delete_by_ids(self, index_name: str, del_ids: list):
        """
        :param index_name:
        :param del_ids:
        """
        _gen = ({"_op_type": "delete", "_index": index_name, "_id": i} for i in del_ids)
        await async_bulk(
            self,
            _gen,
            chunk_size=self.settings.chunk_size,
            raise_on_error=False,
            stats_only=True,
        )

    def add_docs(self, index_name: str, docs: list[dict]):
        async def add_docs_bulk(index_name_: str, docs_: list[dict]):
            """
            :param index_name:
            :param docs:
            """
            try:
                _gen = ({"_index": index_name_, "_source": doc} for doc in docs_)
                await async_bulk(
                    self, _gen, chunk_size=self.chunk_size, stats_only=True
                )
                logger.info("adding {} documents in index {}".format(len(docs_), index_name_))
            except Exception:
                logger.exception(
                    "Impossible adding {} documents in index {}".format(
                        len(docs_), index_name_
                    )
                )
        self.loop.run_until_complete(add_docs_bulk(index_name, docs))
        self.loop.close()