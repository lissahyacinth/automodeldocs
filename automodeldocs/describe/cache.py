from __future__ import annotations

import dataclasses
import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import cachetools
import pandas as pd
from function_discovery.structure import FunctionContainer

from automodeldocs.structures import Improvement


@dataclass
class FunctionDescriptionRequest:
    function_info: FunctionContainer
    improvement: Optional[Improvement] = None

    @staticmethod
    def from_args(
        function_info: FunctionContainer,
        improvement: Optional[Improvement] = None,
        class_name: Optional[str] = None,
        n_iteration: int = None,
        max_iterations: int = None,
    ) -> FunctionDescriptionRequest:
        return FunctionDescriptionRequest(function_info, improvement)


class DescribeDBCache(cachetools.Cache):
    __data: pd.DataFrame
    __currsize: int = 0
    __maxsize: int = -1

    def __init__(self):
        super().__init__(maxsize=-1)
        self._reload_data()

    def _reload_data(self) -> None:
        self.__data = self._load_db()
        self.__currsize = self.__data.shape[0]

    def __contains__(self, item: FunctionDescriptionRequest) -> bool:  # type: ignore
        return self[item] is not None

    def __getitem__(self, key: FunctionDescriptionRequest) -> str:
        self._reload_data()
        # Begin with the entire dataframe and filter it down based on the request attributes.
        filtered_df = self.__data[
            (self.__data["function_source"] == key.function_info.source())
            & (self.__data["function_name"] == key.function_info.name)
        ]

        if key.improvement is not None:
            filtered_df = filtered_df[
                filtered_df["improvements"] == json.dumps(key.improvement.as_dict())
            ]

        if not filtered_df.empty:
            row = filtered_df.iloc[0]
            return row["response"]
        else:
            self.__missing__(key)
            raise RuntimeError("Unreachable")

    def __setitem__(self, key: FunctionDescriptionRequest, value: str):
        row = {
            "function_source": key.function_info.source(),
            "function_name": key.function_info.name,
            "improvements": json.dumps(
                key.improvement.as_dict() if key.improvement is not None else {}
            ),
            "response": value,
        }
        self.__data = pd.concat([self.__data, pd.DataFrame([row])], ignore_index=True)
        self._dump_data()

    @staticmethod
    def _source() -> pathlib.Path:
        return pathlib.Path.home() / "_function_descriptions.parquet"

    @classmethod
    def _load_db(cls) -> pd.DataFrame:
        if cls._source().exists():
            return pd.read_parquet(cls._source())
        empty_df = pd.DataFrame(
            columns=[
                "function_source",
                "function_name",
                "prior_description",
                "improvements",
                "response",
            ]
        )

        # You can now specify the dtype for simpler columns
        empty_df["function_source"] = empty_df["function_source"].astype("str")
        empty_df["function_name"] = empty_df["function_name"].astype("str")
        empty_df["response"] = empty_df["response"].astype("str")
        return empty_df

    def _dump_data(self):
        # TODO: Add queue/write ahead.
        self.__data.to_parquet(self._source())
