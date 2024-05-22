from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from json import JSONDecodeError

from automodeldocs.chat.send_message import reformat_json
from automodeldocs.response.formatted import FormattedOpenAIResponse


@dataclass
class EvaluationResponse:
    documentation_idx: int
    feedback: str
    additional_context_items: list[str]
    missing_information: bool

    @classmethod
    def from_dict(cls, item: dict) -> EvaluationResponse:
        return cls(
            item["documentation_idx"],
            item["feedback"],
            item["additional_context_items"],
            item["missing_information"],
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    async def from_fmt(msg: FormattedOpenAIResponse) -> EvaluationResponse:
        try:
            evaluation_response = json.loads(
                msg.content.replace("\n", "").replace(r"\'", r"\\'")
            )
        except JSONDecodeError:
            reformatted_json = (
                (await reformat_json(msg.content))[-1]
                .content.replace("\n", "")
                .replace(r"\'", r"\\'")
            )
            try:
                evaluation_response = json.loads(reformatted_json)
            except JSONDecodeError:
                breakpoint()
                raise RuntimeError("Tried to reformat, still failed.")

        try:
            _ratings = evaluation_response["ratings"]
            additional_context_items = evaluation_response[
                "additional_context_required"
            ]
            best_documentation_feedback = evaluation_response[
                "best_documentation_feedback"
            ]
            print(additional_context_items)
            additional_items = []
            # Also deal with ' x and y' types
            for additional_context_request in additional_context_items:
                if len(additional_context_request.keys()) > 0:
                    if "," in additional_context_request["name"]:
                        split_items = [
                            a.strip()
                            for a in additional_context_request["name"].split(",")
                        ]
                        for item in split_items:
                            additional_items.append(item)
                    else:
                        additional_items.append(additional_context_request["name"])
            return EvaluationResponse(
                documentation_idx=int(best_documentation_feedback["idx"]),
                feedback=best_documentation_feedback["feedback"],
                additional_context_items=additional_items,
                missing_information=(len(additional_context_items) > 0),
            )
        except KeyError:
            logging.error(f"Couldn't parse JSON reply - {evaluation_response}")
            raise RuntimeError
