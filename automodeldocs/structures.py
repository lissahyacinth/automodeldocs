import dataclasses
from dataclasses import dataclass

from automodeldocs.definitions import OpenAIInputMessage


@dataclass
class DescriptionContext:
    context: dict[str, str]

    def as_str(self) -> str:
        context = ""
        for key, description in self.context.items():
            if len(description.strip()) > 0:
                context = f"Name: {key}" f"\nDescription: {description}\n\n"
        return context


@dataclass
class Feedback:
    prior_description: str
    report: str


@dataclass
class Improvement:
    feedback: list[Feedback]
    context: DescriptionContext

    def as_dict(self) -> dict:
        return {
            "feedback": [dataclasses.asdict(f) for f in self.feedback],
            "context": dataclasses.asdict(self.context),
        }


def message_from_user_str(user_str: str) -> OpenAIInputMessage:
    return {"role": "user", "content": user_str}


def message_from_system_str(user_str: str) -> OpenAIInputMessage:
    return {"role": "system", "content": user_str}


def message_from_assistant_str(user_str: str) -> OpenAIInputMessage:
    return {"role": "assistant", "content": user_str}
