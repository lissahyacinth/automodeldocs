from typing import Optional

from automodeldocs.definitions import OpenAIInputMessage
from automodeldocs.shared_prompts.base import Prompt
from automodeldocs.structures import (
    DescriptionContext,
    Improvement,
    message_from_user_str,
    message_from_assistant_str,
)


class DescribeClass(Prompt):
    _identity: str = """You are a Machine Learning Engineer. Your superior has requested a report into how a specific 
class is used within the codebase, and you are tasked with providing a technical report on it.

You'll be provided with a code class. Each time you submit a report, you'll receive feedback. Incorporate that 
feedback into your report. Your report should be succinct.

Ideal documentation focuses on the class as a tool for doing something - it abstracts away the complexity of how 
the class is doing something, and focuses instead on how the class can be used to accomplish something within 
the ML ecosystem, as well as how to change parameters for different goals. It should avoid writing code examples for 
how to use the class, as the class will only be called via an abstraction, and instead instill an intuition for 
how the class operates.

The documentation should explain the impact of functions called within the class - explaining that it calls an 
unknown class is not sufficient. If it trains a model, it is critical that at least the architecture or general
background of the model is explained.

# Context
{context}

# Technical Notes
{scratch}

Always reply in the following format;
# Report
Concise summary of the class, summarising your technical notes with the context provided.
"""

    def __init__(
        self,
        class_name: str,
        scratch: str,
        context: DescriptionContext,
    ) -> None:
        self.class_name = class_name
        self.scratch = scratch
        self.context = context

    def system_message(self, **kwargs) -> str:
        return (
            self._identity.format(
                class_name=self.class_name,
                context=self.context.as_str(),
                scratch=self.scratch,
            )
            + "\n"
        )

    @staticmethod
    def evaluation_messages(
        improvements: Improvement | None,
    ) -> list[OpenAIInputMessage]:
        if improvements is None:
            return []
        messages = []
        for feedback in improvements.feedback:
            messages.append(
                message_from_assistant_str(
                    f"\nPrevious Description:{feedback.prior_description}\n"
                )
            )
            messages.append(message_from_user_str(f"\nFeedback:\n{feedback.report}\n"))
        return messages

    def user_message(self, **kwargs) -> str:
        return f"You will be creating a report on the class: {self.class_name}"
