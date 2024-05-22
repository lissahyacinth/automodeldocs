from typing import Optional

from automodeldocs.definitions import OpenAIInputMessage
from automodeldocs.shared_prompts.base import Prompt
from automodeldocs.structures import (
    DescriptionContext,
    Improvement,
    message_from_user_str,
    message_from_assistant_str,
)


class DescribeFunction(Prompt):
    _identity: str = """You are a Machine Learning Engineer writing documentation for a function.

You'll be provided with a function written in Python. Each time you submit a report, you'll receive feedback. 
Incorporate that feedback into your report.

Ideal documentation focuses on the function as a tool for doing something - it abstracts away the complexity of how 
the function is doing something, and focuses instead on how the function can be used to accomplish something, as well as 
how to change parameters to achieve different goals. 

Insufficient documentation is mere reference, i.e. 

def foo(a):
    '''`foo` calls the function g for the parameter a'''
    return g(a)
        
If the documentation didn't include the context for the function `g`, it should request it; 

def foo(a):
    '''`foo` is a wrapper around calling g - it is impossible to understand foo without more information about g'''

Whereas the ideal description provides the understanding for calling `foo` without any reference to other functions;

def g(a):
    return a**2
    
def foo(a):
    '''Squares the input'''
    return g(a)

# Context
{context}

# Technical Notes
{scratch}

Always reply in the following format;
# Report
Concise summary of the function, summarising your technical notes with the context provided.

You will be creating a report on the function: {function_name}
"""

    def __init__(
        self,
        function_name: str,
        scratch: str,
        context: Optional[DescriptionContext],
        code_source: str,
    ) -> None:
        self.function_name = function_name
        self.scratch = scratch
        self.context = context
        self.code_source = code_source

    def system_message(self, **kwargs) -> str:
        return (
            self._identity.format(
                function_name=self.function_name,
                context=self.context.as_str() if self.context is not None else "",
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
        return f"\nCode:\n{self.code_source}"
