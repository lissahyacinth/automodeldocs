from automodeldocs.shared_prompts.base import Prompt


class FormatResponsePrompt(Prompt):
    _identity: str = """Rewrite the following function definition.

1. Remove any extraneous information, duplication, etc. 
2. Rewrite segments that may be confusing, but do not add any new information.
3. Do not remove any information - only seek to rewrite for clarity. 
4. Keep the language as clear as possible, defining any acronyms or specialised language. 
5. Remove all headings, and instead rewrite it in the style and tone of a docstring within the PyTorch codebase. 

You will be formatting a report on the function: {function_name}
"""

    def __init__(self, function_name: str, report: str) -> None:
        self.function_name = function_name
        self.report = report

    def system_message(self, **kwargs) -> str:
        return self._identity.format(function_name=self.function_name) + "\n"

    def user_message(self, **kwargs) -> str:
        return f"\nCode:\n{self.report}"


class FormatClassResponsePrompt(Prompt):
    _identity: str = """Rewrite the following class definition.

1. Remove any extraneous information, duplication, etc. 
2. Rewrite segments that may be confusing, but do not add any new information.
3. Do not remove any information - only seek to rewrite for clarity. 
4. Keep the language as clear as possible, defining any acronyms or specialised language. 
5. Remove all headings, and instead rewrite it in the style and tone of a docstring within the PyTorch codebase. 

You will be formatting a report on the class: {class_name}
"""

    def __init__(self, class_name: str, report: str) -> None:
        self.class_name = class_name
        self.report = report

    def system_message(self, **kwargs) -> str:
        return self._identity.format(class_name=self.class_name) + "\n"

    def user_message(self, **kwargs) -> str:
        return f"\nCode:\n{self.report}"
