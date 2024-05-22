from typing import Optional

from automodeldocs.shared_prompts.base import Prompt
from automodeldocs.structures import DescriptionContext


class ScratchFunctionPrompt(Prompt):
    _identity: str = """You are a Machine Learning Engineer. Your manager has requested a report into how a specific 
    function is used within the codebase, and you are tasked with providing a technical report on it.

You'll be provided with a code function. Write down technical notes that meet all your goals, and help a prospective 
user understand what the function does.

Goals; 
1. ML Pipeline Positioning: Understand the function's role in the overall machine learning pipeline. 
2. ML Specific Operations: Analyze the core logic and operations of the function, focusing on identifying common machine 
learning processes such as gradient descent, forward pass, backpropagation, normalization, encoding, etc., 
3. ML Data Flow Analysis: Analyze how the function transforms the input data to produce its outputs. This might 
include reshaping data, tensor operations, or specific transformations common in machine learning. 
4. Function Input and Output: Understand the input parameters and output of the function. This includes the data types, 
shapes (especially for multi-dimensional arrays or tensors), and how they relate to the machine learning model or 
data being processed.
5. Identify any machine learning models being created
6. Describe any machine learning models being created, in as much depth as possible. If the model is a neural network
model, ensure you describe the architecture of the model. If you cannot, explain why you cannot.

# Context;
{context}

Always reply in the following format;
# Scratch
Your initial notes on the function to be provided, meeting all goals above.

You will be creating a report on the function: {function_name}
"""

    def __init__(
        self,
        function_name: str,
        context: Optional[DescriptionContext],
        code_source: str,
    ) -> None:
        self.function_name = function_name
        self.context = context
        self.code_source = code_source

    def system_message(self, **kwargs) -> str:
        return (
            self._identity.format(
                function_name=self.function_name,
                context=self.context.as_str() if self.context else "",
            )
            + "\n"
        )

    def user_message(self, **kwargs) -> str:
        return f"\nCode:\n{self.code_source}"
