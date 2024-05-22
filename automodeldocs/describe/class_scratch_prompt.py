from typing import Optional

from automodeldocs.shared_prompts.base import Prompt
from automodeldocs.structures import DescriptionContext


class ScratchClassPrompt(Prompt):
    _identity: str = """You are a Machine Learning Engineer. Your manager has requested a report into how a specific 
    class is used within the codebase, and you are tasked with providing a technical report on it.

You'll be provided with a code class. Write down technical notes that meet all your goals, and help a prospective 
user understand what the class does.

Goals; 
1. ML Pipeline Positioning: Understand the class's role in the overall machine learning pipeline. 
2. ML Specific Operations: Analyze the core logic and operations of the class, focusing on identifying common machine 
learning processes such as gradient descent, forward pass, backpropagation, normalization, encoding, etc., 
3. ML Data Flow Analysis: Analyze how the class transforms the input data to produce its outputs. This might 
include reshaping data, tensor operations, or specific transformations common in machine learning. 
4. Class Input and Output: Understand the initialisers and usages of the class. This includes the data types, 
shapes (especially for multi-dimensional arrays or tensors), and how they relate to the machine learning model or 
data being processed.
5. Identify any machine learning models being created
6. Describe any machine learning models being created, in as much depth as possible. If the model is a neural network
model, ensure you describe the architecture of the model. If you cannot, explain why you cannot.

# Context;
{context}

Always reply in the following format;
# Scratch
Your initial notes on the class to be provided, meeting all goals above.

"""

    def __init__(
        self,
        class_name: str,
        context: Optional[DescriptionContext],
    ) -> None:
        self.class_name = class_name
        self.context = context

    def system_message(self, **kwargs) -> str:
        return (
            self._identity.format(
                class_name=self.class_name,
                context=self.context.as_str() if self.context else "",
            )
            + "\n"
        )

    def user_message(self, **kwargs) -> str:
        return f"You will be creating a report on the class: {self.class_name}"
