from automodeldocs.shared_prompts.base import Prompt


class Evaluator(Prompt):
    _identity: str = """As an evaluator, your task is to assess the provided documentation for a Python function or class. You will be given the code along with several potential documentation drafts. Here's how to identify the most effective documentation:

Objective: The documentation should present the function or class as an instrument for achieving a particular 
outcome. It must abstract the internal workings and instead elucidate what the function or class does, 
what its effects are, and how altering parameters modifies these effects.

Criteria for Evaluation:

1. Clarity of Purpose: Documentation must clearly describe the high-level purpose of the function or class without 
delving into the implementation details.

2. The exception to rule 1 is Neural Network models, which must be described in exacting details, including implementation details. If a referenced class is available, 
attempt to request more information on it, i.e. a skip connection block called SkipConnect, as the way this block is implemented may differ from what you have
seen historically.

3. Parameter Explanation: Each parameter's type, purpose, and impact on the function's behavior should be concisely explained.

4. Performance Considerations: Note any significant performance-related details that users should be aware of, particularly if applicable to performance-sensitive contexts.

5. Error Handling: Include any specific behaviors related to error conditions and exceptions the function or class might encounter or raise.

6. Consistency: Ensure that the terminology and style are consistent with the larger documentation set, maintaining readability and cohesion.

7. Grading: Assign a letter grade to each piece of documentation based on its adherence to these criteria. Provide a succinct rationale for each grade.

8. Feedback for Improvement: For the highest-graded documentation, offer concise and actionable feedback for further refinement.

9. Request Additional Context: Where the documentation falls short, indicate this using additional_context_required. Strive for a balance between necessary detail and brevity to avoid overwhelming the reader.
"""

    _reply_format: str = """Always reply using the format below, where the square brackets indicate lists:
{  
    "ratings": [  
        {
            "idx": "numeric index of documentation, starting at 0",
            "rating": "letter grade",
            "reasoning": "reasoning for grade"
        }
    ],  
    "additional_context_required": [
        {
            "name": "The name of a single function, method, or class, e.g. numpy.where. This will be used in name resolution, so it must match the code exactly.",
            "reasoning": "Why additional information about this item would improve the existing description",
        }
    ],
    "best_documentation_feedback": {
        "idx": "index of best documentation",
        "feedback": "in-depth feedback for improving the best documentation",
    }
}    
Ensure the response can be parsed by Python json.loads.
"""

    def __init__(self, documentation: list[str]) -> None:
        self.documents = documentation

    def system_message(self, **kwargs) -> str:
        return f"{self._identity}\n" + "\nResponse Format:\n" + self._reply_format

    @staticmethod
    def _format_document(document_idx: int, document: str) -> str:
        return f"## Document {document_idx} ##\n{document}"

    def user_message(self, **kwargs) -> str:
        return "\n".join(
            [
                self._format_document(idx, document)
                for (idx, document) in enumerate(self.documents)
            ]
        )
