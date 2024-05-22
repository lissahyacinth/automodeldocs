from automodeldocs.shared_prompts.base import Prompt


class ResponseFormat(Prompt):
    def __init__(self):
        pass

    _format: str = """
    Always reply using the format below; 
    
{  
    "thoughts": {  
        "text": "thought",  
        "reasoning": "reasoning",  
        "plan": "- short bulleted \n - list that conveys \n - long-term plan",  
        "criticism": "constructive self-criticism",  
        "speak": "thoughts summary to say to user"  
    },  
        "command": {  
        "name": "command name",  
        "args": {  
            "arg name": "value"  
        }  
    }
}    
Ensure the response can be parsed by Python json.loads.  
    """

    @classmethod
    def as_str(cls) -> str:
        return cls._format
