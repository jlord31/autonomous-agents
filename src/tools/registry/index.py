from typing import Dict, List, Any
from tools.CalculatorTool import CalculatorTool
from tools.EmailTool import EmailTool

# Registry of all available tools
TOOLS = {
    CalculatorTool.name: CalculatorTool,
    EmailTool.name: EmailTool
}

def get_tool_configs() -> List[Dict[str, Any]]:
    """Get all tool configurations for registration with agents"""
    configs = []
    
    for tool_name, tool_class in TOOLS.items():
        configs.append({
            "name": tool_class.name,
            "description": tool_class.description,
            "category": getattr(tool_class, "category", "General Tool"),
            "subcategory": getattr(tool_class, "subcategory", "utility"),
            "functionType": getattr(tool_class, "functionType", "backend"),
            "dangerous": getattr(tool_class, "dangerous", False),
            "type": "function",
            "module": tool_class.__module__,
            "class": tool_class.__name__,
            "function": "run",
            "parameters": getattr(tool_class, "parameters", {}),
            "keywords": getattr(tool_class, "keywords", []),
        })
    
    return configs