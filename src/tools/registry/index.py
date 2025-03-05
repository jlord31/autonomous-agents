from typing import Dict, List, Any
from tools.calculator import perform_arithmetic, advanced_math, convert_units
from tools.email import send_email

def get_tool_configs():
    """Return all available tool configurations"""
    
    return [
        # Calculator tools
        {
            "name": "calculator",
            "type": "function",
            "function": {
                "name": "perform_arithmetic",
                "description": "Perform basic arithmetic operations on numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform"
                        },
                        "numbers": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "The numbers to perform the operation on"
                        }
                    },
                    "required": ["operation", "numbers"]
                }
            },
            "implementation": perform_arithmetic
        },
        {
            "name": "calculator",
            "type": "function",
            "function": {
                "name": "advanced_math",
                "description": "Perform advanced mathematical operations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["sqrt", "power", "log"],
                            "description": "The mathematical operation to perform"
                        },
                        "value": {
                            "type": "number",
                            "description": "The primary value for the operation"
                        },
                        "second_value": {
                            "type": "number",
                            "description": "Optional second value (e.g., exponent for power)"
                        }
                    },
                    "required": ["operation", "value"]
                }
            },
            "implementation": advanced_math
        },
        {
            "name": "calculator",
            "type": "function",
            "function": {
                "name": "convert_units",
                "description": "Convert values between different units of measurement",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {
                            "type": "number",
                            "description": "The value to convert"
                        },
                        "from_unit": {
                            "type": "string",
                            "description": "The source unit (e.g., 'm', 'kg', 'c')"
                        },
                        "to_unit": {
                            "type": "string",
                            "description": "The target unit (e.g., 'ft', 'lb', 'f')"
                        }
                    },
                    "required": ["value", "from_unit", "to_unit"]
                }
            },
            "implementation": convert_units
        },
        
        # Email tool
        {
            "name": "send_email",
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send an email to a recipient",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_email": {
                            "type": "string",
                            "description": "Recipient email address"
                        },
                        "subject": {
                            "type": "string",
                            "description": "Email subject line"
                        },
                        "body": {
                            "type": "string",
                            "description": "Email body content"
                        },
                        "from_email": {
                            "type": "string",
                            "description": "Sender email address (optional)"
                        },
                        "cc": {
                            "type": "string",
                            "description": "CC recipients (optional)"
                        }
                    },
                    "required": ["to_email", "subject", "body"]
                }
            },
            "implementation": send_email
        }
    ]