from typing import Dict, Any, Union, List
import math

def perform_arithmetic(
    operation: str, 
    numbers: List[float]
) -> Dict[str, Union[float, str]]:
    """
    Perform basic arithmetic operations on a list of numbers.
    
    Args:
        operation: The operation to perform ('add', 'subtract', 'multiply', 'divide')
        numbers: List of numbers to operate on
        
    Returns:
        Dictionary with result and explanation
    """
    if not numbers:
        return {"result": None, "explanation": "No numbers provided"}
    
    result = numbers[0]
    explanation = f"Starting with {numbers[0]}"
    
    try:
        if operation == "add":
            for num in numbers[1:]:
                result += num
                explanation += f" + {num}"
        elif operation == "subtract":
            for num in numbers[1:]:
                result -= num
                explanation += f" - {num}"
        elif operation == "multiply":
            for num in numbers[1:]:
                result *= num
                explanation += f" × {num}"
        elif operation == "divide":
            for num in numbers[1:]:
                if num == 0:
                    return {"result": None, "explanation": "Cannot divide by zero"}
                result /= num
                explanation += f" ÷ {num}"
        else:
            return {"result": None, "explanation": f"Unknown operation: {operation}"}
            
        explanation += f" = {result}"
        return {"result": result, "explanation": explanation}
    except Exception as e:
        return {"result": None, "explanation": f"Error: {str(e)}"}

def advanced_math(
    operation: str, 
    value: float, 
    second_value: float = None
) -> Dict[str, Union[float, str]]:
    """
    Perform advanced mathematical operations.
    
    Args:
        operation: Operation to perform ('sqrt', 'power', 'log', etc.)
        value: Primary value for the operation
        second_value: Secondary value (e.g., exponent for power)
        
    Returns:
        Dictionary with result and explanation
    """
    try:
        if operation == "sqrt":
            if value < 0:
                return {"result": None, "explanation": "Cannot calculate square root of negative number"}
            result = math.sqrt(value)
            explanation = f"√{value} = {result}"
        elif operation == "power":
            if second_value is None:
                return {"result": None, "explanation": "Second value (exponent) is required for power operation"}
            result = math.pow(value, second_value)
            explanation = f"{value}^{second_value} = {result}"
        elif operation == "log":
            if value <= 0:
                return {"result": None, "explanation": "Cannot calculate logarithm of non-positive number"}
            base = 10 if second_value is None else second_value
            if base <= 0 or base == 1:
                return {"result": None, "explanation": "Invalid logarithm base"}
            result = math.log(value, base)
            explanation = f"log_{base}({value}) = {result}"
        else:
            return {"result": None, "explanation": f"Unknown operation: {operation}"}
            
        return {"result": result, "explanation": explanation}
    except Exception as e:
        return {"result": None, "explanation": f"Error: {str(e)}"}

def convert_units(
    value: float,
    from_unit: str,
    to_unit: str
) -> Dict[str, Union[float, str]]:
    """
    Convert a value from one unit to another.
    
    Args:
        value: The value to convert
        from_unit: The source unit
        to_unit: The target unit
        
    Returns:
        Dictionary with result and explanation
    """
    # Unit conversion factors (to base SI units)
    length_units = {
        "m": 1, "cm": 0.01, "km": 1000, "inch": 0.0254, "ft": 0.3048, "mile": 1609.34
    }
    
    weight_units = {
        "kg": 1, "g": 0.001, "lb": 0.453592, "oz": 0.0283495
    }
    
    temperature_conversions = {
        "c_to_f": lambda c: c * 9/5 + 32,
        "f_to_c": lambda f: (f - 32) * 5/9,
        "c_to_k": lambda c: c + 273.15,
        "k_to_c": lambda k: k - 273.15
    }
    
    try:
        # Handle temperature conversions specially
        if from_unit.lower() in ["c", "celsius"] and to_unit.lower() in ["f", "fahrenheit"]:
            result = temperature_conversions["c_to_f"](value)
            return {"result": result, "explanation": f"{value}°C = {result}°F"}
        elif from_unit.lower() in ["f", "fahrenheit"] and to_unit.lower() in ["c", "celsius"]:
            result = temperature_conversions["f_to_c"](value)
            return {"result": result, "explanation": f"{value}°F = {result}°C"}
            
        # Handle length and weight conversions
        if from_unit in length_units and to_unit in length_units:
            # Convert to base unit (meters) then to target unit
            result = value * length_units[from_unit] / length_units[to_unit]
            return {"result": result, "explanation": f"{value} {from_unit} = {result} {to_unit}"}
        elif from_unit in weight_units and to_unit in weight_units:
            # Convert to base unit (kg) then to target unit
            result = value * weight_units[from_unit] / weight_units[to_unit]
            return {"result": result, "explanation": f"{value} {from_unit} = {result} {to_unit}"}
        else:
            return {"result": None, "explanation": f"Cannot convert between {from_unit} and {to_unit}"}
    except Exception as e:
        return {"result": None, "explanation": f"Error: {str(e)}"}