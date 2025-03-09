import math
import re
import numpy as np
from typing import Dict, Any, Union, List
from sympy import symbols, sympify, solve, Eq

class CalculatorTool:
    name = "calculator"
    description = "Performs mathematical calculations including arithmetic operations, equations, and unit conversions"
    category = "Utility Tool"
    subcategory = "mathematics"
    functionType = "backend"
    dangerous = False
    
    # Define parameter schema similar to the TypeScript format
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate"
            }
        },
        "required": ["expression"]
    }
    
    keywords = ["calculate", "compute", "math", "solve", "equation", "+", "-", "*", "/", "plus", "minus", "divided by", "times"]
    
    @staticmethod
    async def run(props: Dict[str, Any]) -> str:
        """Execute the calculator tool"""
        expression = props.get("expression", "")
        
        if not expression:
            # Try to extract expression from a full query
            query = props.get("query", "")
            expression = CalculatorTool._extract_math_expression(query)
            
        if not expression:
            return "No valid mathematical expression found. Please provide an expression to calculate."
            
        try:
            # First attempt to solve as an equation
            if "=" in expression:
                return CalculatorTool._solve_equation(expression)
            
            # Then try as a standard expression
            # Clean and sanitize the expression
            safe_expr = CalculatorTool._sanitize_expression(expression)
            
            # Replace common math terms
            safe_expr = CalculatorTool._replace_math_terms(safe_expr)
            
            # Calculate
            result = eval(safe_expr, {"__builtins__": None}, {
                "math": math,
                "np": np,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "log": math.log,
                "log10": math.log10,
                "pi": math.pi,
                "e": math.e,
                "abs": abs,
                "pow": pow,
                "round": round
            })
            
            # Format result
            if isinstance(result, float):
                # Determine if result should be shown as integer
                if result.is_integer():
                    result = int(result)
                else:
                    # Round to a reasonable precision
                    result = round(result, 10)
                    # Remove trailing zeros
                    result = str(result).rstrip('0').rstrip('.') if '.' in str(result) else result
            
            return f"The result of {expression} is {result}"
            
        except Exception as e:
            return f"Sorry, I couldn't calculate that. Error: {str(e)}"
    
    @staticmethod
    def _sanitize_expression(expression: str) -> str:
        """Sanitize the expression to prevent code injection"""
        # Remove any non-math characters and restricted functions
        allowed_pattern = r'[\d\+\-\*\/\(\)\.\,\s\^\%]'
        filtered_expr = ''.join(re.findall(allowed_pattern, expression))
        
        # Replace ^ with ** for exponentiation
        filtered_expr = filtered_expr.replace('^', '**')
        
        return filtered_expr
    
    @staticmethod
    def _replace_math_terms(expression: str) -> str:
        """Replace mathematical terms with their function equivalents"""
        # Replace sqrt(x) with math.sqrt(x) etc.
        term_replacements = {
            r'\bsqrt\(': 'math.sqrt(',
            r'\bsin\(': 'math.sin(',
            r'\bcos\(': 'math.cos(',
            r'\btan\(': 'math.tan(',
            r'\blog\(': 'math.log(',
            r'\blog10\(': 'math.log10(',
            r'\babs\(': 'abs(',
            r'\bpi\b': 'math.pi',
            r'\be\b': 'math.e'
        }
        
        result = expression
        for pattern, replacement in term_replacements.items():
            result = re.sub(pattern, replacement, result)
            
        return result
    
    @staticmethod
    def _extract_math_expression(text: str) -> str:
        """Extract a mathematical expression from text"""
        # Look for patterns like (2+8)/2 + 5 or other math expressions
        math_pattern = re.compile(r'[\(]?[\d\+\-\*\/\.\(\)\s\^\%]+[\)]?')
        matches = math_pattern.findall(text)
        
        if matches:
            # Use the longest match as it's likely the full expression
            return max(matches, key=len).strip()
        return ""
    
    @staticmethod
    def _solve_equation(equation: str) -> str:
        """Solve a simple equation using sympy"""
        try:
            # Split by = sign
            sides = equation.split('=')
            if len(sides) != 2:
                return f"Invalid equation format: {equation}. Please use format like 'x + 2 = 5'"
            
            left_side, right_side = sides
            
            # Create a symbol for the variable (assuming single variable)
            var_name = None
            for char in equation:
                if char.isalpha() and char != 'e':  # Skip 'e' as it could be math.e
                    var_name = char
                    break
            
            if not var_name:
                return f"No variable found in equation: {equation}"
                
            var = symbols(var_name)
            
            # Create equation and solve
            eq = Eq(sympify(left_side), sympify(right_side))
            solution = solve(eq, var)
            
            if not solution:
                return f"No solution found for equation: {equation}"
                
            # Format solution
            if len(solution) == 1:
                return f"Solution: {var_name} = {solution[0]}"
            else:
                solutions_str = ", ".join([f"{var_name} = {s}" for s in solution])
                return f"Solutions: {solutions_str}"
                
        except Exception as e:
            return f"Error solving equation: {str(e)}"