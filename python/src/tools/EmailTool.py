from typing import Dict, Any
import json

class EmailTool:
    name = "send_email"
    description = "Sends emails to recipients"
    category = "Communication Tool"
    subcategory = "email"
    functionType = "backend"
    dangerous = False
    
    parameters = {
        "type": "object",
        "properties": {
            "recipient": {
                "type": "string",
                "description": "Email address of the recipient"
            },
            "subject": {
                "type": "string",
                "description": "Subject line of the email"
            },
            "body": {
                "type": "string",
                "description": "Content of the email"
            }
        },
        "required": ["recipient", "subject", "body"]
    }
    
    keywords = ["email", "send email", "message", "compose"]
    
    @staticmethod
    async def run(props: Dict[str, Any]) -> str:
        """Mock sending an email"""
        recipient = props.get("recipient")
        subject = props.get("subject")
        body = props.get("body")
        
        if not recipient or not subject or not body:
            return "Error: Missing required parameters (recipient, subject, or body)"
        
        # In a real implementation, you would send an actual email here
        # For this mock version, we'll just return a success message
        
        # Log the email details
        print(f"\n--- EMAIL WOULD BE SENT ---")
        print(f"To: {recipient}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print(f"-------------------------\n")
        
        return json.dumps({
            "status": "success",
            "message": f"Email to {recipient} was sent successfully",
            "details": {
                "recipient": recipient,
                "subject": subject,
                "body_preview": body[:50] + ("..." if len(body) > 50 else "")
            }
        })