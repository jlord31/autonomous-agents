from typing import Dict, Any, Optional
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email(
    to_email: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None,
    cc: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate sending an email without actually contacting an SMTP server.

    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Email body content
        from_email: Sender email address (defaults to SMTP_USER from environment if not provided)
        cc: Optional CC recipients

    Returns:
        Dictionary with simulation status and details.
    """
    # Use provided from_email or default to SMTP_USER from the environment
    smtp_user = os.environ.get('SMTP_USER', 'default@example.com')
    sender = from_email if from_email else smtp_user

    # Determine recipients list
    recipients = [to_email]
    if cc:
        recipients += [addr.strip() for addr in cc.split(',')]

    # Log/Print email details to simulate sending
    print("Simulated Email Details:")
    print("From:", sender)
    print("To:", to_email)
    if cc:
        print("Cc:", cc)
    print("Subject:", subject)
    print("Body:", body)

    return {
        "success": True,
        "message": f"Email simulation complete. (No actual email sent to {to_email})",
        "details": {
            "subject": subject,
            "recipients": recipients
        }
    }