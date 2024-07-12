# email_step.py

from zenml.steps import BaseStep
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class SendEmailStep(BaseStep):
    """
    A step that sends an email upon pipeline completion.

    Attributes:
        subject (str): The subject of the email.
        body (str): The body of the email.
    """

    def __init__(self, subject: str, body: str):
        """
        Initialize the email step with subject and body.

        Args:
            subject (str): The subject of the email.
            body (str): The body of the email.
        """
        self.subject = subject
        self.body = body

    def entrypoint(self) -> None:
        """
        The main logic for sending the email.
        """
        # E-Mail configuration
        sender_email = "Enterprise_AI@gmx.de"
        receiver_email = "fink.silas@gmx.de"
        password = "EnterpriseAI_Gruppe4"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = self.subject
        msg.attach(MIMEText(self.body, 'plain'))

        try:
            server = smtplib.SMTP('mail.gmx.de', 587)
            server.starttls()
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            server.quit()
            print("Email erfolgreich gesendet!")
        except Exception as e:
            print(f"Fehler beim Senden der Email: {e}")
