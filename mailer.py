import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

def send_mail_html(smtp_user: str, app_password: str, to_addrs: list[str], subject: str, html: str, from_name: str = "AutoStock Bot"):
    msg = MIMEText(html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = formataddr((from_name, smtp_user))
    msg["To"] = ", ".join(to_addrs)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(smtp_user, app_password)
        smtp.sendmail(smtp_user, to_addrs, msg.as_string())
