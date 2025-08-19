# mailer.py
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.header import Header

def send_mail_html(
    smtp_user: str,
    app_password: str,
    to_addrs: list[str],
    subject: str,
    html: str,
    from_name: str = "AutoStock Bot",
    attachments: list[str] | None = None,
):
    """
    Gmail SMTP로 HTML 메일 전송 (+첨부파일 지원)
    """
    msg = MIMEMultipart("mixed")
    msg["From"] = str(Header(f"{from_name} <{smtp_user}>", "utf-8"))
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = Header(subject, "utf-8")

    # HTML 본문
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(html, "html", "utf-8"))
    msg.attach(alt)

    # 첨부파일
    if attachments:
        for path in attachments:
            try:
                with open(path, "rb") as f:
                    part = MIMEApplication(f.read())
                filename = path.split("/")[-1]
                part.add_header("Content-Disposition", "attachment", filename=filename)
                msg.attach(part)
            except Exception as e:
                print(f"[WARN] attachment failed: {path} ({e})")

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(smtp_user, app_password)
        smtp.sendmail(smtp_user, to_addrs, msg.as_string())
