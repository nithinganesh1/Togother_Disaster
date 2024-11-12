import firebase_admin
from firebase_admin import credentials, db
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time  # To add delay between checks

# Initialize the Firebase Admin SDK
cred = credentials.Certificate('C:/Users/HP/Evolve/Disaster_Projects/HostSystem/disaster-39e70-firebase-adminsdk-uphjt-762f35dd15.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://disaster-39e70-default-rtdb.firebaseio.com'
})

# Function to send an email
def send_email():
    # Email sender and receiver details
    sender_email = "evolvenithin.kochi@gmail.com"
    
    # List of receiver emails
    receiver_emails = ["evolvedani.kochi@gmail.com", "sahilevolve.kochi@gmail.com"]
    password = "bldz cvgu uzjk dabp"

    # Create the email content
    subject = "Earthquake Alert!"
    body = '''
An earthquake has been detected by our system. We urge you to issue a public safety alert immediately and advise people to take necessary precautions.

Please instruct the public to:

1. Move to safe, open areas.
2. Take cover indoors under sturdy furniture.
3. Avoid using elevators and be prepared for aftershocks.'''

    # Setup the MIME
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(receiver_emails)  # Join all recipient emails as a single string
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Connect to the SMTP server
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)  # For Gmail
        server.starttls()  # Start TLS for security
        server.login(sender_email, password)  # Login to the email server

        # Send the email to all recipients at once
        server.sendmail(sender_email, receiver_emails, message.as_string())  # Send the email

        server.quit()  # Close the connection
        print("Email sent successfully to all recipients")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Function to monitor Earth value
def monitor_earth_value():
    last_earth_value = None  # To track the previous value of Earth
    
    while True:  # Infinite loop for continuous checking
        ref = db.reference('Earth')  # Reference to the "Earth" value in your database
        earth_value = ref.get()  # Get the current Earth value
        
        # If the value of Earth changes to 1, send an email
        if earth_value == 1 and last_earth_value != 1:
            print("Earth value is 1. Sending alert email.")
            send_email()
        
        # Update the last_earth_value to the current value
        last_earth_value = earth_value

        time.sleep(5)  # Wait for 5 seconds before checking again (you can adjust this)

# Start monitoring the Earth value
monitor_earth_value()
