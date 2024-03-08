def get_time():
    now = datetime.now()
    return (
        f"The current time is {now.strftime('%H:%M')}, use "
        "this information in your response"
    )

def politics():
    return (
        "Sorry, I'm not allowed to talk about politics. If you need any further assistance Please let me know."
    )


def appointment_inquiry():
    return (
        "tell me about doctor schedule"
    )


def chitchat():
    return (
        "Sorry, I'm not allowed to chitchat . If you need any further assistance Please let me know."
    )
    
def greetings():
    return(
        "Hello! Thank you for calling our Clinic. I'm AI agent , How can I assist you today?"
    )

def done_task():
    return(
        "Thank you for providing your details with us, We have booked your appointment and will share you the details on your contact number"
    )