from semantic_router import Route

# time_route = Route(
#     name="get_time",
#     utterances=[
#         "what time is it?",
#         "when should I eat my next meal?",
#         "how long should I rest until training again?",
#         "when should I go to the gym?",
#     ],
# )

politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

chitchat = Route(
    name="greetings",
    utterances=[
    " Who is this?",
    "Is this a clinic",
    "May I speak to the doctor"
        "Hello Good morning",
        "Hello Good afternoon",
        "Hello Good evening",
        "Hello"
        "How are you doing",
    ],
)

chitchat = Route(
    name="done_task",
    utterances=[
    "My name is pooja and contact number is 9780032269 from jagatpurate",
    "Myself Pooja and contact number is 1234567890 from jaipur"
    
    ],
)


appointment_route = Route(
    name="appointment_inquiry",
    utterances=[
     "Can I schedule an appointment with doctor?",
    "I'd like to book a consultation with a doctor.",
    "How can I make an appointment for a medical check-up?",
    "I need to see a doctor. What are my options?",
    "I'm feeling unwell and would like to see a physician.",
    "Is it possible to schedule a telemedicine appointment?",
    "Do you have any available slots for a doctor's visit this week?",
    "I'd like to set up a follow-up appointment with my specialist.",
    "Can I book a same-day appointment?"
   
    ],
)



routes = [politics,chitchat,appointment_route]


