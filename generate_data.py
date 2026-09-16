"""Generates data/sms_data.csv with labeled spam/ham messages."""
import csv
import random

random.seed(42)

spam_templates = [
    "WINNER!! You have been selected to receive a $1000 cash prize. Call now to claim!",
    "URGENT! Your account has been suspended. Click here to verify your details immediately.",
    "Congratulations! You've won a free iPhone. Claim your prize now at bit.ly/claim",
    "FREE entry into our $5000 weekly draw. Text WIN to 80086 now!",
    "You have 1 new voicemail. Press 9 to listen or visit http://voicemail-alert.com",
    "Limited time offer! Get 90% off on all products. Buy now before it's gone!",
    "Your loan of $10,000 has been approved. Reply YES to receive funds today.",
    "Hot singles in your area want to meet you tonight! Click here now.",
    "Congrats! You are eligible for a free credit card upgrade. Call 1800-555-0199 now.",
    "Claim your free vacation package to Bahamas! Limited slots available, act fast!",
    "URGENT: Your bank account will be locked. Verify your PIN immediately at this link.",
    "You've been chosen for a cash reward of $500. Text CASH to 12345 to redeem.",
    "Make $5000 per week working from home! No experience needed, sign up now!",
    "Your package could not be delivered. Click here to reschedule and pay a small fee.",
    "Free ringtones! Text TONE to 88888 and get unlimited downloads today.",
    "Act now! Your subscription is about to expire, renew with 70% discount instantly.",
    "Congratulations, you have won a lottery of $1,000,000! Send your details to claim.",
    "Hurry! Only 3 hours left to claim your free gift card worth $100.",
    "Your computer has a virus. Call our support team immediately at this toll free number.",
    "Special offer just for you: buy one get one free on all items, click link now.",
]

ham_templates = [
    "Hey, are we still meeting for lunch tomorrow at noon?",
    "Can you send me the report before the end of the day?",
    "Happy birthday! Hope you have a wonderful day.",
    "Don't forget to pick up milk on your way home.",
    "The meeting has been rescheduled to 3 PM on Thursday.",
    "Thanks for helping me move last weekend, I really appreciate it.",
    "I'll be a few minutes late, traffic is heavy today.",
    "Did you finish watching that show we talked about?",
    "Let's catch up this weekend, it's been a while.",
    "Please review the attached document and let me know your thoughts.",
    "Mom said dinner is ready, come downstairs.",
    "Great job on the presentation today, the client loved it.",
    "Can you pick up the kids from school today?",
    "I booked the flight tickets for our trip in December.",
    "Reminder: your dentist appointment is tomorrow at 10 AM.",
    "Let me know if you need anything from the grocery store.",
    "The project deadline has been moved to next Monday.",
    "Thanks for the quick reply, I'll get back to you soon.",
    "Are you free for a call this afternoon to discuss the budget?",
    "It was great catching up with you at the reunion last night.",
]

spam_variants = [
    "", " Reply STOP to opt out.", " Limited time only!", " Don't miss this chance!",
    " T&C apply.", " Call now!!!", " 100% guaranteed.",
]
ham_variants = [
    "", " Let me know.", " Thanks!", " See you soon.", " Talk later.", " Take care.",
]

rows = []
for i in range(150):
    t = random.choice(spam_templates)
    v = random.choice(spam_variants)
    rows.append(("spam", t + v))
for i in range(150):
    t = random.choice(ham_templates)
    v = random.choice(ham_variants)
    rows.append(("ham", t + v))

random.shuffle(rows)

with open("data/sms_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "text"])
    writer.writerows(rows)

print(f"Generated {len(rows)} rows -> data/sms_data.csv")
