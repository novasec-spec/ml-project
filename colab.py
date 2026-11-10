transactions = {
        "date": "2026-10-1",
        "description": "lunch",
        "category": "food",
        "amount": 250

}
print(transactions)
print(transactions["description"])



total = 0
for transaction in transactions:
    total = total + transaction["amount"]

print("Total spent:", total)
