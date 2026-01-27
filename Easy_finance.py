#%%
# We start by creating a basic class that represents a bank account.
# This will help explain the concept of a "class" and "object."

class BankAccount:
    # The __init__ method is like a blueprint for creating objects.
    # It defines the attributes (balance) that each bank account will have.
    def __init__(self, account_holder, initial_balance):
        self.account_holder = account_holder  # Public attribute
        self.balance = initial_balance  # Public attribute

    # A method to deposit money into the account
    def deposit(self, amount):
        self.balance += amount  # Add the deposit amount to the balance

    # A method to withdraw money from the account
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount  # Subtract the withdrawal amount from the balance
        else:
            print("Insufficient funds!")

    # A method to check the current balance
    def check_balance(self):
        return self.balance

# Now, let's create a simple class for a savings account that inherits from BankAccount.

class SavingsAccount(BankAccount):
    def __init__(self, account_holder, initial_balance, interest_rate):
        super().__init__(account_holder, initial_balance)
        self.interest_rate = interest_rate  # Additional attribute for savings account

    # Method to apply interest to the balance (this is unique to savings accounts)
    def apply_interest(self):
        self.balance += self.balance * self.interest_rate

#%%
# Let's see how these classes work together.

# Creating a basic bank account for a person named "Alice"
alice_account = BankAccount("Alice", 100)
#%%
# Alice deposits $50 into her account
alice_account.deposit(50)
print(f"Alice's Balance after deposit: ${alice_account.check_balance()}")
#%%
# Alice tries to withdraw $200, which is more than her balance
alice_account.withdraw(200)  # This should trigger an "Insufficient funds!" message
#%%
# Alice withdraws $100, which she has in her account
alice_account.withdraw(100)
print(f"Alice's Balance after withdrawal: ${alice_account.check_balance()}")
#%%
# Creating a savings account for "Bob" with an initial balance of $1000 and an interest rate of 5%
bob_savings = SavingsAccount("Bob", 1000, 0.05)

# Applying interest to Bob's savings account
bob_savings.apply_interest()
print(f"Bob's Balance after interest: ${bob_savings.check_balance()}")
#%%
# Bob deposits $200 into his savings account
bob_savings.deposit(200)
print(f"Bob's Balance after deposit: ${bob_savings.check_balance()}")