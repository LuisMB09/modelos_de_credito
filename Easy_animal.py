# First, let's create a basic class that represents an Animal.
# This will serve as a blueprint for all animals.

class Animal:
    # The __init__ method initializes the attributes of an animal.
    def __init__(self, name, species):
        self.name = name  # Public attribute
        self.species = species  # Public attribute

    # A method to make the animal speak
    def speak(self):
        return f"{self.name} makes a sound."

    # A method to describe the animal
    def describe(self):
        return f"{self.name} is a {self.species}."

# Now, let's create specific types of animals that inherit from the Animal class.

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Calling the parent class's constructor
        self.breed = breed  # Additional attribute for dogs

    # Overriding the speak method
    def speak(self):
        return f"{self.name} barks."

class Cat(Animal):
    def __init__(self, name, color):
        super().__init__(name, "Cat")  # Calling the parent class's constructor
        self.color = color  # Additional attribute for cats

    # Overriding the speak method
    def speak(self):
        return f"{self.name} meows."

# Let's see how these classes work together.

# Creating a dog named "Buddy" who is a Golden Retriever
buddy = Dog("Buddy", "Golden Retriever")
print(buddy.describe())  # "Buddy is a Dog."
print(buddy.speak())  # "Buddy barks."

# Creating a cat named "Whiskers" who is gray
whiskers = Cat("Whiskers", "Gray")
print(whiskers.describe())  # "Whiskers is a Cat."
print(whiskers.speak())  # "Whiskers meows."