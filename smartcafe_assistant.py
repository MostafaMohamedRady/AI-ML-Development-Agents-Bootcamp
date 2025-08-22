import json
import re

'''Loads and queries data from the knowledge base file cafe_data.json.'''
class ResearchAgent:
    def __init__(self, data_file):
        self.data_file = data_file
        self.data = self.load_data()

    """Load the data from the cafe_data.json file."""
    def load_data(self):
        try:
            with open(self.data_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Error: The cafe_data.json  file was not found.")
            return {}
        except json.JSONDecodeError:
            print("Error: Failed to decode cafe_data.json.")
            return {}

    """Return ingredients of the menu item."""
    def get_ingredients(self, item_name):
        menu = self.data.get('menu', {})
        item = self._find_menu_item(item_name, menu)
        if item:
            return f"{item_name} ingredients: {', '.join(item['ingredients'])}"
        return f"Sorry, I couldn't find ingredients for '{item_name}'."

    """Return nutritional info of a drink."""
    def get_nutritional_info(self, item_name):
        menu = self.data.get('menu', {})
        item = self._find_menu_item(item_name, menu)
        if item:
            nutrition = item.get('nutrition', {})
            return (f"{item_name} has {nutrition.get('calories', 'N/A')} calories and "
                    f"{nutrition.get('sugar_g', 'N/A')}g sugar.")
        return f"Sorry, no nutritional info found for '{item_name}'."

    """Return Working hours."""
    def get_working_hours(self, day):
        hours = self.data.get('opening_hours', {})
        for key in hours:
            if key.lower() == day.lower():
                return f"On {key}, we're open from {hours[key]}."
        return f"Sorry, I don't know our hours for '{day}'."

    '''return the price for an item'''
    def get_price(self, item_name):
        menu = self.data.get('menu', {})
        item = self._find_menu_item(item_name, menu)
        if item:
            price = item.get('price_usd', None)
            if price is not None:
                return f"The price of {item_name} is ${price}."
            else:
                return f"Sorry, I don't have the price information for {item}."
        return f"Sorry, I don't know the price for {item}."

    def get_available_drinks(self):
        """Return a list of available drinks."""
        drinks = self.data.get('drinks', [])
        if drinks:
            return "We offer the following drinks:\n- " + "\n- ".join(drinks)
        return "Sorry, no drinks available at the moment."

    def _find_menu_item(self, item_name, menu_dict):
        """Helper method to perform case-insensitive item lookup."""
        for name in menu_dict:
            if name.lower() == item_name.lower():
                return menu_dict[name]
        return None


'''Interacts with the user, parses questions, and displays responses.'''
class ChatBotAgent:
    def __init__(self, research_agent):
        self.research_agent = research_agent

    '''Greet the user and offer help via a command-line interface.'''
    def greet_user(self):
        print("Welcome to SmartCafe Assistant :)!")
        print("======================================")
        print("How can I Help you today?! You can ask me Like ...")
        print("\tWhat’s in a Mocha?") # Menu item ingredients
        print("\tHow many calories in Hot Chocolate?") # Nutritional info
        print("\tWhen are you open on Friday?") # Working hours
        print("\tWhat drinks do you have?") # Available drinks
        print("\thow much is mocha")  # ask about price
        print("**Type 'exit' or 'quit' to leave.")

    def start_chat(self):
        self.greet_user()
        while True:
            '''Continue the loop until the user types exit or quit.'''
            user_input = input("\nUser:: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant:: Goodbye! Come back soon")
                break

            response = self.parse_question(user_input)
            print(f"Assistant:: {response}")

    def parse_question(self, question):
        """Regex-based intent recognition."""
        # Ingredients
        match = re.search(r"what(?:'s| is)?.*in (a |an )?(?P<item>[\w\s]+)\??", question, re.IGNORECASE)
        if match:
            item = match.group('item').strip()
            return self.research_agent.get_ingredients(item)

        match = re.search(r"(what|tell me|could you|could you tell me)?(.*)in\s*(?P<item>[\w\s]+)", question, re.IGNORECASE)
        if match:
            item = match.group('item').strip()

        # Nutritional Info
        match = re.search(
            r"(how many (calories|sugar)|what are the (calories|sugar)|tell me about the (calories|sugar))\s*(in\s*)?(?P<item>[\w\s]+)",
            question, re.IGNORECASE)
        if match:
            item = match.group('item').strip()
            return self.research_agent.get_nutritional_info(item)

        # Working Hours
        match = re.search(r"(when|what time)(.*)(open|hours)(.*)(on|for)?\s*(?P<day>monday|tuesday|wednesday|thursday|friday|saturday|sunday)", question, re.IGNORECASE)
        if match:
            day = match.group('day').capitalize()
            return self.research_agent.get_working_hours(day)

        # Drinks List
        if re.search(r"what.*(drinks|beverages|menu).*have", question, re.IGNORECASE):
            return self.research_agent.get_available_drinks()

        # Handle price query
        match = re.search(r"(how much (is|does) the price of| *.cost of|price of|how much for)\s*(?P<item>[\w\s]+)",question, re.IGNORECASE)
        if match:
            item = match.group('item').strip()
            return self.research_agent.get_price(item)

        return "Sorry, I didn't understand that. Could you rephrase your question?"


def main():
    research_agent = ResearchAgent('cafe_data.json')
    chatbot_agent = ChatBotAgent(research_agent)
    chatbot_agent.start_chat()


if __name__ == "__main__":
    main()
