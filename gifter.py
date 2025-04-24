# from dotenv import load_dotenv
from llama_index.core.workflow import Context

# load_dotenv()

# from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent, AgentOutput, ToolCallResult, ToolCall
from llama_index.llms.ollama import Ollama

import logging
import sys

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

gifts = ""
def get_upcoming_event() -> dict:
    """Gets detail about the next upcoming event and details about the person for whom the event"""
    print("Called Get upcoming event")
    return {
        "event": "Daugher's Birthday",
        # "person": ""
        "age": "4",
        "gift preferences": "soft toys, lego",
    }

    # return f"Gifts needs to be planned for {person} for Birthday whose age is {age} and likes {preferences}"


def search_gift(gift) -> str:
    """Searches and checks the availability of the gift"""
    print("### Serach gift:" + gifts + "<" + gift)
    return "gift not found"
    # return f"""{gifts} not found"""
    # return f"""
    #     {gifts} is found.
    # """


async def order_gift(gift) -> str:
    """Orders the gift and finalize the purchase"""
    print("### order_gift gift:" + gifts + "<" + gift)
    return f"""
        {gifts} ordered successfully
    """


# llm = Ollama(model="llama3.2", request_timeout=120.0, system_prompt="Use tools when necessary.")
llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)


async def store_gift(ctx: Context, gift: str) -> str:
    """Useful tp store gift for future reference."""
    print("----store_gift")
    current_state = await ctx.get("state")
    if "gift" not in current_state:
        current_state["gift"] = {}
    current_state["gift"] = gift
    global  gifts
    gifts = gift
    await ctx.set("state", current_state)
    return "Gifts stored."


gift_finder = FunctionAgent(
    name="GiftFinder",
    description="Useful for finding the upcoming event and to find the correct gift.",
    tools=[get_upcoming_event, store_gift, search_gift, order_gift],
    llm=llm,

#     system_prompt="""
#     You are an AI-powered gift planning agent. Follow these steps in strict order
#  Do not skip any step. Do not reorder steps. Follow the process exactly:
# Step 1. Use the "get_upcoming_event" tool to find the next upcoming event.
# Step 2. Based on the event details and gift preferences, select the best gift.
#    - The gift should be at most two words (e.g., "soft toy", "lego set").
# Step 3. Store the gift for future reference using "store_gift".
# Step 4. Use the "search_gift" tool to check if the gift is available.
#     """,

    system_prompt= """
    You are a structured gift-planning agent. Follow these steps **in order**:

Step 1. Call "get_upcoming_event" to find the next upcoming event.
   - Extract details like event name, person’s age, and gift preferences.

Step 2. Choose the best gift based on the preferences.  
   - The gift must be two words maximum (e.g., "soft toy", "lego set").

Step 3. Call "store_gift" tool with the selected gift.

Step 4.Only AFTER storing, call "search_gift" tool to check availability.
   - If the gift is found, proceed to step 5.
   - If the gift is not found, inform the user with the message "Gift not found" and exit

Step 5.Call "order_gift" tool to finalize the purchase.

 Do not call "search_gift" or "order_gift" before storing the gift.
 Execute the steps exactly in order.**

    """

#
#
#    - If the gift is found, order it using "order_gift".
#    - Otherwise, inform the user that the gift is not available.
    # system_prompt="""
    # You are an agent who will find the next upcoming event using "get_upcoming_event" tool
    # Based on the event and the preference for the person. Can you find the best gift that could surprise the person
    # The gift should be max of two words representing keywords which can be used
    # to order gift in ecommerce site.
    # The gift should be stored for future reference using "store_gift" tool.
    # After gift is stored use "search_gift" tool to find the gift
    # If gift found use "order_gift" tool to order the gift
    # Else tell the user gift not found
    # """,

    # Once the the gift is stored. You need to handoff control to "ShopFinder" agent so that he can order the gift without any user input
    # can_handoff_to=["ShopFinder"],
)

shop_finder = FunctionAgent(
    name="ShopFinder",
    description="Finds and orders the best gift in a ecommerce site.",
    tools=[search_gift, order_gift],
    llm=llm,
    system_prompt="""
    You are an agent who will search and find the best gift item based on highest user rating using "search_gift" tool.
    Once the gift item is found using "order_gift" tool to order the gift.
    """,
)

agent_workflow = AgentWorkflow(
    agents=[gift_finder, shop_finder],
    root_agent=gift_finder.name,
    initial_state={
        "gift": "",
    },
)


async def main():
    # handler = agent_workflow.run(system="""
    #     When you start do the the following steps
    #     Step1: Find the upcoming event and identify the best gift for the person
    #     Step2: Search and find the gift in ecommerce site and place the order
    # """, user_msg="What is the plan for today?")

    handler = agent_workflow.run(
        #     system="""
        #     When you start do the the following steps
        #     Step1: Find the upcoming event and identify the best gift for the person
        #      Step2: Search and find the gift in ecommerce site and place the order
        # """,
        user_msg="""Tell me if there is any upcoming event. 
                   And can you find the best gift for the person
                   Once you find the gift can you order it in ecommerce site
                 
                 """)
    #
    # handler = gift_finder.run(user_msg="""
    #     What is the upcoming event?
    # """)

    current_agent = None
    current_tool_calls = ""
    async for event in handler.stream_events():
        if (
                hasattr(event, "current_agent_name")
                and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}")
            print(f"🤖 Agent: {current_agent}")
            print(f"{'=' * 50}\n")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("📤 Output:", event.response.content)
            if event.tool_calls:
                print(
                    "🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                )
        elif isinstance(event, ToolCallResult):
            print(f"🔧 Tool Result ({event.tool_name}):")
            print(f"  Arguments: {event.tool_kwargs}")
            print(f"  Output: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"🔨 Calling Tool: {event.tool_name}")
            print(f"  With arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

