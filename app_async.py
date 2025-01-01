import autogen
import panel as pn
import asyncio

llm_config_list = [
    {
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    }
] 

# Global configuration for the LLM with fixed temperature and seed
llm_global_config = {
    "config_list": llm_config_list, 
    "temperature": 0,  # Zero temperature for deterministic outputs
    "seed": 53        # Fixed seed for reproducibility
}

# Future object for handling asynchronous input
input_future = None

class MyConversableAgent(autogen.ConversableAgent):
    async def a_get_human_input(self, prompt: str) -> str:
        global input_future
        print('AGET!!!!!!')  # Display the prompt
        chat_interface.send(prompt, user="System", respond=False)
        
        # Create a new Future object for this input operation if none exists
        if input_future is None or input_future.done():
            input_future = asyncio.Future()

        # Wait for the callback to set a result on the future
        await input_future

        # Extract the value and reset the future for the next input operation
        input_value = input_future.result()
        input_future = None
        return input_value

# Initialize the user proxy agent
user_proxy = MyConversableAgent(
    name="Admin",
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("exit"),
    system_message="""A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.""",
    code_execution_config=False,
    human_input_mode="ALWAYS",
)

# Initialize the engineer agent
engineer_agent = autogen.AssistantAgent(
    name="Engineer",
    human_input_mode="NEVER",
    llm_config=llm_global_config,
    system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
    Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
''',
)

# Initialize the scientist agent
scientist_agent = autogen.AssistantAgent(
    name="Scientist",
    human_input_mode="NEVER",
    llm_config=llm_global_config,
    system_message="""Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code."""
)

# Initialize the planner agent
planner_agent = autogen.AssistantAgent(
    name="Planner",
    human_input_mode="NEVER",
    system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
    The plan may involve an engineer who can write code and a scientist who doesn't write code.
    Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
''',
    llm_config=llm_global_config,
)

# Initialize the executor agent
executor_agent = autogen.UserProxyAgent(
    name="Executor",
    system_message="Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper","use_docker": False},
)

# Initialize the critic agent
critic_agent = autogen.AssistantAgent(
    name="Critic",
    system_message="""Critic. Double check plan, claims, code from other agents and provide feedback. 
    Check whether the plan includes adding verifiable info such as source URL.""",
    llm_config=llm_global_config,
    human_input_mode="NEVER",
)

# Set up group chat with all agents
group_chat = autogen.GroupChat(agents=[user_proxy, engineer_agent, scientist_agent, planner_agent, executor_agent, critic_agent], messages=[], max_round=20)

# Initialize the group chat manager
manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_global_config)

# Define avatars for visual identification in chat
agent_avatars = {
    user_proxy.name: "👨‍💼",
    engineer_agent.name: "👩‍💻",
    scientist_agent.name: "👩‍🔬",
    planner_agent.name: "🗓",
    executor_agent.name: "🛠",
    critic_agent.name: '📝'
}

def print_messages(recipient, messages, sender, config):
    """
    Print messages sent between agents and update the chat interface.
    
    Args:
        recipient: The agent receiving the message
        messages: List of message objects
        sender: The agent sending the message
        config: Configuration dictionary
    
    Returns:
        tuple: (False, None) to continue agent communication flow
    """
    print(f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")

    content = messages[-1]['content']

    # Check if message has required attributes and send to appropriate user
    if all(key in messages[-1] for key in ['name']):
        chat_interface.send(content, user=messages[-1]['name'], avatar=agent_avatars[messages[-1]['name']], respond=False)
    else:
        chat_interface.send(content, user=recipient.name, avatar=agent_avatars[recipient.name], respond=False)
    
    return False, None  # Required to ensure the agent communication flow continues

# Register message handlers for each agent
user_proxy.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
engineer_agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
scientist_agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
planner_agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
executor_agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})
critic_agent.register_reply([autogen.Agent, None], reply_func=print_messages, config={"callback": None})

# Initialize Panel extension
pn.extension(design="material")

# Flag to indicate if the chat initiation task has been created
initiate_chat_task_created = False

async def delayed_initiate_chat(agent, recipient, message):
    global initiate_chat_task_created
    # Indicate that the task has been created
    initiate_chat_task_created = True

    # Wait for 2 seconds
    await asyncio.sleep(2)

    # Now initiate the chat
    await agent.a_initiate_chat(recipient, message=message)

async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    global initiate_chat_task_created
    global input_future

    if not initiate_chat_task_created:
        asyncio.create_task(delayed_initiate_chat(user_proxy, manager, contents))
    else:
        if input_future and not input_future.done():
            input_future.set_result(contents)
        else:
            print("There is currently no input being awaited.")

# Initialize the chat interface
chat_interface = pn.chat.ChatInterface(callback=callback)
chat_interface.send("Send a message!", user="System", respond=False)
chat_interface.servable()