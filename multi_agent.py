import functools
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from multi_agent_prompts import(
TEAM_SUPERVISOR_SYSTEM_PROMPT,
TRAVEL_AGENT_SYSTEM_PROMPT,
LANGUAGE_ASSISTANT_SYSTEM_PROMPT,
VISUALIZER_SYSTEM_PROMPT,
DESIGNER_SYSTEM_PROMPT
)
from colorama import Fore, Style
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langchain_core.runnables.config import RunnableConfig

from main import set_environment_variables
from tools import generate_image, markdown_to_pdf_file
from tools.rag_tool import setup_rag_tool, answer_query


set_environment_variables("Multi_Agent_Team")

TRAVEL_AGENT_NAME = "travel_agent"
LANGUAGE_ASSISTANT_NAME = "language_assistant"
VISUALIZER_NAME = "visualizer"
DESIGNER_NAME = "designer"

TEAM_SUPERVISOR_NAME = "team_supervisor"
RAG_AGENT_NAME = "rag_agent"

rag_chain = setup_rag_tool()

MEMBERS = [RAG_AGENT_NAME, TRAVEL_AGENT_NAME, LANGUAGE_ASSISTANT_NAME, VISUALIZER_NAME, DESIGNER_NAME, TEAM_SUPERVISOR_NAME]

OPTIONS = ["FINISH"] + MEMBERS #this is creating a list of things
#for the supervisor to choose from. if it chooses one of the members, it runs their task
#if it chooses finish, it ends it all

#TAVILY_TOOL = TavilySearchResults()
LLM = ChatOpenAI(model="gpt-3.5-turbo-0125")

#basechatmodel is the base class for all openai LLMs, you can provide any model to this
def create_agent(llm: BaseChatModel, tools: list, system_prompt: str):
    prompt_template = ChatPromptTemplate.from_messages(
        [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt_template) #builds an agent
    agent_executor = AgentExecutor(agent=agent, tools=tools) #combines the agent and the nodes #takes the agent, and the list of tools
    return agent_executor

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add] #sequence makes the messages (agent:hi human: hellow) into a list/tuple, operator.add adds to this list/tuple every new step
    next: str #name of the next agent to be called

def agent_node(state: AgentState, agent, name): #name is the string name of the agent
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]} #returns the messages :[\
    #result["output"] is the result of the agent's call
#this tells the LLM TEAM SUPERVISOR to see what function its supposed to call next


router_function_def = {
    "name": "route",
    "description": "Select the next role.", #what the function does
    "parameters": { #things the function takes as input
        "title": "routeSchema", #title doesnt matter
        "type": "object",
        "properties": {
            "next": {
                "title": "next",
                "anyOf": [
                    {"enum": OPTIONS}, #any of the options

                ],
            }
        },
        "required": ["next"],
    },
}

team_supervisor_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", TEAM_SUPERVISOR_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            "Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=", ".join(OPTIONS), members=", ".join(MEMBERS)) #.partial lets us fill in
#the two variables we already have (OPTIONS, MEMBERS), and leave out the messages part to be added later
#the ", ".join(OPTIONS) lets us create a singular string seperating each option/member by 1 coma and a space

#use vanilla langchain chain to chain our nodes together

team_supervisor_chain = (
    team_supervisor_prompt_template
    | LLM.bind_functions(functions=[router_function_def], function_call="route") #your telling the LLM to call the function by route
    | JsonOutputFunctionsParser() #we use the jsonoutputfunctions parser to extract the "next" parameter
    #, then it appendsthe next parameter to the agentstate
)
travel_agent = create_agent(llm=LLM, tools=[], system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT)
#we use functools.partial because the agent_node function needs the "state", which will only be available at
#runtime
travel_agent_node = functools.partial(
    agent_node, agent=travel_agent, name=TRAVEL_AGENT_NAME
)


language_assistant = create_agent(llm=LLM, tools=[], system_prompt=LANGUAGE_ASSISTANT_SYSTEM_PROMPT)
#we use functools.partial because the agent_node function needs the "state", which will only be available at
#runtime
language_assistant_node = functools.partial(
    agent_node, agent=language_assistant, name=LANGUAGE_ASSISTANT_NAME
)


visualizer = create_agent(llm=LLM, tools=[generate_image], system_prompt=VISUALIZER_SYSTEM_PROMPT)
#we use functools.partial because the agent_node function needs the "state", which will only be available at
#runtime
visualizer_node = functools.partial(
    agent_node, agent=visualizer, name=VISUALIZER_NAME
)


designer = create_agent(llm=LLM, tools=[markdown_to_pdf_file], system_prompt=DESIGNER_SYSTEM_PROMPT)
#we use functools.partial because the agent_node function needs the "state", which will only be available at
#runtime
designer_node = functools.partial(
    agent_node, agent=designer, name=DESIGNER_NAME
)

def rag_agent_node(state: AgentState):
    # Use the latest user message as the query
    user_message = None
    for msg in state["messages"]:
        if msg.type == "human":
            user_message = msg.content
    if not user_message:
        user_message = state["messages"][-1].content
    summary, sources = answer_query(rag_chain, user_message)
    return {"messages": [HumanMessage(content=summary, name=RAG_AGENT_NAME)]}

workflow = StateGraph(AgentState)
workflow.add_node(TRAVEL_AGENT_NAME,travel_agent_node)
workflow.add_node(LANGUAGE_ASSISTANT_NAME,language_assistant_node)
workflow.add_node(VISUALIZER_NAME,visualizer_node)
workflow.add_node(DESIGNER_NAME,designer_node)
workflow.add_node(TEAM_SUPERVISOR_NAME, team_supervisor_chain)
workflow.add_node(RAG_AGENT_NAME, rag_agent_node)

for member in MEMBERS:
    workflow.add_edge(member, TEAM_SUPERVISOR_NAME) #add a connection with each team member to the team supervisor
workflow.add_edge(DESIGNER_NAME, END) #connect designer with the END, because after designer it ends

conditional_map = {name: name for name in MEMBERS}
conditional_map["FINISH"] = DESIGNER_NAME  # Route FINISH to designer
workflow.add_conditional_edges(
    TEAM_SUPERVISOR_NAME, lambda state: state["next"], conditional_map #this looks inside agentstate's "next" key
)

#set entry point of your graph

workflow.set_entry_point(TEAM_SUPERVISOR_NAME)

travel_agent_graph = workflow.compile()

for chunk in travel_agent_graph.stream(
        {"messages": [HumanMessage(content="I want to know about Christopher Columbus.")]},
        config=RunnableConfig(recursion_limit=100)
):
    if "__end__" in chunk:
        print(f"{Fore.CYAN}Process completed. Workflow has reached END.{Style.RESET_ALL}")
        break
    print(chunk)
    print(f"{Fore.GREEN}################{Style.RESET_ALL}")
