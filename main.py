from dotenv import load_dotenv
import os
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.tools.render import render_text_description
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.agents.format_scratchpad.log import format_log_to_str
from langchain_core.agents import AgentAction, AgentFinish

load_dotenv()  # reads .env into environment


@tool
def get_text_length(text: str) -> int:
    """Return the length of a text by characters"""
    print("I made it into the function")
    print(f"get_text_length called with {text}")
    text = text.strip("\n").strip('"')  # Stripping away non alpha characters
    return len(text)


def find_tool_by_name(tools: list[tool], tool_name: str) -> tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


def main():
    print("Hello from langchain-course-section4!")
    # print(get_text_length("Rufino"))
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )
    # llm = ChatOpenAI(temperature=0, model_kwargs={"stop":["\nObservation","Observation","Observation:"]})
    llm = ChatOpenAI(
        temperature=0, stop=["\nObservation", "Observation", "Observation:"]
    )
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # res = agent.invoke({"input": "What is the length of 'RUFINO' in characters?"})
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of RUFINO in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )

    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")
        intermediate_steps.append((agent_step, str(observation)))

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
        {
            "input": "What is the length of RUFINO in characters?",
            "agent_scratchpad": intermediate_steps,
        }
    )
    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)


if __name__ == "__main__":
    main()
