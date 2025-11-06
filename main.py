from dotenv import load_dotenv

load_dotenv()
''' The following imports are no longer needed when using create_agent() 
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
# from langchain_core.output_parsers.pydantic import PydanticOutputParser
# I have commented out the Pydantic Output Parser to use the .with_structured_output method (Easier to use and more reliable)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda '''

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

''' We no longer need format instructions because it will come from langchain'''
#from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]

''' A significant amount of code is removed for the create_agent implementation 
llm = ChatOpenAI(model="gpt-4-turbo")
strutured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")
# output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
# I have commented out the Pydantic Output Parser to use the .with_structured_output method (Easier to use and more reliable)

react_prompt_with_format_instructions=PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
#).partial(format_instructions=output_parser.get_format_instructions())
# Removed the format instruction prompt because we will leverage the AgentResponse schema object we created in schema.py
).partial(format_instructions="") '''

model = ChatOpenAI(model="gpt-4")

'''We will use a new agent variable that will leverage the 1.0 create_agent() langchain function
agent = create_react_agent(
    llm=llm,  # notice we did not use the new structured_llm object.  We only leverage that in the last step of the chain for formatting the output
    tools=tools,
    prompt=react_prompt_with_format_instructions,
) '''

agent = create_agent(
    model,
    tools=tools,
    response_format=AgentResponse,
)

''' We no longer need to create a chain or an outputparser
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x: x['output'])
# parse_output = RunnableLambda(lambda x: output_parser.parse(x))
# Commented out this since formating will be done by the structured_llm object that we will replace in the chain below
# chain = agent_executor| extract_output| parse_output 
chain = agent_executor| extract_output| strutured_llm  '''


def main():
    #print("Hello from langchain-course!")
    '''The invoke method is not done a bit differently using the agent object rather than the chain object
    result = chain.invoke(
        input={
            "input": "search for 3 job postings for an ai engineer using langchain in California on linkdin and list their details",
        }
    )'''
    result = agent.invoke(
        {
            "messages": [
                {
                    "role":"user",
                    "content":"search for 3 job postings for an AI Engineer or Developer using langchain in the US East Coast on linkdin and list their details",
                }
            ]
        }
    )
    
    #print(result)
    print(result["structured_response"])


if __name__ == "__main__":
    main()
