from pydoc import describe
from tabnanny import verbose
from typing import Any

import pandas as pd

from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_experimental.llms.anthropic_functions import prompt
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor, AgentType
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent, create_python_agent
import qrcode

load_dotenv()

def main():
    print("Start...")

    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """

    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tools=tools,
    )

    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    csv_agent_executor: AgentExecutor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        path="episode_info.csv",
        allow_dangerous_code=True,
        verbose=True
    )

    ######################################### Router Grand Agent ############################################

    def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    tools = [
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""useful when you need to transform natural language to python and execute the python code,
                            returning the results of the code execution
                            DOES NOT ACCEPT CODE AD INPUT""",
        ),
        Tool(
            name="CSV Agent",
            func=csv_agent_executor.invoke,
            description="""useful when you need to answer question over episode_info.csv file,
                            takes an input the entire question and returns the answer after running pandas calculation""",
        ),
    ]

    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tools=tools,
    )

    grand_agent_executor = AgentExecutor(agent=grand_agent, tools=tools, verbose=True)

    print(
        grand_agent_executor.invoke(
            {
                "input": "which season has the most episodes?",
            }
        )
    )

    print(
        grand_agent_executor.invoke(
            {
                "input": "Generate and save in current working directory 15 qrcodes that point to `www.udemy.com/course/langchain`",
            }
        )
    )

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    #
    # agent_executor.invoke(
    #     input={
    #         "input": """generate and save in current working directory 15 QRcodes
    #                                 that point to www.udemy.com/course/langchain, you have qrcode package installed already"""
    #     }
    # )
    #
    # csv_agent = create_csv_agent(
    #     llm=ChatOllama(temperature=0, model="llama3"),
    #     path="/Users/baeksohyeon/code-interpreter/episode_info.csv",
    #     verbose=True,
    # )
    #
    # csv_agent.invoke(
    #     input={"input": "how many columns are there in file episode_info.csv"}
    # )
    # csv_agent.invoke(
    #     input={
    #         "input": "print the seasons by ascending order of the number of episodes they have"
    #     }
    # )

def main2():
    print("Start...")
    python_agent_executor = create_python_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        # verbose=True,
    )

    csv_agent = create_csv_agent(
        llm=ChatOpenAI(temperature=0,model="gpt-3.5-turbo"),
        path="/Users/baeksohyeon/code-interpreter/episode_info.csv",
        verbose=True,
        # handle_parsing_errors=True,
        allow_dangerous_code=True
    )
    csv_agent.invoke(
        input={"input": "how many columns are there in file episode_info.csv"}
    )
    csv_agent.invoke(
        input={
            "input": "print the seasons by ascending order of the number of episodes they have"
        }
    )


if __name__ == "__main__":
    main2()