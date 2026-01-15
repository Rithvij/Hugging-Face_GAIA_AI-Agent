# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType

# Defining the AnswerExcelTool class which extends BaseTool
class AnswerExcelTool(BaseTool):
    name : str = "answer_excel_tool"
    description: str = "Given the path to a file containing an excel file and a query, this tool tries to get an answer by querying the excel file. Provide the whole question in input. Another agent will later break down the task."

    def _run(self, query: str, file_path: str) -> str:
        # Method to run the tool, using a query and the file path to an Excel file
        df = pd.read_excel(file_path)  # Reading the Excel file into a DataFrame

        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # Configuring the LLM

        agent_executor = create_pandas_dataframe_agent(
            # Creating a Pandas DataFrame agent with the LLM and DataFrame
            llm,
            df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True  # IMPORTANT: Understand the risks
        )

        return agent_executor(query)  # Executing the query using the agent

# Importing necessary libraries and modules
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr
from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import time
from openai import OpenAI

# Defining the AnswerQuestionTool class which extends BaseTool
class AnswerQuestionTool(BaseTool):
    name : str = "answer_question_tool"
    description: str = "Use this tool to answer any elementary question that you can solve without needing access to any external tool. Simply provide the question in input, reporting the whole question including desired output format. You can use this tool for example for vegetable classification."
    _llm = PrivateAttr()
    _system_prompt = PrivateAttr()

    def __init__(self):
        # Initializing the AnswerQuestionTool
        super().__init__()
        #self._llm = ChatGoogleGenerativeAI(
        #    model="gemini-2.0-flash",
        #    temperature=0)
        #self._llm = ChatOpenAI(model="o4-mini", temperature=0)


        self._system_prompt = SystemMessage("""You are a helpful assistant.
                                            You will be given a question and you will have to answer that question.
                                            Provide also the reasoning for your answer as well as your final answer.

                                            When provided with a list you must stick with the exact terms provided in the list and not make any modification.
                                            Green beans, corn and zucchini are NOT VEGEATABLES BOTANICALLY!
                                            Let's think step by step.
                                            """)
        
    def _run(self, question: str) -> str:
        # Method to run the tool and get an answer for the given question
        human_message = HumanMessage(
            # Creating a human message with the question content
            content=[
                {"type": "text", "text": question},
            ]
        )
    
        time.sleep(5)  # Adding a delay for rate limits
        client = OpenAI()  # Initializing the OpenAI client
        response = client.responses.create(
            # Creating a response using OpenAI's API
            model="o4-mini",
            messages = [
                {
                    "role": "system", "content": self._system_prompt.text()
                },
                {
                    "role": "user", "content": question
                }]
            )
        #response = self._llm.invoke([self._system_prompt, human_message])
    
        return response  # Returning the response from the OpenAI API

# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import PrivateAttr
import os
from dotenv import load_dotenv
import whisper
import base64

load_dotenv(".env", override=True)  # Loading environment variables

# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")  # Fetching Azure OpenAI endpoint from environment
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION_GEN", "2023-12-01-preview") # Default API version
# # AZURE_OPENAI_DEPLOYMENT_NAME will be used as the 'model' for API calls
# AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"


# Defining the AnswerQuestionFromFileTool class which extends BaseTool
class AnswerQuestionFromFileTool(BaseTool):
    name: str = "answer_question_from_file_tool"
    description: str = """
        This tool allows you to answer a question taking into account information that were provided inside a file. 
        You must provide the file in b64 when processing here.

        Args:
            The question that needs to be answered.
            The file extension of the file that is being processed.
        """
    _llm = PrivateAttr()

    def __init__(self):
        # Initializing the AnswerQuestionFromFileTool
        super().__init__()
        self._llm = ChatGoogleGenerativeAI(  # Setting up the LLM with specific parameters
            model="gemini-2.0-flash",
            temperature=0)


    def _run(self, question: str, file_name: str, file_extension: str) -> str:

        with open(file_name, "rb") as f:
            file = f.read()

        if file_extension in ["png", "jpg"]:
            encoded_file = base64.b64encode(file).decode("utf-8")

            message = {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_file}"}
        elif file_extension == "pdf":
            encoded_file = base64.b64encode(file).decode("utf-8")
            message = {"type": "image_url", 
                    "image_url": f"data:application/pdf;base64,{encoded_file}"
                  }
        else:
            message = {"type": "text", "text": "The file is not supported."}

        message_local = HumanMessage(
            content=[
                {"type": "text", "text": question + "\nLet's think step by step."},
                message,
            ]
        )

        response = self._llm.invoke([message_local])

        return response

# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from pathlib import Path
import os
from transformers import pipeline
import torch
from langchain_openai import ChatOpenAI
import time

# Defining the AudioTool class which extends BaseTool
class AudioTool(BaseTool):
    name : str = "answer_question_audio_tool"
    description: str = "This tool will reply to a query based on the audio given the path of a locally stored file. This file DOES NOT DOWNLOAD the file from the web. Run the download_file_tool first" 

    def _run(self, query: str, file_path: str) -> str:
        # Method to transcribe the provided audio file and answer the query using LLM
        try:
            #pipe = pipeline(
            #    task="automatic-speech-recognition",
            #    model="openai/whisper-base",
            #    torch_dtype=torch.float32,
            #    device=0,
            #    return_timestamps=True
            #)         
            #result = pipe(str(Path("./") / Path(file_path)), return_timestamps=True)
            model = whisper.load_model("base")
            result = model.transcribe(audio=str(Path("./") / Path(file_path)), language='en')  # Transcribing the audio using Whisper model
        except Exception as e:
            print("Exception", e)

        print(result["text"])

        human_message = HumanMessage([{"type": "text", "text": query},
            {"type": "text", "text": f"\n\nTranscript: {result['text']}"}])

        system_message = SystemMessage("""You are a helpful assistant. Whenever you receive a transcript of an audio recording along with a user's query:

        1. Carefully read the query multiple times to ensure you fully grasp what is being asked.
        
        2. Start by thinking, in clear bullet points, each precise requirement implied by the user's instructions (e.g., which portions of the transcript to use, what to include or exclude, and any specific formatting). 
        
        3. After thinking more about the requirements, fulfill the request exactly as specified. Follow all content and formatting rules without deviation (for instance, “list only names,” “omit quantities,” “use comma-separated values,” “alphabetize,” etc.). 
        
        4. Ensure that your final answer adheres strictly to the user's criteria and contains nothing beyond what was requested.
        
        Always prioritize accuracy and strict adherence to the user's stated needs before providing the answer. REPLY ONLY WITH WHAT THE HUMAN ASKED. Return only the final answer!""")
        
        time.sleep(5)
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        
        response = llm.invoke([system_message, human_message])  # Getting the response from the LLM

        return response  # Returning the response from the LLM

# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from chessimg2pos import predict_fen
from stockfish import Stockfish
import chess

# Defining the ChessTool class which extends BaseTool
class ChessTool(BaseTool):
    name : str = "chess_tool"
    description : str = "Given the path of an image, this tool returns the best next move that can be done on the chessboard. You must give ONLY the PATH of the image here! Pass in input b or w as color_turn based on whose turn is it. Use w if unspecified."

    def _run(self, img_path: str, color_turn: str) -> str:
        # Method to analyze the chessboard image and return the best move
        # Get the FEN string
        fen = predict_fen("./downloaded_files/image.png")  # Predicting the FEN string from the chessboard image

        # The fen predicted is always with a1 at the bottom left.
        # If it's black turn than the bottom left is h8, you need to reverse the positions retrieved.
        if color_turn == "b":
            ranks = fen.split('/')
            rotated_matrix = []
            for old_row in reversed(ranks):
                rotated_matrix.append(list(reversed(old_row)))
            final_fen = "/".join(["".join(row) for row in rotated_matrix])
            for length in reversed(range(2, 9)):
                final_fen = final_fen.replace(length * "1", str(length))
        else:
            final_fen = fen

        fen = f"{final_fen} {color_turn} - - 0 1"

        try:
            # Initializing Stockfish chess engine
            stockfish = Stockfish(path="C:/Users/FORMAGGA/Documents/personal/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe")

            stockfish.set_fen_position(fen)

            next_move = str(stockfish.get_best_move())  # Getting the best move from Stockfish
        except Exception as e:
            print("Exception", e)
            raise e
        
        piece = stockfish.get_what_is_on_square(next_move[:2])  # Getting the piece on the starting square of the move

        next_move_fen = piece.name + next_move[2:]  # Constructing the FEN representation of the move

        return next_move_fen  # Returning the best move in FEN format

# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool, ToolException
from typing import Optional
import subprocess
import tempfile
import os
from pydantic import PrivateAttr

# Defining the PythonExecutionTool class which extends BaseTool
class PythonExecutionTool(BaseTool):
    # A LangChain “tool” that takes a string of Python code,
    # writes it to a temporary .py file, executes it in a fresh
    # Python subprocess, captures stdout/stderr, and returns the result.

    name : str = "python_execution"
    description : str = (
        "Executes a string of Python code in an isolated subprocess. "
        "Returns stdout on success, or stderr (with exit code) on failure."
    )
    _python_executable: str = PrivateAttr()
    _timeout: int = PrivateAttr()
    _temp_dir: str = PrivateAttr()

    def __init__(
        self,
        python_executable: str = "C:\\Users\\FORMAGGA\\Documents\\personal\\Final_Assignment_Template\\.venv\\Scripts\\python.exe",
        timeout: int = 5,
        *,
        temp_dir: Optional[str] = None
    ):
        
        """
        :param python_executable: Path to the Python interpreter to invoke.
        :param timeout: Maximum seconds to allow the code to run.
        :param temp_dir: Optional directory in which to create the temp file.
        """
        super().__init__()
        self._python_executable = python_executable
        self._timeout = timeout
        self._temp_dir = temp_dir

    def _run(self, code: str) -> str:
        """
        Synchronously execute the provided Python code.
        :param code: The complete Python source to run.
        :return: Captured stdout if exit code is 0; otherwise stderr + exit code.
        :raises ToolException: On internal error (e.g. unable to write temp file).
        """
        # 1. Write code to a temporary file on disk to avoid shell-quoting issues.
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".py", delete=False, dir=self._temp_dir, mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
        except Exception as e:
            raise ToolException(f"Failed to write temp file: {e!r}")

        # 2. Invoke a fresh Python process on that file, capturing stdout & stderr.
        try:
            result = subprocess.run(
                [self._python_executable, "-u", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            return f"⚠️ Execution timed out after {self._timeout} seconds."
        except Exception as e:
            raise ToolException(f"Failed to launch subprocess: {e!r}")
        finally:
            # 3. Clean up the temp file no matter what
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # 4. Process the result
        if result.returncode != 0:
            return (
                f"❌ Process exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr}"
            )
        return result.stdout 


from langchain_core.tools.base import BaseTool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv(".env", override=True)

# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT_GEN")
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY_GEN")
# OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION_GEN", "2023-12-01-preview") # Default API version
# # AZURE_OPENAI_DEPLOYMENT_NAME will be used as the 'model' for API calls
# AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4.1"

class CodeGenTool(BaseTool):
    name : str = "code_generator_tool"
    description: str = "Given the instructions provided, it generates Python code as text. It's important that the instructions provide: which args must be provided in input, the content of the function and what is the desired output."

    def _run(self, function_description: str, input: str, output: str) -> str:
        if not function_description:
            return "You need to pass in a function description. Retry providing the right parameters."

        system = SystemMessage("""You are an expert software engineer, your goal is to generate a piece of code.
                               YOU MUST GENERATE A **PYTHON** FUNCTION. 
                               You will be given a description of what the function needs to do, for example "Generate a function that retrieves a web page from the internet".
                               Then you will be given information about what the input parameters are and the output.

                               In the output code you must list the imports as well.
                               It's VERY IMPORTANT that you stick to the contraints given for input and output.
                               If you believe there is a better way to do things, IGNORE THIS IDEA and stick to what is given in input.
                                """)
        
        human = HumanMessage(f"Description of the function:\n{function_description}\n\nInput parameters:\n{input}\n\nOutput result:\n{output}\n\n")

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5)
        
        response = llm.invoke([system, human])

        return response

from langchain_core.tools.base import BaseTool
import requests
import base64
import pandas as pd
import os
import tempfile
import whisper

class DownloadFile(BaseTool):
    name : str = "download_file_tool"
    description: str = """
        This tool downloads a file (image, pdf, python code, excel, etc.) given the name of the file. The url for the request will be composed in the function so ONLY the name of the file should be passed in.

        You may have to download a file in 2 different scenarios:
        - A file given already as part of the task. In this case the format of the url must be: {DEFAULT_API_URL}/files/{file_name} THE EXTENSION OF THE FILE MUST NOT(!!) BE INCLUDED!
        - A url retrieved from the internet in the format https://some_url. In that case, you simply need to provide the url of the file that needs to be retrieved.

        Args: 
            file_name: the name of the file to be retrieved DEFAULT_API_URL/files/task_id
            file_extension: the extension of the file, without the dot. So for example "pdf", "img", "py", "xlsx", etc.

        Output:
        IF the file is a document, image or audio:
        A string with the path to the file.
        
        IF the file is a piece of code:
            A dict made of:
                The text of the image

        IF the file is an excel:
            A dict made of:
            A pandas dataframe
        """

    def _run(self, file_url: str, file_extension: str) -> dict:
        response = requests.get(file_url)
        
        if response.status_code == 200:
            msg = "File downloaded successfully!!"
            if file_extension in ["png", "jpg", "pdf"]:
                file = response.content
                
                with open("downloaded_files/image.png", "wb") as f:
                    f.write(file)

                return "downloaded_files/image.png"
            elif file_extension in ["mp3", "wav"]:
                res = response.content
                with open("downloaded_files/audio.mp3", mode="wb") as f:
                    f.write(res)

                return f"./downloaded_files/audio.{file_extension}"

            elif file_extension == "py":
                return {"text": response.text}
            elif file_extension == "xlsx":
                file_name = file_url.split("/")[-1]

                with open(f"./downloaded_files/{file_name}.xlsx", "wb") as f:
                    f.write(response.content)

                return f"./downloaded_files/{file_name}.xlsx"
            else:
                return "The file extension is not valid."
        else:
            msg = "There was an error downloading the file."

            return msg

        
# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from typing import List
import requests

# Defining the FetchWebPageTool class which extends BaseTool
class FetchWebPageTool(BaseTool):
    name : str = "fetch_web_page_tool"
    description: str = "Provided the urls of 1 or more web pages, this tool returns the full content of the web page. This tool needs to be called AFTER calling the web_page_tool. It's important to fetch only pages which are useful to your task!"

    def _run(self, urls: List[str]) -> List[str]:
        # Method to fetch the full content of the provided web pages
        pages = [requests.get(url).text for url in urls]  # Fetching the content of each URL

        return pages  # Returning the fetched content of the web pages

from langchain_core.tools.base import BaseTool

class ReverseString(BaseTool):
    name: str = "reverse_string_tool"
    description: str = ("This tool inverts the order of the characters within a sentence. It is particularly useful if you can't understand the content in any language.")

    def _run(self, string: str) -> str:
        return string[::-1]

# Importing necessary libraries and modules
from langchain_core.tools.base import BaseTool
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import TavilySearchResults, DuckDuckGoSearchResults
from langchain_tavily import TavilySearch
import os
from pydantic import PrivateAttr
from langchain_community.document_loaders import WebBaseLoader
import json
import requests

load_dotenv(".env", override=True)  # Loading environment variables

# Defining the WebSearchTool class which extends BaseTool
class WebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = "Perform a web search and extract concise factual answers. The query should be concise, below 400 characters. Use for online facts not in GAIA/Wikipedia—e.g. sports stats, Olympic participation, published papers, museum specimen locations, competition winners, and other up-to-date info."
    _search: TavilySearch = PrivateAttr()

    def __init__(self):
        # Initializing the WebSearchTool
        super().__init__()
        self._search = TavilySearch(max_results=3, topic="general")  # Setting up the TavilySearch with specific parameters
    
    def _run(self, query: str) -> dict:
        # Method to run the web search tool with the given query
        search_results = []  # Initializing the list for search results
        search_results.append(self._search.run(query))  # Performing the search and adding the results to the list

        return search_results  # Returning the search results

# Importing necessary libraries and modules
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import PrivateAttr
from langchain_core.tools.base import BaseTool
from langchain_community.document_loaders import WikipediaLoader
import requests
from bs4 import BeautifulSoup
import wikipedia

# Defining the WikipediaTool class which extends BaseTool
class WikipediaTool(BaseTool):
    name: str = "wikipedia_tool"
    description: str = "Search Wikipedia for a given query, retrieving the corresponding page's HTML content. The query should not contain any noise and ask for something specific."

    def __init__(self):
        # Initializing the WikipediaTool
        super().__init__()

    def _run(self, query: str):
        # Method to run the Wikipedia tool with the given query
        print(f"wikipedia_search_html called with query='{query}'")  # Logging the query
        # Step 1: Get Wikipedia HTML
        page = wikipedia.page(query)  # Fetching the Wikipedia page for the query
        html = page.html()  # Extracting the HTML content of the page

        # Step 2: Parse HTML
        soup = BeautifulSoup(html, "html.parser")  # Parsing the HTML content
        content_div = soup.find("div", class_="mw-parser-output")  # Finding the content division
        # content_div = soup.find("table", class_="wikitable")
        if not content_div:
            return ""

        # Step 3: Find all tags to remove (style, script, sup, infobox, etc.)
        to_decompose = []  # Collecting tags to be removed
        for tag in content_div.find_all():  # Looping through all tags in the content division
            tag_classes = tag.get("class", [])
            if (
                tag.name in ["style", "script", "sup"]
                or any(cls in ["infobox", "navbox", "reference"] for cls in tag_classes)
            ):
                to_decompose.append(tag)

        # Remove them after collecting
        for tag in to_decompose:  # Decompose and remove the collected tags
            tag.decompose()
        
        return str(content_div)  # Returning the cleaned content division as string

from langchain_core.tools.base import BaseTool, ToolException
import requests
from youtube_transcript_api import YouTubeTranscriptApi
import re

class YoutubeTranscriptTool(BaseTool):
    name: str = "youtube_transcript_tool"
    description: str = "This tool can be used to retrieve the transcript of a youtube video given the FULL youtube link. You must pass the full youtube link!"

    def _run(self, youtube_link: str) -> str:
        """
        Fetch transcript for a YouTube video URL.
        Args:
            youtube_link: The full URL of the YouTube video.
        Returns:
            The transcript as a single string.
        """
        # Get the video ID from the youtube URL
        re_match = re.search(r"watch\?v=([^&]+)", youtube_link)
        if not re_match:
            raise ValueError(f"Invalid YouTube URL: {youtube_link}")
        video_id = re_match.group(1)

        # Initialize the transcriptAPI and retrieve the transcript for the given videoID
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)

        transcript = []
        for snippet in fetched_transcript:
            transcript.append(snippet.text)

        return "\n".join(transcript)

import os
import time
import requests
import gradio as gr
import pandas as pd
from contextlib import redirect_stdout
from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


# --- Load Environment Variables ---
load_dotenv(".env", override=True)

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- State TypedDict ---
class State(TypedDict):
    file_path : str
    file: Optional[str]
    parsed_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]
    parsed_file_message: dict
    question: str
    response: str

# --- Basic Agent Definition ---
class BasicAgent:
    def __init__(self):
        # Tools for the agent
        tools = [
            CodeGenTool(), PythonExecutionTool(temp_dir="./"), YoutubeTranscriptTool(), 
            AnswerQuestionFromFileTool(), AnswerQuestionTool(), DownloadFile(), 
            ReverseString(), WebSearchTool(), WikipediaTool(), AnswerExcelTool(), 
            ChessTool(), AudioTool(), FetchWebPageTool()
        ]

        # LLM Configuration
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.llm_with_tools = llm.bind_tools(tools)

        # Build state graph
        builder = StateGraph(State)
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_node("final_answer", BasicAgent.final_answer)

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition, path_map={
            "tools": "tools",
            "__end__": "final_answer"
        })
        builder.add_edge("tools", "assistant")
        builder.add_edge("final_answer", END)

        self.react_graph = builder.compile()

    def __call__(self, question: str, task_id: str, file_name: Optional[str]) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        messages = [HumanMessage(question)]
        messages = self.react_graph.invoke({
            "messages": messages,
            "file_path": file_name,
            "question": question
        })

        with open(f'messages_{task_id}.txt', 'w', encoding='utf-8') as out:
            with redirect_stdout(out):
                for m in messages['messages']:
                    m.pretty_print()

        final_answer = messages["messages"][-1].content.strip()
        print(f"Final answer is {final_answer}")
        return final_answer

    def assistant(self, state: State):
        file_name = state["file_path"].split(".")[0] if state["file_path"] else None
        file_extension = state["file_path"].split(".")[1] if state["file_path"] else None

        prompt = f"""
        You are a general AI assistant. When I ask you a question:

        Share your reasoning process clearly.

        End with the exact template:
        FINAL ANSWER: [YOUR FINAL ANSWER]

        Guidelines for FINAL ANSWER: Use a single number, minimal phrase, or comma-separated list.

        NEVER REPEAT THE SAME SEARCH MORE THAN ONCE.

        Start with Wikipedia, then web search if needed. Use all tools to find the correct answer.

        If a file is provided (named {file_name} with extension {file_extension}), first action MUST BE TO CALL the download_file tool with URL:
        {DEFAULT_API_URL}/files/{file_name}
        Do NOT include the extension in the URL and send WITHOUT MODIFICATION.
        """

        sys_msg = SystemMessage(content=prompt)
        time.sleep(40)  # Simulate processing delay
        return {"messages": [self.llm_with_tools.invoke([sys_msg] + state["messages"])]}

    @staticmethod
    def final_answer(state: State):
        system_prompt = f"""
        You will be given an answer and a question. Remove everything unnecessary and answer exactly.
        Do not include 'FINAL ANSWER' in output. Follow the question format strictly.
        """
        human_prompt = f"""
        Question: {state['question']}
        Answer: {state['messages'][-1]}
        """
        # response = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0).invoke([
        #     SystemMessage(content=system_prompt),
        #     HumanMessage(content=human_prompt)
        # ])
        response = ChatOpenAI(model="gpt-4.1-mini", temperature=0).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
        ])
        return {"messages": state["messages"] + [response]}

# --- Function to Run & Submit All Questions ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # Fetch questions
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            return "Fetched questions list is empty or invalid format.", None
    except Exception as e:
        return f"Error fetching questions: {e}", None

    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")
        if not task_id or question_text is None:
            continue
        try:
            submitted_answer = agent(question_text, task_id, file_name)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        return final_status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results_log)

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        Instructions:
        1. Clone this space, modify agent code, tools, and packages.
        2. Log in to Hugging Face using the button below.
        3. Click 'Run Evaluation & Submit All Answers' to fetch, answer, and submit.
        """
    )
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"Repo URL: https://huggingface.co/spaces/{space_id_startup}")
    print("-"*(60 + len(" App Starting ")) + "\n")
    demo.launch(debug=True, share=False, auth = None)
