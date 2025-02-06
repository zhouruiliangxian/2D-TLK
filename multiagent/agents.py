from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_core.messages import convert_to_messages

from langgraph.checkpoint.memory import MemorySaver
import subprocess
import os

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

import re
import subprocess
import os
import uuid
import json

from typing import Annotated
from langgraph.types import Command
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import InjectedState


from typing_extensions import Literal
from langgraph.graph import MessagesState, StateGraph, START
import base64
import copy


global input_path
global output_path

memory = MemorySaver()
visual_model = ChatOpenAI(
    temperature=0,
    model="moonshot-v1-8k-vision-preview",
    openai_api_base="https://api.moonshot.cn/v1",
    openai_api_key="",
)

llm_model = ChatOpenAI(
    temperature=0,
    model="glm-4-air",
    openai_api_key="",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/",
)


@tool
def code_executor(code: str):
    """Execute code in the Python environment, return result."""

    file_name = f"{uuid.uuid4()}.py"
    base_name = r"D:\llm\langchain\layer_identification\code"
    file_path = os.path.join(base_name, file_name)

    os.makedirs("code", exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as code_file:
        code_file.write(code)

    try:
        env_activation = "activate mmseg"
        script_command = f"python {file_path}"

        full_command = f"{env_activation} && {script_command}"
        process = subprocess.run(
            full_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        output = process.stdout
        error = process.stderr

        # 如果有错误，抛出异常
        if process.returncode != 0:
            return {"error": error}

        return {"output": output}

    except Exception as e:
        return {"error": str(e)}


@tool
def identify_layer():
    """identify the layer of the image"""

    import requests

    def send_image_to_server(image_path, output_path):
        url = "http://127.0.0.1:5000/process_image"

        data = {"img_bgr_file": image_path, "save_path": output_path}

        response = requests.post(url, data=data)

        if response.status_code == 200:
            return f"Server Response: {response.json()}"
        else:
            return f"Error: {response.json()}"

    global output_path

    output_path = os.path.normpath(
        os.path.join(os.path.dirname(input_path), "pre-" + os.path.basename(input_path))
    )
    res = send_image_to_server(input_path, output_path)
    json_str = re.search(r"Server Response: (.+)", res).group(1)
    json_str = json_str.replace("'", '"')
    response_data = json.loads(json_str)

    res = response_data.get("result", "")

    tool_msg = "this is the result of the identify layer :\n " + res

    tool_msg = (
        tool_msg
        + "\n\n"
        + "Class means the type of layer, Count means the number of the layer."
    )
    return tool_msg


def make_handoff_tool(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def handoff_to_agent(
        state: Annotated[dict, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred. You are now at {agent_name}. Please continue.",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )

    return handoff_to_agent


def make_agent(model, tools, model_name, system_prompt=None):
    model_with_tools = model.bind_tools(tools)
    tools_by_name = {tool.name: tool for tool in tools}

    def call_model(state: MessagesState) -> Command[Literal["call_tools", "__end__"]]:
        msgs = state["messages"]
        vis_tool = model_name == "visual_model" and isinstance(msgs[-1], ToolMessage)

        if model_name == "llm_model" or vis_tool:
            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt}] + msgs

            response = model_with_tools.invoke(msgs)
        else:
            content = []
            content.append({"type": "text", "text": msgs[-1].content})

            global input_path
            global output_path

            if input_path:
                with open(input_path, "rb") as f:
                    input_data = f.read()
                    input_data = base64.b64encode(input_data).decode("utf-8")
                    input_data = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{input_data}"},
                    }
                    content.append(input_data)

            if output_path:
                with open(output_path, "rb") as f:
                    output_data = f.read()
                    output_data = base64.b64encode(output_data).decode("utf-8")
                    output_data = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{output_data}"},
                    }
                    content.append(output_data)

            human_msg = HumanMessage(
                content=content,
            )
            msgs_copy = copy.deepcopy(msgs)
            msgs_copy[-1] = human_msg
            if system_prompt:
                msgs = [SystemMessage(content=system_prompt)] + msgs
            # print(msgs)
            response = model_with_tools.invoke(msgs)

        if len(response.tool_calls) > 0:
            return Command(goto="call_tools", update={"messages": [response]})

        return {"messages": [response]}

    def call_tools(state: MessagesState) -> Command[Literal["call_model"]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_ = tools_by_name[tool_call["name"]]
            tool_input_fields = tool_.get_input_schema().model_json_schema()[
                "properties"
            ]

            if "state" in tool_input_fields:
                # inject state
                tool_call = {**tool_call, "args": {**tool_call["args"], "state": state}}

            tool_response = tool_.invoke(tool_call)
            if isinstance(tool_response, ToolMessage):
                results.append(Command(update={"messages": [tool_response]}))

            elif isinstance(tool_response, Command):
                results.append(tool_response)
        return results

    graph = StateGraph(MessagesState)
    graph.add_node(call_model)
    graph.add_node(call_tools)
    graph.add_edge(START, "call_model")
    graph.add_edge("call_tools", "call_model")

    return graph.compile()


visual_agent = make_agent(
    visual_model,
    [identify_layer, make_handoff_tool(agent_name="coder_agent")],
    model_name="visual_model",
    system_prompt="You are a visual agent. You can call the tool to identify the layer of the provided image. You also can ask a coder for help with coding.",
)


def call_visual_agent(
    state: MessagesState,
) -> Command[Literal["coder_agent", "human"]]:

    response = visual_agent.invoke(state)
    return Command(update=response, goto="human")


coder_agent = make_agent(
    llm_model,
    [code_executor, make_handoff_tool(agent_name="visual_agent")],
    model_name="llm_model",
    system_prompt="You are a coder agent.\
        You can call the tool to execute the code.\
            You must always return valid JSON fenced by a markdown code block when you using the code_executor tool. Do not return any additional text. \
                Ensure that the provided code prints the execution result. \
                You can ask a visual agent for help when you need to identify the layer of the provided image.",
)


def call_coder_agent(
    state: MessagesState,
) -> Command[Literal["visual_agent", "human"]]:

    response = coder_agent.invoke(state)
    return Command(update=response, goto="human")


def human_node(
    state: MessagesState, config
) -> Command[Literal["visual_agent", "coder_agent", "human"]]:
    """A node for collecting user input."""

    user_input = input()

    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")

    active_agent = langgraph_triggers[0].split(":")[1]

    return Command(
        update={
            "messages": [
                {
                    "role": "human",
                    "content": user_input,
                }
            ]
        },
        goto=active_agent,
    )


builder = StateGraph(MessagesState)
builder.add_node("visual_agent", call_visual_agent)
builder.add_node("coder_agent", call_coder_agent)
builder.add_node("human", human_node)


builder.add_edge(START, "visual_agent")
graph = builder.compile(checkpointer=memory)

"""
## visualize the graph
from IPython.display import Image, display  
  
# display(Image(graph .get_graph(xray=True).draw_mermaid_png()))  
 
display(Image(graph.get_graph().draw_mermaid_png()))

"""


def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")


input_path = r"D:\llm\langchain\layer_identification\8_6.jpg"  ## input image path
output_path = ""


"""
RUN:

config = {"configurable": {"thread_id": "1"}}
for chunk in graph.stream(
    {"messages": [("user", "Identify the layer of the image")]
     }, subgraphs=True,config=config
):
    pretty_print_messages(chunk)


config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"messages": [("user", "Visualize each category's proportion with relative colors and save the image using the code.")]
     }, subgraphs=True,config=config
):
    pretty_print_messages(chunk)



"""
