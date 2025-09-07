from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

# This prompt template is a placeholder. You can customize it.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        ("placeholder", "{chat_history}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# --- Tool Definitions ---

@tool
def predict_price(address: str) -> float:
    """Predict the house price for a given address."""
    # Dummy implementation
    print(f"Predicting price for {address}...")
    return 600_000.0  # Replace with real prediction logic

@tool
def predict_network(city: str, date: str) -> str:
    """Predict the weather for a given city and date."""
    # Dummy implementation
    print(f"Predicting weather for {city} on {date}...")
    return "Sunny"  # Replace with real prediction logic

@tool
def final_answer(answer: str, tools_used: list[str]) -> dict:
    """Use this tool to provide a final answer to the user when you have used other tools."""
    return {"answer": answer, "tools_used": tools_used}

# Map of tools
tools = [predict_price, predict_network, final_answer]
name2tool = {tool.name: tool.func for tool in tools}

def execute_tool(tool_call: AIMessage) -> ToolMessage:
    """Execute a tool call synchronously"""
    tool_name = tool_call.tool_calls[0]["name"]
    tool_args = tool_call.tool_calls[0]["args"]
    tool_out = name2tool[tool_name](**tool_args)
    return ToolMessage(
        content=f"{tool_out}",
        tool_call_id=tool_call.tool_calls[0]["id"]
    )

# --- Custom Agent Executor ---

class AgentExecutor:
    def __init__(self, llm, max_iterations: int = 3):
        self.chat_history: list[BaseMessage] = []
        self.max_iterations = max_iterations
        # The agent is now created with the llm passed during initialization
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: x.get("agent_scratchpad", [])
            }
            | prompt
            | llm.bind_tools(tools)
        )

    def invoke(self, input: str, verbose: bool = False) -> dict:
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
        used_tools = []

        while count < self.max_iterations:
            if verbose:
                print(f"--- Iteration {count+1} ---")
            
            response = self.agent.invoke({
                "input": input,
                "chat_history": self.chat_history,
                "agent_scratchpad": agent_scratchpad
            })
            
            if verbose:
                print(f"LLM Response: {response.content}")
                print(f"Tool Calls: {response.tool_calls}")

            if response.tool_calls:
                tool_call_msg = AIMessage(content=response.content, tool_calls=response.tool_calls)
                tool_result = execute_tool(tool_call_msg)
                
                agent_scratchpad.extend([tool_call_msg, tool_result])
                
                tool_name = response.tool_calls[0]["name"]
                if tool_name != "final_answer":
                    used_tools.append(tool_name)
                
                if tool_name == "final_answer":
                    try:
                        final_answer_data = eval(tool_result.content)
                        final_answer = final_answer_data.get("answer", "Could not parse final answer.")
                    except:
                        final_answer = tool_result.content
                    break
            else:
                final_answer = response.content
                break
            
            count += 1

        self.chat_history.extend([
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else "No answer found")
        ])
        
        return {
            "answer": final_answer if final_answer else "No answer found",
            "tools_used": list(set(used_tools))
        }