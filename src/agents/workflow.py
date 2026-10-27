from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
import time
import psutil
import json
import os

# Import simulator for tool execution
from src.simulator.factory import FactorySimulator
from getpass import getpass

os.environ["GROQ_API_KEY"] = getpass("Enter your Groq API key: ")

class AgentState(TypedDict):
    request: str
    messages: list
    mode: Literal["baseline", "tool", "verified", "hybrid"]
    retry_count: int
    verification_result: dict
    metrics: dict

# Global model instance to avoid reloading weights on every task
from langchain_groq import ChatGroq

_llm_instance = None

def get_llm(model_size: str = "small"):
    global _llm_instance

    if _llm_instance is None:
        _llm_instance = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=512,
            max_retries=2,
            timeout=30,
        )

    return _llm_instance

def call_llm(state: AgentState, model_size: str = "small") -> dict:
    llm = get_llm(model_size)

    start_time = time.time()
    process = psutil.Process()
    start_mem = process.memory_info().rss

    system_prompt = """You are an Industrial Operations Assistant.

Return ONLY a valid JSON object matching this schema:
{
  "answer": "string",
  "structured_data": null or {
    "order_id": "string",
    "machine_id": "string",
    "estimated_hours": 0.0,
    "reasoning": "string"
  },
  "confidence": 0.9
}

Do not include markdown or text outside the JSON object.
Do not invent order IDs or machine IDs.
"""

    messages = [
        ("system", system_prompt)
    ]

    recent_messages = state["messages"][-6:]

    for msg in recent_messages:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("assistant", msg.content))
        elif isinstance(msg, ToolMessage):
            messages.append(
                ("human", f"Tool result: {msg.content}")
            )

    messages.append(("human", state["request"]))

    response = llm.invoke(messages)

    end_time = time.time()
    end_mem = process.memory_info().rss

    content = response.content

    usage = response.usage_metadata or {}

    state["metrics"]["latency"] = end_time - start_time
    state["metrics"]["memory_delta_mb"] = (
        end_mem - start_mem
    ) / (1024 * 1024)

    state["metrics"]["tokens_estimated"] = usage.get(
        "total_tokens", 0
    )

    state["messages"].append(
        AIMessage(content=content)
    )

    return state

def call_tools(state: AgentState) -> dict:
    """Simplified tool caller that parses the LLM's JSON and executes the simulator."""
    simulator = FactorySimulator()
    last_msg = state["messages"][-1]
    
    try:
        # Try to parse the LLM's JSON response
        data = json.loads(last_msg.content)
        structured = data.get("structured_data")
        
        if structured and "machine_id" in structured and "order_id" in structured:
            # Execute the real factory tool
            result = simulator.calculate_schedule(
                order_id=structured["order_id"],
                machine_id=structured["machine_id"]
            )
            state["messages"].append(ToolMessage(content=str(result), tool_call_id="call_1"))
        else:
            state["messages"].append(ToolMessage(content="No valid tool arguments found in JSON.", tool_call_id="call_1"))
    except json.JSONDecodeError:
        state["messages"].append(ToolMessage(content="Error: LLM did not output valid JSON.", tool_call_id="call_1"))
        
    return state
def verify_output(state: AgentState) -> dict:
    last_msg = next(
        (
            msg for msg in reversed(state["messages"])
            if isinstance(msg, AIMessage)
        ),
        None
    )

    try:
        if last_msg is None:
            raise ValueError("No AI response found")

        data = json.loads(last_msg.content)

        is_valid = (
            isinstance(data, dict)
            and isinstance(data.get("answer"), str)
            and isinstance(data.get("confidence"), (int, float))
            and 0 <= data["confidence"] <= 1
        )

        state["verification_result"] = {
            "valid": is_valid,
            "reason": (
                "Response structure is valid"
                if is_valid
                else "Invalid response structure"
            )
        }

    except (json.JSONDecodeError, ValueError, TypeError):
        state["verification_result"] = {
            "valid": False,
            "reason": "Invalid JSON or missing response"
        }

    return state

def route_after_verification(state: AgentState) -> str:
    if state["verification_result"]["valid"]:
        return END
    if state["mode"] == "verified" and state["retry_count"] < 2:
        state["retry_count"] += 1
        return "llm_node"
    if state["mode"] == "hybrid" and state["retry_count"] < 1:
        state["retry_count"] += 1
        return "strong_llm_node"
    return END

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("llm_node", lambda s: call_llm(s, "small"))
    workflow.add_node("strong_llm_node", lambda s: call_llm(s, "large")) # Placeholder for hybrid
    workflow.add_node("tool_node", call_tools)
    workflow.add_node("verify_node", verify_output)
    
    workflow.add_edge("llm_node", "tool_node")
    workflow.add_edge("tool_node", "verify_node")
    workflow.add_conditional_edges("verify_node", route_after_verification, {
        "llm_node": "llm_node", 
        "strong_llm_node": "strong_llm_node", 
        END: END
    })
    workflow.add_edge("strong_llm_node", END)
    workflow.set_entry_point("llm_node")
    return workflow.compile()
