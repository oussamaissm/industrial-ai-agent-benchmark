# tool_agent.py
# This file can be used to instantiate the unified graph from workflow.py 
# with specific configuration overrides for the tool agent mode.
from src.agents.workflow import build_agent_graph

def get_tool_agent_graph():
    # In a full implementation, this would configure the graph 
    # specifically for the tool_agent ablation (e.g., disabling verify_node for baseline)
    return build_agent_graph()
