from mcp.server.fastmcp import FastMCP
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.simulator.factory import FactorySimulator

mcp = FastMCP("FactoryOperations")
simulator = FactorySimulator()

@mcp.tool()
def get_machine_status(machine_id: str) -> str:
    """Get the current operational status and downtime of a specific machine."""
    return str(simulator.get_machine_status(machine_id))

@mcp.tool()
def get_order_details(order_id: str) -> str:
    """Retrieve quantity, priority, and deadline for a production order."""
    return str(simulator.get_order_details(order_id))

@mcp.tool()
def calculate_schedule(order_id: str, machine_id: str) -> str:
    """Validate and calculate a production schedule for an order on a specific machine."""
    return str(simulator.calculate_schedule(order_id, machine_id))

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
