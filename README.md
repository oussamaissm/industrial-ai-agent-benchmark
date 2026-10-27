# Efficient AI Agents for Industrial Automation

Research-grade benchmark for evaluating Small Language Models (SLMs), tool use, and deterministic verification in industrial engineering tasks.

## Setup
\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
python -m data.generate_factory_data
\`\`\`

## Running the MCP Server
\`\`\`bash
python -m src.tools.factory_mcp_server
\`\`\`

## Running the Benchmark
\`\`\`bash
python -m src.evaluation.runner
python -m src.evaluation.metrics
\`\`\`
