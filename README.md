This is a platform to build agentic frameworks via recursive generation.

**Objective**:
An LLM agent driven programmatic platform engine that automates the recursive decomposition of high-level software requirements into a highly granular, agentic skill tree, and subsequently compiles that tree into an executable agent framework.
Given a product requirement, the engine first breaks down the problem into skills needed. Then for each skill, break it down into sub-skills; sub-skills are further broken down into sub-sub-skills. This recursive process continues until there is no more reasonable granularity.

**Scope of the MVP**:
The MVP will focus strictly on the generation and structural compilation phases using a semi-automated, human-in-the-loop approach. The platform itself will be an orchestrated multi-agent system built in Google ADK, generating a static Python file for a target project. The MVP will not dynamically execute the generated target code.

**Repo structure**
ai-regressive-generator/
├── platform_engine/          # The builder (ADK Workflow)
│   ├── main.py               # Entry point
│   ├── config.py             # LLM configurations & prompts
│   └── internal_agents/
│       ├── architect.py      # Generates macro-milestones
│       ├── decomposer.py     # Recursive skill splitter
│       └── compiler.py       # Validates JSON and outputs ADK Python code
│
└── output/                   # Target project workspaces
    └── {project_name}/
        ├── blueprint_raw.json      # Output from Decomposer
        ├── blueprint_verified.json # Approved by human
        ├── run.py                  # Compiled ADK script (Target Software)
        └── primitives/             # Stub functions/tools generated


Use docker and uv to manage the environment

