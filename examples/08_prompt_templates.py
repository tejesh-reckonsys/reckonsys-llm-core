"""
Prompt templates with Jinja2.

    python examples/08_prompt_templates.py

Requires: pip install "reckonsys-llm-core[templates,claude]"
          ANTHROPIC_API_KEY

Expects template files at examples/prompts/:
    system.md.j2   — e.g. "You are a {{ role }}."
    user.md.j2     — e.g. "Answer this question: {{ query }}"
"""

from reckonsys_llm_core import ChatMessage, create_llm
from reckonsys_llm_core.templates import configure_templates, render_prompt

# Call once at startup (e.g. in settings.py or main.py)
configure_templates(
    "examples/prompts",
    trim_blocks=True,
    lstrip_blocks=True,
)

client = create_llm("claude", "claude-opus-4-6")

system = render_prompt("system.md.j2", {"role": "concise assistant"})
user = render_prompt("user.md.j2", {"query": "What is the capital of France?"})

response = client.query(
    messages=[ChatMessage(role="user", content=user)],
    system=system,
)
print(response.content)
