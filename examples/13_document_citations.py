"""
Document citations — passing text chunks as document blocks and getting cited passages back.

Claude can cite specific passages from DocumentContent blocks in its response.
Citations appear in LLMResponse.citations with document_title and document_index set.

    python examples/13_document_citations.py

Requires: ANTHROPIC_API_KEY
"""

from reckonsys_llm_core import ChatMessage, DocumentContent, LLMClient, TextContent
from reckonsys_llm_core.strategies.claude import ClaudeLLMStrategy, create_claude_client

strategy = ClaudeLLMStrategy(
    client=create_claude_client(),
    model="claude-opus-4-6",
)
client = LLMClient(strategy)

# ---------------------------------------------------------------------------
# 1. Single document — ask a question, get citations back
# ---------------------------------------------------------------------------

DOC = """
The Python programming language was created by Guido van Rossum and first released in 1991.
It emphasises code readability and uses significant indentation. Python supports multiple
programming paradigms, including structured, object-oriented, and functional programming.
Python is dynamically typed and garbage-collected. It was named after the BBC comedy series
Monty Python's Flying Circus, not after the snake.
"""

response = client.query(
    messages=[
        ChatMessage(
            role="user",
            content=[
                DocumentContent(text=DOC.strip(), title="Python Overview"),
                TextContent(text="Who created Python, and what was it named after? Cite your sources."),
            ],
        )
    ],
)

print("=== Single document ===")
print("Answer:", response.content)
print()
if response.citations:
    print(f"Citations ({len(response.citations)}):")
    for i, c in enumerate(response.citations, 1):
        print(f"  [{i}] \"{c.cited_text}\"")
        if c.document_title:
            print(f"      from: {c.document_title} (doc index {c.document_index})")
else:
    print("(no citations returned)")

print()

# ---------------------------------------------------------------------------
# 2. Multiple documents — Claude chooses which to cite
# ---------------------------------------------------------------------------

DOC_A = "The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889."
DOC_B = "The Colosseum is an ancient amphitheatre in the centre of Rome, Italy. It was completed in 80 AD."

response2 = client.query(
    messages=[
        ChatMessage(
            role="user",
            content=[
                DocumentContent(text=DOC_A, title="Eiffel Tower Facts"),
                DocumentContent(text=DOC_B, title="Colosseum Facts"),
                TextContent(text="When was each landmark built? Cite your sources."),
            ],
        )
    ],
)

print("=== Multiple documents ===")
print("Answer:", response2.content)
print()
if response2.citations:
    print(f"Citations ({len(response2.citations)}):")
    for i, c in enumerate(response2.citations, 1):
        print(f"  [{i}] \"{c.cited_text}\"")
        if c.document_title:
            print(f"      from: {c.document_title} (doc index {c.document_index})")
else:
    print("(no citations returned)")
