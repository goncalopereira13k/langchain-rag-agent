import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.agents.rag_agent import build_agent, is_plain_text

if __name__ == "__main__":
    agent = build_agent()
    query = input("Question: ").strip()
    if not query:
        print("No question provided.")
        sys.exit(1)

    print("\nStreaming agent response...\n")
    final_message = None
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        final_message = event["messages"][-1]
        final_message.pretty_print()

    if final_message and hasattr(final_message, "content") and isinstance(final_message.content, str):
        if not is_plain_text(final_message.content):
            print("\n[WARNING] Response may contain injected formatting — review before use.")
