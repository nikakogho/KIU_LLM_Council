import asyncio
import sys
from llm_council.runner import ask_all

async def main():
    user_prompt = " ".join(sys.argv[1:]).strip()
    if not user_prompt:
        user_prompt = input("Prompt: ").strip()

    replies = await ask_all(user_prompt, system_prompt="You are a helpful assistant. Answer clearly.")
    for r in replies:
        print("\n" + "=" * 80)
        print(f"{r.provider} | {r.model} | {r.latency_ms} ms | error={bool(r.error)}")
        if r.error:
            print("ERROR:", r.error)
        else:
            print(r.text)

if __name__ == "__main__":
    asyncio.run(main())
