import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import router_call

async def test_router():
    print("Testing Web Search Router...")
    resp = await router_call("quelle heure est-il à Paris ?")
    print(f"Web Search Response: {resp}")

    print("\nTesting Story Router...")
    resp = await router_call("raconte-moi une histoire de princesses et de dragons")
    print(f"Story Response: {resp}")

    print("\nTesting Normal Interaction Router...")
    resp = await router_call("Comment vas-tu aujourd'hui ?")
    print(f"Normal Response: {resp}")

if __name__ == "__main__":
    asyncio.run(test_router())
