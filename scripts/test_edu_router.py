import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import router_call

async def test_edu_router():
    queries = [
        "je veux apprendre l'alphabet",
        "apprends moi à lire s'il te plaît",
        "apprends moi à compter jusqu'à 10"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        resp = await router_call(q)
        print(f"Response: {resp}")

if __name__ == "__main__":
    asyncio.run(test_edu_router())
