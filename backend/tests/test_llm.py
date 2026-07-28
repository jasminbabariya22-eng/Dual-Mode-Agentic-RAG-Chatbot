import asyncio
from backend.app.core.llm import get_llm

async def main():
    llm = get_llm()
    response = await llm.ainvoke("Hello")
    print(response)

asyncio.run(main())