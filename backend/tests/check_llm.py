import asyncio

from backend.app.core.llm import get_llm

async def main():
    llm = get_llm()

    print(type(llm))

    response = await llm.ainvoke("Say hello in one sentence.")

    print(response)

asyncio.run(main())