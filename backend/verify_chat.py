import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        print("Testing POST /api/v1/chat")
        resp = await client.post(
            "/api/v1/chat",
            json={"question": "What is the laptop warranty period?", "session_id": "test-123"}
        )
        print("Status:", resp.status_code)
        if resp.status_code == 200:
            print(resp.json())
        else:
            print(resp.text)
        
        print("\nTesting POST /api/v1/chat/stream")
        async with client.stream(
            "POST", 
            "/api/v1/chat/stream",
            json={"question": "What is the laptop warranty period?", "session_id": "test-124"}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    print(line)

if __name__ == "__main__":
    asyncio.run(main())
