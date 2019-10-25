import asyncio
import aiohttp


async def create_tasks(loop, paras, func):
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [asyncio.ensure_future(func(session, **para)) for para in paras]
        await asyncio.gather(*tasks)
    return tasks


def create_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop = asyncio.get_event_loop()
    return loop


