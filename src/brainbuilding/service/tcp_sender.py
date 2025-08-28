import asyncio
import json
import logging
import multiprocessing as mp
from queue import Empty


async def tcp_sender(queue: mp.Queue, host: str, port: int):
    """Async process that handles TCP connection and sending results"""
    writer = None
    retries = 0
    max_retries = 500

    while retries < max_retries:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            logging.info("TCP connection established")
            break
        except ConnectionRefusedError:
            retries += 1
            logging.warning(
                f"Connection attempt {retries} failed, retrying in 1 second..."
            )
            await asyncio.sleep(1)

    if writer is None:
        logging.error("Failed to establish TCP connection")
        return

    try:
        while True:
            try:
                result = queue.get_nowait()
                message = result
                data = json.dumps(message).encode() + b"\n"
                writer.write(data)
                await writer.drain()
                logging.info(f"Sent classification result: {result}")
            except Empty:
                await asyncio.sleep(0.0001)
            except Exception as e:
                logging.error(f"Error sending result: {e}")
                await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        logging.info("TCP sender received shutdown signal")
    except Exception as e:
        logging.error(f"TCP sender error: {e}")
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        logging.info("TCP sender shutting down")


def tcp_sender_process(queue: mp.Queue, host: str, port: int):
    """Process that runs the TCP sender"""
    asyncio.run(tcp_sender(queue, host, port))
