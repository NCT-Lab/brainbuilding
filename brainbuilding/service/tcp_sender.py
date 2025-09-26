import asyncio
import json
import logging
import multiprocessing as mp
from queue import Empty


async def tcp_sender_async(
    queue: mp.Queue, host: str, port: int, max_retries: int
):
    """Async process that handles TCP connection and sending results"""
    writer = None
    retries = 0

    while retries < max_retries:
        try:
            print(f"Connecting to {host}:{port}")
            reader, writer = await asyncio.open_connection(host, port)
            logging.info("TCP connection established")
            handshake = {"type": "handshake", "client_name": "eeg-processor"}
            writer.write(json.dumps(handshake).encode() + b"\n")
            await writer.drain()
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
                if isinstance(result, dict):
                    payload = dict(result)
                    if payload.get("type") is None:
                        payload["type"] = "data"
                else:
                    payload = {"type": "data", "payload": result}
                data = json.dumps(payload).encode() + b"\n"
                writer.write(data)
                await writer.drain()
                logging.info(f"Sent message: {payload}")
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


def tcp_sender_process(
    queue: mp.Queue, host: str, port: int, max_retries: int
):
    """Process that runs the TCP sender"""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(tcp_sender_async(queue, host, port, max_retries))


class TCPSender:
    def __init__(self, queue: mp.Queue, host: str, port: int, retries: int):
        self.queue = queue
        self.host = host
        self.port = port
        self.retries = retries
        self.process = None

    def start(self):
        print(f"Starting TCP sender process for {self.host}:{self.port}")
        self.process = mp.Process(
            target=tcp_sender_process,
            args=(self.queue, self.host, self.port, self.retries),
        )
        self.process.start()
        logging.info("TCP sender process started")

    def stop(self):
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join()
            logging.info("TCP sender process stopped")
