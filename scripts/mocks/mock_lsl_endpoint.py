import asyncio
import json
import pyxdf
from pylsl import StreamInfo, StreamOutlet
import json

with open("data/raw-dataset-2/1/Task.json") as f:
    EVENTS = json.load(f)

event_map = {
    'EyeWarmupDesc': 1,
    'EyeWarmupBlink': 2,
    'EyeWarmupText': 3,
    'EyeWarmupMove': 4,
    'Background': 5,
    'Attention': 10,
    'Image': 10,
    'Point': 10,
    'Rest': 10,
    'Vas': 10
}

class MockLSLSender:
    def __init__(self, xdf_file: str):
        self.xdf_file = xdf_file
        self.streams = []
        self.outlets = {}
        self._load_xdf()

    def _load_xdf(self):
        """Load XDF file using pyxdf."""
        print(f"Loading XDF file: {self.xdf_file}")
        self.streams, header = pyxdf.load_xdf(self.xdf_file)
        print(f"Found {len(self.streams)} streams")
        
        # Create outlets for each stream
        for stream in self.streams:
            name = stream['info']['name'][0]
            print(f"\nCreating outlet for stream: {name}")
            
            info = StreamInfo(
                name=name,
                type=stream['info']['type'][0],
                channel_count=int(stream['info']['channel_count'][0]),
                nominal_srate=float(stream['info']['nominal_srate'][0]),
                channel_format='float32',
                source_id=stream['info']['source_id'][0]
            )
            
            outlet = StreamOutlet(info)
            self.outlets[name] = {
                'outlet': outlet,
                'data': stream['time_series'],
                'timestamps': stream['time_stamps']
            }
            
            print(f"\nStream details:")
            print(f"- Type: {stream['info']['type'][0]}")
            print(f"- Channel count: {stream['info']['channel_count'][0]}")
            print(f"- Sampling rate: {stream['info']['nominal_srate'][0]} Hz")

    def _create_synchronized_iterator(self):
        """Create an iterator that yields data points in temporal order across all streams."""
        # Create a list of (timestamp, stream_name, sample_index, sample) tuples for all streams
        all_samples = []
        for name, stream_info in self.outlets.items():
            samples = list(zip(
                stream_info['timestamps'],
                [name] * len(stream_info['timestamps']),
                range(len(stream_info['timestamps'])),
                stream_info['data']
            ))
            all_samples.extend(samples)
        
        # Sort by timestamp
        all_samples.sort(key=lambda x: x[0])
        return all_samples
    
    def get_sample_type(self, sample_id: int) -> str:
        return EVENTS['samples'][EVENTS['events'][sample_id]['sample_id']]['sample_type']

    async def run(self):
        """Run the mock LSL sender."""
        print("Starting synchronized streaming...")
        await asyncio.sleep(0.5)
        
        synchronized_samples = self._create_synchronized_iterator()
        
        for timestamp, stream_name, sample_idx, sample in synchronized_samples:
            sample_id = sample[0]
            outlet = self.outlets[stream_name]['outlet']

            if stream_name == "Brainbuilding-Events":
                print(f"Event {self.get_sample_type(sample_id)} from {stream_name} at {timestamp:.3f}")
                print(sample)
                sample = [event_map[self.get_sample_type(sample_id)], 0]
            else:
                pass
            outlet.push_sample(sample, timestamp)
            await asyncio.sleep(1/500)
            
        print("Finished streaming all data")
            

class ResultsReceiver:
    def __init__(self, host: str = 'localhost', port: int = 5000):
        self.host = host
        self.port = port
        self.running = False

    async def connect_and_receive(self):
        """Connect to the LSL processor and receive results."""
        try:
            reader, writer = await asyncio.open_connection(self.host, self.port)
            print(f"Connected to LSL processor at {self.host}:{self.port}")
            
            self.running = True
            while self.running:
                try:
                    data = await reader.readline()
                    if not data:
                        break
                    
                    # Parse and display the results
                    results = json.loads(data.decode())
                    self._display_results(results)
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except Exception as e:
                    print(f"Error receiving data: {e}")
                    break
                    
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            self.running = False
            if 'writer' in locals():
                writer.close()
                await writer.wait_closed()

    def _display_results(self, results: dict):
        """Display the received results in a readable format."""
        print("\n=== Received Results ===")
        print(f"Stream: {results['stream_name']}")
        print(f"Timestamp: {results['timestamp']:.3f}")
        print(f"Data: {results['data'][:5]}...")  # Show first 5 values

async def main():
    sender = MockLSLSender(
        xdf_file="data/raw-dataset-2/1/data.xdf"
    )
    
    await asyncio.gather(
        sender.run(),
    )

if __name__ == "__main__":
    asyncio.run(main())
