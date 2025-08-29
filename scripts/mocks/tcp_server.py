import socket
import json

class TCPServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server listening on {self.host}:{self.port}")

            while True:
                client_socket, address = self.server_socket.accept()
                print(f"Connection from {address}")
                self.handle_client(client_socket)

        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.server_socket.close()

    def handle_client(self, client_socket):
        try:
            data = b""
            while True:
                chunk = client_socket.recv(1024)
                if not chunk:
                    break
                data += chunk

            if data:
                try:
                    json_data = json.loads(data.decode('utf-8'))
                    print("Received JSON data:")
                    print(json.dumps(json_data, indent=2))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print("Received raw data:", data.decode('utf-8'))

        except Exception as e:
            print(f"Error handling client: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    server = TCPServer()
    server.start()
