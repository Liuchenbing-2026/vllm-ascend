# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Exercise the example launcher's real TCP preflight without loading a model."""

import ast
import errno
import socket
import unittest
from pathlib import Path


class TestBidirectionalLauncherPorts(unittest.TestCase):
    @staticmethod
    def probe(port):
        script = (
            Path(__file__).resolve().parents[2] / "examples/disaggregated_prefill_v1/serve_bidirectional_multiturn.sh"
        )
        preflight = script.read_text().split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        tree = ast.parse(preflight)
        loop = next(node for node in tree.body if isinstance(node, ast.For) and "socket.socket" in ast.unparse(node))
        namespace = {"socket": socket, "errno": errno, "port": port}
        exec(compile(ast.Module(body=loop.body, type_ignores=[]), f"{script} (preflight)", "exec"), namespace)

    def test_available_port(self):
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            port = server.getsockname()[1]
        self.probe(port)

    def test_active_listener_rejected(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            with self.assertRaises(OSError) as error:
                self.probe(server.getsockname()[1])
            self.assertEqual(error.exception.errno, errno.EADDRINUSE)

    def test_time_wait_allows_restart(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            port = server.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                accepted, _ = server.accept()
                # Close the server side first, putting its port into TIME_WAIT.
                accepted.close()
                self.assertEqual(client.recv(1), b"")
        self.probe(port)


if __name__ == "__main__":
    unittest.main()
