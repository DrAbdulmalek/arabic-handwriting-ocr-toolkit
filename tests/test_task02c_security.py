"""TASK-02C — اختبارات تكامل فعلية لحراسة الوصول ونقل الرمز."""
import http.client
import importlib
import json
import os
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

srv = importlib.import_module("7_box_editor_server")


class TestTokenFile(unittest.TestCase):
    def test_token_file_is_private_and_contains_only_the_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box-editor.token"
            result = srv.store_auth_token("unit-test-secret", path)
            self.assertEqual(result.read_text(encoding="utf-8"), "unit-test-secret")
            if os.name == "posix":
                self.assertEqual(result.stat().st_mode & 0o777, 0o600)


class TestHTTPAccessGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_token = srv.AUTH_TOKEN
        cls.old_origins = srv.ALLOWED_ORIGINS
        cls.old_port = srv.SERVER_PORT
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), srv.Handler)
        cls.port = cls.server.server_address[1]
        srv.AUTH_TOKEN = "integration-test-token"
        srv.SERVER_PORT = cls.port
        srv.ALLOWED_ORIGINS = {f"http://127.0.0.1:{cls.port}"}
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)
        srv.AUTH_TOKEN = cls.old_token
        srv.ALLOWED_ORIGINS = cls.old_origins
        srv.SERVER_PORT = cls.old_port

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=4)
        request_headers = {"Connection": "close"}
        request_headers.update(headers or {})
        conn.request(method, path, body=body, headers=request_headers)
        response = conn.getresponse()
        payload = response.read()
        result = (response.status, response.headers, payload)
        conn.close()
        return result

    def test_static_editor_shell_is_available_before_authentication(self):
        status, _, _ = self.request("GET", "/")
        self.assertEqual(status, 200)

    def test_api_rejects_missing_token_even_without_origin(self):
        status, _, _ = self.request("GET", "/api/health")
        self.assertEqual(status, 401)

    def test_query_parameter_token_is_rejected(self):
        status, _, _ = self.request("GET", "/api/health?token=integration-test-token")
        self.assertEqual(status, 401)

    def test_cookie_token_is_rejected(self):
        status, _, _ = self.request(
            "GET", "/api/health", headers={"Cookie": "tkn=integration-test-token"}
        )
        self.assertEqual(status, 401)

    def test_header_token_is_accepted(self):
        status, headers, _ = self.request(
            "GET", "/api/health", headers={"X-Auth-Token": "integration-test-token"}
        )
        self.assertEqual(status, 200)
        self.assertIsNone(headers.get("Set-Cookie"))

    def test_wrong_origin_is_rejected_even_with_valid_token(self):
        status, _, _ = self.request(
            "GET", "/api/health",
            headers={
                "Origin": "https://evil.example",
                "X-Auth-Token": "integration-test-token",
            },
        )
        self.assertEqual(status, 403)

    def test_dns_rebinding_hostname_is_rejected(self):
        host = f"attacker.example:{self.port}"
        status, _, _ = self.request(
            "GET", "/api/health",
            headers={
                "Host": host,
                "Origin": f"http://{host}",
                "X-Auth-Token": "integration-test-token",
            },
        )
        self.assertEqual(status, 403)

    def test_local_file_panel_can_read_only_aggregate_health(self):
        status, _, _ = self.request(
            "GET", "/api/health",
            headers={"Origin": "null", "Host": f"127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200)

    def test_local_file_panel_cannot_use_spoofed_host(self):
        status, _, _ = self.request(
            "GET", "/api/health",
            headers={"Origin": "null", "Host": f"attacker.example:{self.port}"},
        )
        self.assertEqual(status, 403)

    def test_local_file_panel_cannot_read_page_data(self):
        status, _, _ = self.request(
            "GET", "/api/pages",
            headers={"Origin": "null", "Host": f"127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 403)

    def test_unauthenticated_post_never_calls_mutation(self):
        with patch.object(srv, "api_update") as mutation:
            status, _, _ = self.request(
                "POST", "/api/update",
                body=json.dumps({"items": []}),
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 401)
        mutation.assert_not_called()

    def test_post_rejects_wrong_content_type(self):
        with patch.object(srv, "api_update") as mutation:
            status, _, _ = self.request(
                "POST", "/api/update", body="x",
                headers={
                    "Content-Type": "text/plain",
                    "X-Auth-Token": "integration-test-token",
                },
            )
        self.assertEqual(status, 415)
        mutation.assert_not_called()

    def test_post_requires_content_length(self):
        status, _, _ = self.request(
            "POST", "/api/update",
            headers={
                "Content-Type": "application/json",
                "X-Auth-Token": "integration-test-token",
            },
        )
        self.assertEqual(status, 411)

    def test_post_rejects_transfer_encoding(self):
        status, _, _ = self.request(
            "POST", "/api/update",
            headers={
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
                "X-Auth-Token": "integration-test-token",
            },
        )
        self.assertEqual(status, 501)

    def test_post_rejects_oversized_body_before_reading(self):
        status, _, _ = self.request(
            "POST", "/api/update",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(srv.MAX_BODY + 1),
                "X-Auth-Token": "integration-test-token",
            },
        )
        self.assertEqual(status, 413)

    def test_authenticated_post_reaches_mutation(self):
        with patch.object(srv, "api_update", return_value={"ok": True}) as mutation:
            status, _, body = self.request(
                "POST", "/api/update",
                body=json.dumps({"items": []}),
                headers={
                    "Content-Type": "application/json",
                    "X-Auth-Token": "integration-test-token",
                },
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {"ok": True})
        mutation.assert_called_once_with([])

    def test_export_filename_is_sanitized_in_real_response_header(self):
        malicious = "report" + '"' + chr(13) + chr(10) + "Set-Cookie: injected=1"
        path = "/api/export_page?name=" + quote(malicious, safe="")
        with patch.object(srv, "export_page", return_value=(b"safe", "text/plain")), \\
             patch.object(srv, "app_log"):
            status, headers, _ = self.request(
                "GET", path,
                headers={"X-Auth-Token": "integration-test-token"},
            )
        self.assertEqual(status, 200)
        disposition = headers.get("Content-Disposition", "")
        self.assertNotIn(chr(13), disposition)
        self.assertNotIn(chr(10), disposition)
        self.assertIsNone(headers.get("Set-Cookie"))
        self.assertIn("filename*=UTF-8''", disposition)


if __name__ == "__main__":
    unittest.main()
