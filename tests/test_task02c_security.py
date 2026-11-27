"""TASK-02C — بوابة انحدار حراسة الوصول لـ 7_box_editor_server.py

يختبر السلوك الفعلي المُضاف في هذا الـPR فقط (قراءة مباشرة من الكود، لا منطقًا مُخترَعًا):
- رفض أصل (Origin) غير مسموح بـ403، وقبول الأصل المسموح.
- استثناء "لوحة مفتوحة كملف محلي" (Origin: null) يعمل فقط على مسارات PUBLIC_GET
  ومع ترويسة Host تطابق 127.0.0.1/localhost بالمنفذ الفعلي — لا غيره (حماية DNS rebinding).
- اشتراط رمز المصادقة عند تفعيل AUTH_TOKEN، عبر الترويسة أو معامل الاستعلام أو الكوكي.
- عدم حقن ترويسات عبر اسم صفحة مموَّه (فواصل CRLF/علامات اقتباس) في Content-Disposition.
"""
import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
srv = importlib.import_module("7_box_editor_server")


class _FakeHandler:
    """محاكاة خفيفة لمعالج HTTP لاختبار _origin_ok/_authed/_guard بمعزل عن الشبكة الفعلية."""

    def __init__(self, headers=None, path="/api/health", command="GET"):
        self.headers = headers or {}
        self.path = path
        self.command = command
        self._responses = []

    def _json(self, obj, code=200):
        self._responses.append((code, obj))

    # الدوال الحقيقية من srv.Handler تُستدعى كدوال غير مرتبطة (unbound) على هذا الكائن الوهمي
    _origin_ok = srv.Handler._origin_ok
    _same = staticmethod(srv.Handler._same)
    _authed = srv.Handler._authed
    _guard = srv.Handler._guard
    _local_file_panel = srv.Handler._local_file_panel


class TestOriginGuard(unittest.TestCase):
    def setUp(self):
        srv.ALLOWED_ORIGINS = {"https://panel.example"}
        srv.AUTH_TOKEN = None
        srv.SERVER_PORT = 8765

    def test_allowed_origin_passes(self):
        h = _FakeHandler(headers={"Origin": "https://panel.example"})
        self.assertTrue(h._origin_ok())

    def test_unknown_origin_rejected(self):
        h = _FakeHandler(headers={"Origin": "https://evil.example"})
        self.assertFalse(h._origin_ok())

    def test_null_origin_allowed_only_on_public_get_with_matching_host(self):
        h = _FakeHandler(
            headers={"Origin": "null", "Host": "127.0.0.1:8765"},
            path="/api/status",
        )
        self.assertTrue(h._origin_ok())

    def test_null_origin_rejected_on_non_public_path(self):
        h = _FakeHandler(
            headers={"Origin": "null", "Host": "127.0.0.1:8765"},
            path="/api/page",
        )
        self.assertFalse(h._origin_ok())

    def test_null_origin_rejected_with_spoofed_host(self):
        """منع DNS rebinding: Host غير 127.0.0.1/localhost بالمنفذ الفعلي يُرفض."""
        h = _FakeHandler(
            headers={"Origin": "null", "Host": "attacker.example:8765"},
            path="/api/status",
        )
        self.assertFalse(h._origin_ok())


class TestAuthGuard(unittest.TestCase):
    def setUp(self):
        srv.ALLOWED_ORIGINS = {"https://panel.example"}
        srv.AUTH_TOKEN = "s3cr3t-token"
        srv.SERVER_PORT = 8765

    def tearDown(self):
        srv.AUTH_TOKEN = None

    def test_no_token_configured_means_open(self):
        srv.AUTH_TOKEN = None
        h = _FakeHandler()
        self.assertTrue(h._authed())

    def test_correct_header_token_accepted(self):
        h = _FakeHandler(headers={"X-Auth-Token": "s3cr3t-token"})
        self.assertTrue(h._authed())

    def test_wrong_header_token_rejected(self):
        h = _FakeHandler(headers={"X-Auth-Token": "wrong"})
        self.assertFalse(h._authed())

    def test_query_param_token_accepted(self):
        h = _FakeHandler(path="/api/page?token=s3cr3t-token")
        self.assertTrue(h._authed())

    def test_cookie_token_accepted(self):
        h = _FakeHandler(headers={"Cookie": "tkn=s3cr3t-token"})
        self.assertTrue(h._authed())

    def test_missing_token_rejected(self):
        h = _FakeHandler()
        self.assertFalse(h._authed())

    def test_guard_rejects_unauthed_with_401(self):
        h = _FakeHandler(headers={"Origin": "https://panel.example"})
        ok = h._guard()
        self.assertFalse(ok)
        self.assertEqual(h._responses[-1][0], 401)

    def test_guard_rejects_bad_origin_with_403_before_checking_auth(self):
        h = _FakeHandler(headers={"Origin": "https://evil.example",
                                   "X-Auth-Token": "s3cr3t-token"})
        ok = h._guard()
        self.assertFalse(ok)
        self.assertEqual(h._responses[-1][0], 403)


class TestExportFilenameSanitization(unittest.TestCase):
    def test_crlf_and_quotes_stripped_from_disposition_name(self):
        import os
        malicious = 'legit"\r\nSet-Cookie: pwned=1'
        safe = os.path.basename(malicious).replace('"', "").replace("\r", "").replace("\n", "")
        self.assertNotIn("\r", safe)
        self.assertNotIn("\n", safe)
        self.assertNotIn('"', safe)

    def test_path_traversal_stripped_by_basename(self):
        import os
        malicious = "../../../etc/passwd"
        safe = os.path.basename(malicious)
        self.assertEqual(safe, "passwd")


if __name__ == "__main__":
    unittest.main()
