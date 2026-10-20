import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from run_notified import load_email_credentials


class CredentialTests(unittest.TestCase):
    def test_literal_colon_format_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp)/'email.txt'
            path.write_text('GMAIL_USER: sender@example.com\nGMAIL_APP_PASSWORD: abcd efgh ijkl mnop\n')
            path.chmod(0o600)
            load_email_credentials(path)
            self.assertEqual(os.environ['GMAIL_USER'], 'sender@example.com')
            self.assertEqual(os.environ['GMAIL_APP_PASSWORD'], 'abcd efgh ijkl mnop')
            path.unlink()
            load_email_credentials(path)

    def test_invalid_file_does_not_leak_values_or_partially_load(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp)/'email.txt'
            path.write_text('GMAIL_USER: sender@example.com\nWRONG: fake-secret\n')
            path.chmod(0o600)
            with self.assertRaises(RuntimeError) as result:
                load_email_credentials(path)
            self.assertNotIn('fake-secret', str(result.exception))
            self.assertNotIn('GMAIL_USER', os.environ)

    def test_placeholder_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp)/'email.txt'
            path.write_text('GMAIL_USER: sender@example.com\nGMAIL_APP_PASSWORD: ***\n')
            path.chmod(0o600)
            with self.assertRaises(RuntimeError):
                load_email_credentials(path)


if __name__ == '__main__':
    unittest.main()
