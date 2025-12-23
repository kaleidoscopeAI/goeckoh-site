def _reset(self):
    _wipe_internal_state_for_tests()
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

def tearDown(self):
    self._reset()

@patch("colorama.ansitowin32.winapi_test", lambda: True)
def testJustFixWindowsConsole(self):
    if sys.platform != "win32":
        # just_fix_windows_console should be a no-op
        just_fix_windows_console()
        self.assertIs(sys.stdout, orig_stdout)
        self.assertIs(sys.stderr, orig_stderr)
    else:
        def fake_std():
            # Emulate stdout=not a tty, stderr=tty
            # to check that we handle both cases correctly
            stdout = Mock()
            stdout.closed = False
            stdout.isatty.return_value = False
            stdout.fileno.return_value = 1
            sys.stdout = stdout

            stderr = Mock()
            stderr.closed = False
            stderr.isatty.return_value = True
            stderr.fileno.return_value = 2
            sys.stderr = stderr

        for native_ansi in [False, True]:
            with patch(
                'colorama.ansitowin32.enable_vt_processing',
                lambda *_: native_ansi
            ):
                self._reset()
                fake_std()

                # Regular single-call test
                prev_stdout = sys.stdout
                prev_stderr = sys.stderr
                just_fix_windows_console()
                self.assertIs(sys.stdout, prev_stdout)
                if native_ansi:
                    self.assertIs(sys.stderr, prev_stderr)
                else:
                    self.assertIsNot(sys.stderr, prev_stderr)

                # second call without resetting is always a no-op
                prev_stdout = sys.stdout
                prev_stderr = sys.stderr
                just_fix_windows_console()
                self.assertIs(sys.stdout, prev_stdout)
                self.assertIs(sys.stderr, prev_stderr)

                self._reset()
                fake_std()

                # If init() runs first, just_fix_windows_console should be a no-op
                init()
                prev_stdout = sys.stdout
                prev_stderr = sys.stderr
                just_fix_windows_console()
                self.assertIs(prev_stdout, sys.stdout)
                self.assertIs(prev_stderr, sys.stderr)


