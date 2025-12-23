"""Capture the output of sys.stdout:

   with captured_stdout() as stdout:
       print('hello')
   self.assertEqual(stdout.getvalue(), 'hello\n')

Taken from Lib/support/__init__.py in the CPython repo.
"""
return captured_output("stdout")


