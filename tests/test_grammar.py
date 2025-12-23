import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from grammar import correct_text

class TestGrammar(unittest.TestCase):

    def test_pronoun_flipping(self):
        # High GCL (calm), so correction should happen
        gcl = 0.8
        self.assertEqual(correct_text("you are happy", gcl), "I am happy")
        self.assertEqual(correct_text("are you sad", gcl), "am I sad")
        self.assertEqual(correct_text("your toy", gcl), "my toy")
        self.assertEqual(correct_text("You are my friend", gcl), "I am my friend")
    
    def test_single_word_expansion(self):
        gcl = 0.8
        self.assertEqual(correct_text("ball", gcl), "I want the ball")
        self.assertEqual(correct_text("car", gcl), "I want the car")

    def test_low_gcl_safety_mode(self):
        # Low GCL (stressed), so it should just repeat
        gcl = 0.2
        self.assertEqual(correct_text("you are angry", gcl), "you are angry")
        self.assertEqual(correct_text("your book", gcl), "your book")
        self.assertEqual(correct_text("go", gcl), "go")

    def test_empty_and_none_input(self):
        gcl = 0.8
        self.assertIsNone(correct_text(None, gcl))
        self.assertIsNone(correct_text("", gcl))
        self.assertIsNone(correct_text("   ", gcl))

if __name__ == '__main__':
    unittest.main()
