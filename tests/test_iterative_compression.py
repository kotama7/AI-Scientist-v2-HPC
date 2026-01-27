
import unittest
from unittest.mock import MagicMock, patch
from ai_scientist.memory.memgpt_store import _compress_with_llm

class TestIterativeCompression(unittest.TestCase):
    def test_iterative_compression_successful(self):
        """Test that compression loops until it fits."""
        max_chars = 10
        iters = 3
        
        # Mock LLM behavior:
        # Iter 1: Returns text length 20 (fails)
        # Iter 2: Returns text length 15 (fails)
        # Iter 3: Returns text length 8 (succeeds)
        mock_client = MagicMock()
        mock_model = "gpt-model"
        
        # We patch ai_scientist.llm.get_response_from_llm because it is imported from there
        with patch("ai_scientist.llm.get_response_from_llm") as mock_get_response:
            mock_get_response.side_effect = [
                ("A" * 20, None), # Iter 1
                ("B" * 15, None), # Iter 2
                ("C" * 8, None),  # Iter 3
            ]
            
            result = _compress_with_llm(
                text="Initial text that is way too long",
                max_chars=max_chars,
                context_hint="test",
                client=mock_client,
                model=mock_model,
                max_iterations=iters,
                use_cache=False 
            )
            
            self.assertEqual(result, "C" * 8)
            self.assertEqual(mock_get_response.call_count, 3)

    def test_iterative_compression_give_up(self):
        """Test that compression eventually truncates if it never fits."""
        max_chars = 10
        iters = 2

        # Mock LLM behavior: always returns too long
        mock_client = MagicMock()
        mock_model = "gpt-model"

        with patch("ai_scientist.llm.get_response_from_llm") as mock_get_response:
            # First iteration: returns shorter than input but still too long
            # Second iteration: returns same length or longer, triggers safety break
            mock_get_response.side_effect = [
                ("A" * 20, None),  # Iter 1: shorter than original but > max_chars
                ("B" * 20, None),  # Iter 2: same length as previous, safety break triggers
            ]

            result = _compress_with_llm(
                text="Initial text that is definitely longer than twenty characters",
                max_chars=max_chars,
                context_hint="test",
                client=mock_client,
                model=mock_model,
                max_iterations=iters,
                use_cache=False
            )

            # The safety check breaks when LLM returns text >= current length AND > max_chars.
            # In iter 2: len("B"*20) >= len("A"*20) AND len("B"*20) > 10, so it breaks
            # BEFORE assigning "B"*20 to current_text. So current_text remains "A"*20.
            # Final truncation happens on "A"*20 -> "A"*7 + "..."
            expected = "A" * 7 + "..."
            self.assertEqual(result, expected)
            self.assertEqual(mock_get_response.call_count, 2)

if __name__ == "__main__":
    unittest.main()
