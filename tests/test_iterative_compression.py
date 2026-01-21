
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
            mock_get_response.side_effect = [
                ("A" * 20, None), 
                ("B" * 20, None), 
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
            
            # Should fall back to truncation of the last result ("B" * 20) -> "B" * 10
            # Wait, implementing checks: if len(compressed) > max_chars, it tries next iter.
            # After max_iters loop finishes, "current_text" is the last response.
            # Then we verify if len(current_text) > max_chars -> truncate.
            # So expected is truncated last response.
            # _truncate adds "..." if truncated. max_chars=10 -> 7 chars + "..."
            expected = "B" * 7 + "..."
            self.assertEqual(result, expected)
            self.assertEqual(mock_get_response.call_count, 2)

if __name__ == "__main__":
    unittest.main()
