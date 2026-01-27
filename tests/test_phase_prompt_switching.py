"""Tests for phase prompt switching based on memory.enabled configuration."""

import unittest

from ai_scientist.prompt_loader import load_prompt


class TestPhasePromptSwitching(unittest.TestCase):
    """Test cases for verifying prompt switching logic."""

    def test_phase0_prompts_exist(self):
        """Test that both Phase 0 prompts exist."""
        standard = load_prompt("config/phases/phase0_planning")
        memory = load_prompt("config/phases/phase0_planning_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_phase0_memory_prompt_has_memory_update(self):
        """Test that Phase 0 memory prompt contains memory_update instructions."""
        memory = load_prompt("config/phases/phase0_planning_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_phase0_standard_prompt_no_memory_update(self):
        """Test that Phase 0 standard prompt does not contain memory_update."""
        standard = load_prompt("config/phases/phase0_planning")
        # Standard prompt should not have memory_update blocks
        self.assertNotIn("<memory_update>", standard)

    def test_phase1_prompts_exist(self):
        """Test that both Phase 1 prompts exist."""
        standard = load_prompt("config/phases/phase1_installer")
        memory = load_prompt("config/phases/phase1_installer_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_phase1_memory_prompt_has_memory_update(self):
        """Test that Phase 1 memory prompt contains memory_update instructions."""
        memory = load_prompt("config/phases/phase1_installer_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_phase1_standard_prompt_no_memory_update(self):
        """Test that Phase 1 standard prompt does not contain memory_update."""
        standard = load_prompt("config/phases/phase1_installer")
        self.assertNotIn("<memory_update>", standard)


class TestTaskPromptSwitching(unittest.TestCase):
    """Test cases for task-level prompt switching."""

    def test_draft_prompts_exist(self):
        """Test that both draft prompts exist."""
        standard = load_prompt("agent/parallel/tasks/draft/introduction")
        memory = load_prompt("agent/parallel/tasks/draft/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_draft_memory_prompt_has_memory_update(self):
        """Test that draft memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/draft/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_debug_prompts_exist(self):
        """Test that both debug prompts exist."""
        standard = load_prompt("agent/parallel/tasks/debug/introduction")
        memory = load_prompt("agent/parallel/tasks/debug/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_debug_memory_prompt_has_memory_update(self):
        """Test that debug memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/debug/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_improve_prompts_exist(self):
        """Test that both improve prompts exist."""
        standard = load_prompt("agent/parallel/tasks/improve/introduction")
        memory = load_prompt("agent/parallel/tasks/improve/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_improve_memory_prompt_has_memory_update(self):
        """Test that improve memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/improve/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_summary_prompts_exist(self):
        """Test that both summary prompts exist."""
        standard = load_prompt("agent/parallel/tasks/summary/introduction")
        memory = load_prompt("agent/parallel/tasks/summary/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_summary_memory_prompt_has_memory_update(self):
        """Test that summary memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/summary/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_parse_metrics_prompts_exist(self):
        """Test that both parse_metrics prompts exist."""
        standard = load_prompt("agent/parallel/tasks/parse_metrics/introduction")
        memory = load_prompt("agent/parallel/tasks/parse_metrics/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_parse_metrics_memory_prompt_has_memory_update(self):
        """Test that parse_metrics memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/parse_metrics/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_execution_review_prompts_exist(self):
        """Test that both execution_review prompts exist."""
        standard = load_prompt("agent/parallel/tasks/execution_review/introduction")
        memory = load_prompt("agent/parallel/tasks/execution_review/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_execution_review_memory_prompt_has_memory_update(self):
        """Test that execution_review memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/tasks/execution_review/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())


class TestNodePromptSwitching(unittest.TestCase):
    """Test cases for node-level prompt switching (hyperparam, ablation)."""

    def test_hyperparam_prompts_exist(self):
        """Test that both hyperparam prompts exist."""
        standard = load_prompt("agent/parallel/nodes/hyperparam/introduction")
        memory = load_prompt("agent/parallel/nodes/hyperparam/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_hyperparam_memory_prompt_has_memory_update(self):
        """Test that hyperparam memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/nodes/hyperparam/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_ablation_prompts_exist(self):
        """Test that both ablation prompts exist."""
        standard = load_prompt("agent/parallel/nodes/ablation/introduction")
        memory = load_prompt("agent/parallel/nodes/ablation/introduction_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_ablation_memory_prompt_has_memory_update(self):
        """Test that ablation memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/nodes/ablation/introduction_with_memory")
        self.assertIn("memory_update", memory.lower())


class TestVLMPromptSwitching(unittest.TestCase):
    """Test cases for VLM analysis prompt switching."""

    def test_vlm_analysis_prompts_exist(self):
        """Test that both VLM analysis prompts exist."""
        standard = load_prompt("agent/parallel/vlm_analysis")
        memory = load_prompt("agent/parallel/vlm_analysis_with_memory")

        self.assertIsInstance(standard, str)
        self.assertIsInstance(memory, str)
        self.assertGreater(len(standard), 0)
        self.assertGreater(len(memory), 0)

    def test_vlm_analysis_memory_prompt_has_memory_update(self):
        """Test that VLM analysis memory prompt contains memory_update instructions."""
        memory = load_prompt("agent/parallel/vlm_analysis_with_memory")
        self.assertIn("memory_update", memory.lower())

    def test_vlm_analysis_has_memory_context_placeholder(self):
        """Test that VLM analysis prompts have memory_context_block placeholder."""
        standard = load_prompt("agent/parallel/vlm_analysis")
        memory = load_prompt("agent/parallel/vlm_analysis_with_memory")

        self.assertIn("{memory_context_block}", standard)
        self.assertIn("{memory_context_block}", memory)

    def test_vlm_analysis_has_task_desc_placeholder(self):
        """Test that VLM analysis prompts have task_desc placeholder."""
        standard = load_prompt("agent/parallel/vlm_analysis")
        memory = load_prompt("agent/parallel/vlm_analysis_with_memory")

        self.assertIn("{task_desc}", standard)
        self.assertIn("{task_desc}", memory)


class TestMemoryConfigPrompts(unittest.TestCase):
    """Test cases for memory configuration prompts."""

    def test_paper_section_prompts_exist(self):
        """Test that paper section prompts exist."""
        prompts = [
            "config/memory/paper_section_generation",
            "config/memory/paper_section_generation_system_message",
            "config/memory/paper_section_outline",
            "config/memory/paper_section_outline_system_message",
            "config/memory/paper_section_fill",
            "config/memory/paper_section_fill_system_message",
        ]
        for prompt_name in prompts:
            content = load_prompt(prompt_name)
            self.assertIsInstance(content, str, f"Failed for {prompt_name}")
            self.assertGreater(len(content), 0, f"Empty content for {prompt_name}")

    def test_keyword_extraction_prompts_exist(self):
        """Test that keyword extraction prompts exist."""
        prompt = load_prompt("config/memory/keyword_extraction")
        system_msg = load_prompt("config/memory/keyword_extraction_system_message")

        self.assertIsInstance(prompt, str)
        self.assertIsInstance(system_msg, str)
        self.assertGreater(len(prompt), 0)
        self.assertGreater(len(system_msg), 0)

    def test_compression_prompts_exist(self):
        """Test that compression prompts exist."""
        prompt = load_prompt("config/memory/compression")
        system_msg = load_prompt("config/memory/compression_system_message")

        self.assertIsInstance(prompt, str)
        self.assertIsInstance(system_msg, str)
        self.assertGreater(len(prompt), 0)
        self.assertGreater(len(system_msg), 0)


if __name__ == "__main__":
    unittest.main()
