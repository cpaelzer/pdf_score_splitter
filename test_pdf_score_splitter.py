"""Tests for the PDF score splitter - with mocked Copilot to avoid API costs.

Copyright (c) 2026 Christian Ehrhardt <paelzer@gmail.com>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import Mock, patch

import pytest

from pdf_score_splitter import (
    InstrumentPage,
    SplitCommand,
    build_copilot_prompt,
    check_dependencies,
    extract_json_from_response,
    group_pages_by_instrument,
    process_instrument_results,
    sanitize_filename,
)


class TestSanitizeFilename:
    """Test filename sanitization."""

    def test_basic_sanitization(self):
        assert sanitize_filename("1. Klarinette Bb") == "1._Klarinette_Bb"
        assert sanitize_filename("Flöte") == "Flöte"
        assert sanitize_filename("E-Bass") == "E-Bass"

    def test_special_characters(self):
        assert sanitize_filename("Vocal/Gesang") == "VocalGesang"
        assert sanitize_filename("Test (with) [brackets]") == "Test_with_brackets"

    def test_spaces(self):
        assert sanitize_filename("Multiple   Spaces") == "Multiple_Spaces"
        assert sanitize_filename("  Leading Trailing  ") == "Leading_Trailing"


class TestExtractJsonFromResponse:
    """Test JSON extraction from Copilot responses."""

    def test_json_in_code_block(self):
        response = '```json\n{"1": "Flöte", "2": "Oboe"}\n```'
        result = extract_json_from_response(response)
        assert result == '{"1": "Flöte", "2": "Oboe"}'

    def test_json_without_code_block(self):
        response = 'Here is the result: {"1": "Flöte", "2": "Oboe"} and more text'
        result = extract_json_from_response(response)
        assert result == '{"1": "Flöte", "2": "Oboe"}'

    def test_nested_json(self):
        response = '{"1": {"name": "Flöte"}, "2": "Oboe"}'
        result = extract_json_from_response(response)
        assert result == '{"1": {"name": "Flöte"}, "2": "Oboe"}'

    def test_no_json(self):
        response = "This has no JSON at all"
        result = extract_json_from_response(response)
        assert result is None


class TestProcessInstrumentResults:
    """Test instrument result processing."""

    def test_basic_processing(self):
        instruments = {1: "Flöte", 2: "Oboe", 3: "Fagott"}
        results = process_instrument_results(instruments, 3)

        assert len(results) == 3
        assert results[0].instrument == "Flöte"
        assert results[1].instrument == "Oboe"
        assert results[2].instrument == "Fagott"
        assert not any(r.is_continuation for r in results)

    def test_continuation_pages(self):
        instruments = {1: "Keyboard", 2: "CONTINUATION", 3: "Percussion"}
        results = process_instrument_results(instruments, 3)

        assert results[0].instrument == "Keyboard"
        assert results[1].instrument == "Keyboard"
        assert results[1].is_continuation
        assert results[2].instrument == "Percussion"

    def test_unknown_handling(self):
        instruments = {1: "Flöte", 2: "UNKNOWN"}
        results = process_instrument_results(instruments, 2)

        # UNKNOWN becomes continuation of previous instrument
        assert results[1].instrument == "Flöte"
        assert results[1].is_continuation


class TestGroupPagesByInstrument:
    """Test grouping pages into split commands."""

    def test_single_page_instruments(self):
        pages = [
            InstrumentPage(1, "Flöte"),
            InstrumentPage(2, "Oboe"),
            InstrumentPage(3, "Fagott"),
        ]
        commands = group_pages_by_instrument(pages)

        assert len(commands) == 3
        assert commands[0].start_page == 1
        assert commands[0].end_page == 1
        assert commands[0].output_file == "Flöte.pdf"
        assert commands[0].output_dir is None

    def test_multi_page_instruments(self):
        pages = [
            InstrumentPage(1, "Keyboard"),
            InstrumentPage(2, "Keyboard", is_continuation=True),
            InstrumentPage(3, "Percussion"),
        ]
        commands = group_pages_by_instrument(pages)

        assert len(commands) == 2
        assert commands[0].start_page == 1
        assert commands[0].end_page == 2
        assert commands[0].output_file == "Keyboard.pdf"
        assert commands[0].output_dir is None

    def test_empty_pages(self):
        commands = group_pages_by_instrument([])
        assert commands == []

    def test_group_with_output_dir(self):
        from pathlib import Path
        pages = [
            InstrumentPage(1, "Flöte"),
            InstrumentPage(2, "Oboe"),
        ]
        output_dir = Path("/tmp/output")
        commands = group_pages_by_instrument(pages, output_dir)

        assert len(commands) == 2
        assert commands[0].output_dir == output_dir
        assert commands[1].output_dir == output_dir
        assert commands[0].output_file == "Flöte.pdf"
        assert commands[1].output_file == "Oboe.pdf"


class TestSplitCommand:
    """Test split command generation."""

    def test_single_page_command(self):
        cmd = SplitCommand("Flöte", 1, 1, "Flöte.pdf", None)
        pdftk_cmd = cmd.to_pdftk_command("Noten.pdf")

        assert pdftk_cmd == ["pdftk", "Noten.pdf", "cat", "1", "output", "Flöte.pdf"]

    def test_multi_page_command(self):
        cmd = SplitCommand("Percussion", 47, 49, "Percussion.pdf", None)
        pdftk_cmd = cmd.to_pdftk_command("Noten.pdf")

        assert pdftk_cmd == ["pdftk", "Noten.pdf", "cat", "47-49", "output", "Percussion.pdf"]

    def test_command_with_output_dir(self):
        from pathlib import Path
        output_dir = Path("/tmp/output")
        cmd = SplitCommand("Flöte", 1, 1, "Flöte.pdf", output_dir)
        pdftk_cmd = cmd.to_pdftk_command("Noten.pdf")

        assert pdftk_cmd == ["pdftk", "Noten.pdf", "cat", "1", "output", "/tmp/output/Flöte.pdf"]


class TestBuildCopilotPrompt:
    """Test Copilot prompt building."""

    def test_prompt_structure(self):
        page_texts = {1: "Flöte text here", 2: "Oboe text here"}
        prompt = build_copilot_prompt(page_texts)

        assert "wind band" in prompt.lower()
        assert "Page 1:" in prompt
        assert "Page 2:" in prompt
        assert "Flöte text here" in prompt
        assert "JSON" in prompt

    def test_prompt_truncates_text(self):
        long_text = "█" * 1000  # Use a unique character
        page_texts = {1: long_text}
        prompt = build_copilot_prompt(page_texts)

        # Should only include first 400 chars
        assert prompt.count("█") == 400


@pytest.fixture
def mock_subprocess_run():
    """Mock subprocess.run to avoid actual system calls."""
    with patch("pdf_score_splitter.subprocess.run") as mock:
        yield mock


class TestCheckDependencies:
    """Test dependency checking."""

    def test_all_dependencies_present(self, mock_subprocess_run):
        # All commands succeed
        mock_subprocess_run.return_value = Mock(returncode=0)

        # Should not raise
        check_dependencies()

    def test_missing_dependency(self, mock_subprocess_run):
        # Simulate missing command
        mock_subprocess_run.side_effect = FileNotFoundError()

        with pytest.raises(Exception) as exc_info:
            check_dependencies()

        assert "Missing required dependencies" in str(exc_info.value)


class TestAskCopilotBatch:
    """Test Copilot batch processing."""

    def test_successful_request(self, mock_subprocess_run):
        """Test successful Copilot request."""
        mock_subprocess_run.return_value = Mock(
            returncode=0,
            stdout='```json\n{"1": "Flöte", "2": "Oboe"}\n```',
            stderr="",
        )

        from pdf_score_splitter import ask_copilot_batch

        page_texts = {1: "Flöte text", 2: "Oboe text"}
        result = ask_copilot_batch(page_texts)

        assert result == {1: "Flöte", 2: "Oboe"}

    def test_copilot_timeout(self, mock_subprocess_run):
        """Test timeout handling."""
        from pdf_score_splitter import PDFAnalyzerError, ask_copilot_batch

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("copilot", 60)

        with pytest.raises(PDFAnalyzerError) as exc_info:
            ask_copilot_batch({1: "test"})

        assert "timed out" in str(exc_info.value)

    def test_copilot_not_found(self, mock_subprocess_run):
        """Test handling when copilot command not found."""
        from pdf_score_splitter import PDFAnalyzerError, ask_copilot_batch

        mock_subprocess_run.side_effect = FileNotFoundError()

        with pytest.raises(PDFAnalyzerError) as exc_info:
            ask_copilot_batch({1: "test"})

        assert "not found" in str(exc_info.value)

    def test_copilot_error_code(self, mock_subprocess_run):
        """Test handling when copilot returns error code."""
        from pdf_score_splitter import PDFAnalyzerError, ask_copilot_batch

        mock_subprocess_run.return_value = Mock(returncode=1, stdout="", stderr="Auth error")

        with pytest.raises(PDFAnalyzerError) as exc_info:
            ask_copilot_batch({1: "test"})

        assert "error" in str(exc_info.value).lower()

    def test_invalid_json_response(self, mock_subprocess_run):
        """Test handling of invalid JSON in response."""
        from pdf_score_splitter import PDFAnalyzerError, ask_copilot_batch

        mock_subprocess_run.return_value = Mock(
            returncode=0, stdout="This is not JSON at all", stderr=""
        )

        with pytest.raises(PDFAnalyzerError) as exc_info:
            ask_copilot_batch({1: "test"})

        assert "JSON" in str(exc_info.value)

    def test_malformed_json(self, mock_subprocess_run):
        """Test handling of malformed JSON."""
        from pdf_score_splitter import PDFAnalyzerError, ask_copilot_batch

        mock_subprocess_run.return_value = Mock(
            returncode=0, stdout='{"1": "Flöte", invalid}', stderr=""
        )

        with pytest.raises(PDFAnalyzerError) as exc_info:
            ask_copilot_batch({1: "test"})

        assert "parse" in str(exc_info.value).lower()


class TestGetPageCount:
    """Test PDF page count extraction."""

    def test_successful_page_count(self, mock_subprocess_run):
        """Test extracting page count from pdfinfo."""
        from pathlib import Path

        from pdf_score_splitter import get_page_count

        mock_subprocess_run.return_value = Mock(
            returncode=0, stdout="Pages:          42\nOther: info", stderr=""
        )

        count = get_page_count(Path("test.pdf"))
        assert count == 42

    def test_pdfinfo_error(self, mock_subprocess_run):
        """Test handling of pdfinfo error."""
        from pathlib import Path

        from pdf_score_splitter import PDFAnalyzerError, get_page_count

        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            1, "pdfinfo", stderr="Invalid PDF"
        )

        with pytest.raises(PDFAnalyzerError) as exc_info:
            get_page_count(Path("bad.pdf"))

        assert "Failed to read PDF" in str(exc_info.value)

    def test_pdfinfo_timeout(self, mock_subprocess_run):
        """Test handling of pdfinfo timeout."""
        from pathlib import Path

        from pdf_score_splitter import PDFAnalyzerError, get_page_count

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("pdfinfo", 10)

        with pytest.raises(PDFAnalyzerError) as exc_info:
            get_page_count(Path("slow.pdf"))

        assert "Timeout" in str(exc_info.value)

    def test_no_page_count_in_output(self, mock_subprocess_run):
        """Test handling when page count not found in output."""
        from pathlib import Path

        from pdf_score_splitter import PDFAnalyzerError, get_page_count

        mock_subprocess_run.return_value = Mock(returncode=0, stdout="No pages here", stderr="")

        with pytest.raises(PDFAnalyzerError) as exc_info:
            get_page_count(Path("weird.pdf"))

        assert "Could not determine page count" in str(exc_info.value)


class TestExtractTextFromPage:
    """Test OCR text extraction."""

    def test_successful_ocr(self, mock_subprocess_run):
        """Test successful OCR extraction."""
        from pathlib import Path

        from pdf_score_splitter import extract_text_from_page

        # Mock both convert and tesseract
        mock_subprocess_run.return_value = Mock(returncode=0, stdout="Flöte\n1. Stimme", stderr="")

        text = extract_text_from_page(Path("test.pdf"), 1)
        assert "Flöte" in text

    def test_ocr_error(self, mock_subprocess_run):
        """Test OCR error handling (should not raise, just return empty)."""
        from pathlib import Path

        from pdf_score_splitter import extract_text_from_page

        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "tesseract", stderr=b"OCR failed")

        # Should not raise, just return empty string
        text = extract_text_from_page(Path("test.pdf"), 1)
        assert text == ""

    def test_ocr_timeout(self, mock_subprocess_run):
        """Test OCR timeout handling (should not raise, just return empty)."""
        from pathlib import Path

        from pdf_score_splitter import extract_text_from_page

        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("tesseract", 30)

        # Should not raise, just return empty string
        text = extract_text_from_page(Path("test.pdf"), 1)
        assert text == ""


class TestIntegrationWithMockedCopilot:
    """Integration tests with mocked Copilot API."""

    @pytest.fixture
    def mock_copilot_response(self):
        """Mock Copilot CLI response."""
        return Mock(
            returncode=0,
            stdout=json.dumps(
                {
                    "1": "Flöte",
                    "2": "Oboe",
                    "3": "1. Klarinette Bb",
                    "4": "CONTINUATION",
                }
            ),
            stderr="",
        )

    def test_end_to_end_with_mock(self, mock_subprocess_run, mock_copilot_response):
        """Test the full flow with mocked Copilot."""
        # Setup mocks
        mock_subprocess_run.return_value = mock_copilot_response

        from pdf_score_splitter import ask_copilot_batch

        page_texts = {
            1: "Flöte arrangement...",
            2: "Oboe arrangement...",
            3: "1. Klarinette in Bb...",
            4: "Continuation of klarinette...",
        }

        result = ask_copilot_batch(page_texts)

        assert result[1] == "Flöte"
        assert result[2] == "Oboe"
        assert result[3] == "1. Klarinette Bb"
        assert result[4] == "CONTINUATION"
