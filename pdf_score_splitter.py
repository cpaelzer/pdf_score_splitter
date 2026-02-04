"""Wind Band PDF Score Splitter - AI-powered instrument recognition and splitting.

Copyright (c) 2026 Christian Ehrhardt <paelzer@gmail.com>
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class InstrumentPage:
    """Represents an instrument assignment for a page."""

    page_num: int
    instrument: str
    is_continuation: bool = False


@dataclass
class SplitCommand:
    """Represents a pdftk split command."""

    instrument: str
    start_page: int
    end_page: int
    output_file: str
    output_dir: Path | None = None

    def to_pdftk_command(self, input_pdf: str) -> list[str]:
        """Convert to pdftk command arguments."""
        page_range = (
            str(self.start_page)
            if self.start_page == self.end_page
            else f"{self.start_page}-{self.end_page}"
        )
        output_path = str(self.output_dir / self.output_file) if self.output_dir else self.output_file
        return ["pdftk", input_pdf, "cat", page_range, "output", output_path]


class PDFAnalyzerError(Exception):
    """Base exception for PDF analyzer errors."""

    pass


class DependencyError(PDFAnalyzerError):
    """Raised when required dependencies are missing."""

    pass


def check_dependencies() -> None:
    """Check that all required system dependencies are available."""
    dependencies = {
        "pdfinfo": "Install with: apt-get install poppler-utils",
        "convert": "Install with: apt-get install imagemagick",
        "tesseract": "Install with: apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng",
        "pdftk": "Install with: apt-get install pdftk",
        "copilot": "Install GitHub Copilot CLI from: https://github.com/github/gh-copilot",
    }

    missing = []
    for cmd, install_msg in dependencies.items():
        try:
            subprocess.run(
                [cmd, "--help" if cmd != "pdfinfo" else "--version"],
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            missing.append(f"  - {cmd}: {install_msg}")

    if missing:
        raise DependencyError("Missing required dependencies:\n" + "\n".join(missing))


def get_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except subprocess.CalledProcessError as e:
        raise PDFAnalyzerError(
            f"Failed to read PDF file '{pdf_path}'. Is it a valid PDF? Error: {e.stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise PDFAnalyzerError(
            f"Timeout reading PDF file '{pdf_path}'. File may be corrupted."
        ) from e

    match = re.search(r"Pages:\s+(\d+)", result.stdout)
    if not match:
        raise PDFAnalyzerError(
            f"Could not determine page count for '{pdf_path}'. "
            f"The PDF may be corrupted or in an unsupported format."
        )
    return int(match.group(1))


def extract_text_from_page(pdf_path: Path, page_num: int) -> str:
    """Extract text from a specific PDF page using OCR."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_file:
            img_path = Path(img_file.name)

        try:
            # Convert PDF page to image
            subprocess.run(
                [
                    "convert",
                    "-density",
                    "150",
                    f"{pdf_path}[{page_num - 1}]",
                    "-background",
                    "white",
                    "-flatten",
                    str(img_path),
                ],
                capture_output=True,
                check=True,
                timeout=30,
            )

            # OCR the image
            result = subprocess.run(
                ["tesseract", str(img_path), "stdout", "-l", "deu+eng", "--psm", "4"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

            return result.stdout.strip()

        finally:
            img_path.unlink(missing_ok=True)

    except subprocess.CalledProcessError as e:
        print(f"  Warning: OCR failed for page {page_num}: {e.stderr[:100]}", file=sys.stderr)
        return ""
    except subprocess.TimeoutExpired:
        print(f"  Warning: OCR timeout for page {page_num}", file=sys.stderr)
        return ""


def build_copilot_prompt(page_texts: dict[int, str]) -> str:
    """Build the batch prompt for Copilot analysis."""
    prompt = """I am analyzing a wind band/concert band music score PDF. Each page has been OCR'd (may contain recognition errors).

TASK: Identify which instrument each page is for based on the OCR text.

IMPORTANT CONTEXT:
- Instrument names typically appear in the first few lines of a page
- OCR may have errors (character substitutions, spacing issues)
- Names can be in German, English, or mixed
- Spelling may vary (e.g., "Saxophon" vs "Saxofon", "Klarinette" vs "Clarinette")
- Instruments may be abbreviated (e.g., "Kl." = Klarinette, "Trp." = Trompete, "Pos." = Posaune)
- Look for context clues like part numbers (1., 2., 3., 4.) and keys (Bb, Eb, F, C)

COMMON WIND BAND INSTRUMENTS (with variations):
Woodwinds:
- Flöte/Flute, Piccolo/Pikkoloflöte
- Oboe, Englischhorn/English Horn
- Fagott/Bassoon, Kontrafagott
- Klarinette/Clarinet/Clarinette (Bb/Eb/A), Bass-Klarinette

Saxophones (spelling: Saxophon/Saxofon/Saxophone):
- Sopran-Saxophon/Soprano Sax
- Alt-Saxophon/Alto Sax (Eb)
- Tenor-Saxophon/Tenor Sax (Bb)
- Bariton-Saxophon/Baritone Sax (Eb)

Brass:
- Flügelhorn/Flugelhorn (Bb)
- Trompete/Trumpet/Cornetto (Bb/C)
- Horn/Waldhorn/French Horn (F/Eb/Bb)
- Posaune/Trombone (C/Bb), Tenorposaune, Bassposaune
- Bariton/Euphonium/Tenorhorn (C/Bb)
- Tuba/Bass (C/Eb/Bb/F)

Other:
- E-Bass/Electric Bass/Bass-Gitarre
- Gitarre/Guitar
- Keyboard/Piano/Klavier/Synthesizer
- Vocal/Gesang/Voice/Chor/Choir

Percussion (many variations possible):
- Schlagzeug/Percussion/Drums (general)
- Pauke/Timpani/Pauken
- Glockenspiel
- Xylophon/Xylophone, Marimba, Vibraphon/Vibraphone
- Triangel/Triangle, Becken/Cymbals, Tamburin/Tambourine
- Kleine Trommel/Snare Drum, Große Trommel/Bass Drum
- Tom-Toms, Congas, Bongos, Claves, Maracas
- (Other percussion instruments as context suggests)

SPECIAL CASES:
- CONTINUATION: page with no instrument header (continuation of previous instrument)
- Multiple pages per instrument are common
- Some scores have numbered parts (1. Klarinette, 2. Klarinette, etc.)
- Keys after instrument name: Bb (B♭), Eb (E♭), F, C, A

OCR TEXT FROM EACH PAGE (first 400 characters):

"""

    for page_num, text in sorted(page_texts.items()):
        preview = text[:400].replace("\n", " ")
        prompt += f"\nPage {page_num}: {preview}\n"

    prompt += """

INSTRUCTIONS:
1. Look at the OCR text for each page (focus on first few lines where instrument names appear)
2. Identify the instrument, handling OCR errors and spelling variations flexibly
3. Include part numbers (1., 2., etc.) and keys (Bb, Eb, F, C) when present
4. For continuation pages (no instrument header), return "CONTINUATION"
5. Only use "UNKNOWN" if you truly cannot determine the instrument

OUTPUT FORMAT:
Respond with a JSON object mapping page numbers to instrument designations.

Example format:
{
  "1": "Flöte",
  "2": "1. Klarinette Bb",
  "3": "2. Klarinette Bb",
  "4": "CONTINUATION",
  "5": "Percussion",
  ...
}

Be flexible with instrument names - use the clearest German or English name based on what you see.
For percussion, use the specific instrument name if clear (e.g., "Glockenspiel", "Pauke"),
or "Percussion" for general percussion parts.

JSON response:"""

    return prompt


def ask_copilot_batch(page_texts: dict[int, str]) -> dict[int, str]:
    """Make a single Copilot request for all pages at once."""
    prompt = build_copilot_prompt(page_texts)

    print("  Sending batch request to Copilot...", file=sys.stderr)
    print(f"  Prompt includes {len(prompt):,} characters", file=sys.stderr)

    try:
        result = subprocess.run(
            ["copilot", "--allow-all-tools"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        raise PDFAnalyzerError(
            "Copilot request timed out after 60 seconds. "
            "Try with a smaller PDF or check your internet connection."
        ) from e
    except FileNotFoundError as e:
        raise PDFAnalyzerError(
            "GitHub Copilot CLI not found. "
            "Please install it and authenticate: https://github.com/github/gh-copilot"
        ) from e

    if result.returncode != 0:
        raise PDFAnalyzerError(
            f"Copilot CLI returned error (code {result.returncode}). "
            f"Make sure you're logged in with 'copilot' or check your authentication. "
            f"Error: {result.stderr[:200]}"
        )

    response = result.stdout
    print(f"  Received response: {len(response):,} characters", file=sys.stderr)

    # Extract JSON from response
    json_str = extract_json_from_response(response)
    if not json_str:
        raise PDFAnalyzerError(
            "Could not extract valid JSON from Copilot response. "
            "This may be a temporary issue - try running again."
        )

    try:
        instruments = json.loads(json_str)
        print(f"  Successfully parsed {len(instruments)} instrument assignments", file=sys.stderr)
        return {int(k): v for k, v in instruments.items()}
    except json.JSONDecodeError as e:
        raise PDFAnalyzerError(
            f"Failed to parse JSON from Copilot: {e}. "
            f"Response may be malformed - try running again."
        ) from e


def extract_json_from_response(response: str) -> str | None:
    """Extract JSON object from Copilot response."""
    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find a complete JSON object by counting braces
    brace_count = 0
    json_start = -1

    for i, char in enumerate(response):
        if char == "{":
            if brace_count == 0:
                json_start = i
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0 and json_start != -1:
                return response[json_start : i + 1]

    return None


def process_instrument_results(
    instruments_dict: dict[int, str], total_pages: int
) -> list[InstrumentPage]:
    """Process Copilot results and handle continuation pages."""
    results: list[InstrumentPage] = []
    last_known_instrument: str | None = None

    for page_num in range(1, total_pages + 1):
        instrument = instruments_dict.get(page_num, "UNKNOWN")
        is_continuation = False

        if instrument == "CONTINUATION":
            if last_known_instrument:
                instrument = last_known_instrument
                is_continuation = True
                print(f"  Page {page_num}: {instrument} (continuation)", file=sys.stderr)
            else:
                instrument = f"Unknown_Page_{page_num}"
                print(f"  Page {page_num}: {instrument}", file=sys.stderr)
        elif instrument == "UNKNOWN":
            if last_known_instrument:
                instrument = last_known_instrument
                is_continuation = True
                print(f"  Page {page_num}: {instrument} (assumed continuation)", file=sys.stderr)
            else:
                instrument = f"Unknown_Page_{page_num}"
                print(f"  Page {page_num}: {instrument}", file=sys.stderr)
        else:
            last_known_instrument = instrument
            print(f"  Page {page_num}: {instrument}", file=sys.stderr)

        results.append(InstrumentPage(page_num, instrument, is_continuation))

    return results


def group_pages_by_instrument(pages: list[InstrumentPage], output_dir: Path | None = None) -> list[SplitCommand]:
    """Group consecutive pages by instrument into split commands."""
    if not pages:
        return []

    commands: list[SplitCommand] = []
    current_instrument = pages[0].instrument
    start_page = pages[0].page_num

    for i, page in enumerate(pages[1:], start=1):
        if page.instrument != current_instrument:
            # End current group
            safe_name = sanitize_filename(current_instrument)
            commands.append(
                SplitCommand(
                    current_instrument, start_page, pages[i - 1].page_num, f"{safe_name}.pdf", output_dir
                )
            )
            # Start new group
            current_instrument = page.instrument
            start_page = page.page_num

    # Add final group
    safe_name = sanitize_filename(current_instrument)
    commands.append(
        SplitCommand(current_instrument, start_page, pages[-1].page_num, f"{safe_name}.pdf", output_dir)
    )

    return commands


def sanitize_filename(instrument: str) -> str:
    """Sanitize instrument name for use as filename."""
    safe_name = re.sub(r"[^\w\s\.-]", "", instrument)
    safe_name = re.sub(r"\s+", "_", safe_name)
    return safe_name.strip("_.")


def confirm_execution(commands: list[SplitCommand], pdf_path: Path) -> bool:
    """Ask user to confirm execution of split commands."""
    print("\n" + "=" * 80)
    print(f"Ready to split '{pdf_path.name}' into {len(commands)} files:")
    print("=" * 80)

    for cmd in commands:
        page_info = (
            f"page {cmd.start_page}"
            if cmd.start_page == cmd.end_page
            else f"pages {cmd.start_page}-{cmd.end_page}"
        )
        print(f"  {cmd.output_file:40s} ({page_info})")

    print("=" * 80)
    response = input("\nProceed with splitting? [Y/n]: ").strip().lower()
    return response in ("", "y", "yes")


def execute_split_commands(
    commands: list[SplitCommand], pdf_path: Path
) -> tuple[list[SplitCommand], list[tuple[SplitCommand, str]]]:
    """Execute pdftk commands and track successes/failures."""
    successes: list[SplitCommand] = []
    failures: list[tuple[SplitCommand, str]] = []

    print(f"\nExecuting {len(commands)} split commands...\n")

    for i, cmd in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] Creating {cmd.output_file}...", end=" ", file=sys.stderr)

        try:
            subprocess.run(
                cmd.to_pdftk_command(str(pdf_path)),
                capture_output=True,
                check=True,
                timeout=30,
            )
            print("✓", file=sys.stderr)
            successes.append(cmd)
        except subprocess.CalledProcessError as e:
            error_msg = (
                e.stderr.decode("utf-8", errors="replace")[:200] if e.stderr else "Unknown error"
            )
            print(f"✗ Failed: {error_msg}", file=sys.stderr)
            failures.append((cmd, error_msg))
        except subprocess.TimeoutExpired:
            print("✗ Timeout", file=sys.stderr)
            failures.append((cmd, "Command timed out after 30 seconds"))

    return successes, failures


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split wind band PDF scores by instrument using AI-powered OCR analysis"
    )
    parser.add_argument("--pdf", type=Path, default=Path("Noten.pdf"), help="PDF file to analyze")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt and execute immediately"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output directory for split PDF files (default: current directory)"
    )
    args = parser.parse_args()

    try:
        # Check dependencies
        print("Checking dependencies...", file=sys.stderr)
        check_dependencies()
        print("✓ All dependencies found\n", file=sys.stderr)

        # Validate PDF
        if not args.pdf.exists():
            raise PDFAnalyzerError(
                f"PDF file not found: {args.pdf}\nPlease check the path and try again."
            )

        # Create output directory if specified
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            print(f"Output directory: {args.out}\n", file=sys.stderr)

        # Get page count
        total_pages = get_page_count(args.pdf)
        print(f"Processing {total_pages} pages from '{args.pdf}'...\n", file=sys.stderr)

        # Step 1: Extract OCR text
        print("Step 1: Extracting text with OCR...", file=sys.stderr)
        page_texts: dict[int, str] = {}
        for page_num in range(1, total_pages + 1):
            print(f"  OCR page {page_num}/{total_pages}...", end="\r", file=sys.stderr)
            text = extract_text_from_page(args.pdf, page_num)
            page_texts[page_num] = text
        print(f"  ✓ OCR complete for {total_pages} pages.          ", file=sys.stderr)

        # Step 2: Analyze with Copilot
        print("\nStep 2: Analyzing all pages with Copilot (1 request)...", file=sys.stderr)
        instruments_dict = ask_copilot_batch(page_texts)

        # Step 3: Process results
        print("\nStep 3: Processing results...", file=sys.stderr)
        instrument_pages = process_instrument_results(instruments_dict, total_pages)

        # Step 4: Group into commands
        commands = group_pages_by_instrument(instrument_pages, args.out)

        # Step 5: Confirm and execute
        if not args.yes and not confirm_execution(commands, args.pdf):
            print("\nCancelled by user.")
            return 1

        successes, failures = execute_split_commands(commands, args.pdf)

        # Report results
        print("\n" + "=" * 80)
        print(f"✓ Successfully created {len(successes)} files")
        if failures:
            print(f"✗ Failed to create {len(failures)} files:")
            for cmd, error in failures:
                print(f"  - {cmd.output_file}: {error}")
            print("\nPlease check these pages manually:")
            for cmd, _ in failures:
                page_info = (
                    f"page {cmd.start_page}"
                    if cmd.start_page == cmd.end_page
                    else f"pages {cmd.start_page}-{cmd.end_page}"
                )
                print(f"  - {cmd.instrument} ({page_info})")
            return 1

        print("=" * 80)
        print("\n✓ All files created successfully!")
        return 0

    except PDFAnalyzerError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        print("Please report this issue with the full error message.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
