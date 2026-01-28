# Wind Band PDF Score Splitter

AI-powered tool to automatically split wind band/concert band PDF scores into separate files for each instrument part using OCR and GitHub Copilot.

## Features

- 🤖 **AI-Powered**: Uses GitHub Copilot to intelligently identify instruments
- ⚡ **Efficient**: Makes only 1 API request per PDF (not per page)
- 🌍 **Multi-language**: Handles German and English instrument names
- 🔧 **Robust**: Deals with OCR errors and spelling variations
- ✅ **Interactive**: Confirms before execution and reports results clearly
- 📊 **Comprehensive**: Recognizes most wind band instruments (PRs welcome if it misses yours)

## Prerequisites

### 1. System Dependencies

The tool requires several system utilities for PDF processing and OCR
and to later install copilot CLI.

#### Step 1.1: Install System Dependencies

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    python3 \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    imagemagick \
    poppler-utils \
    pdftk-java \
    npm
```

#### Step 1.2: Set Up UV and Python Virtual Environment (Optional, for Development)

**Note**: The main tool (`pdf_score_splitter.py`) has no Python dependencies and can run with just `python3` and the system tools. The virtual environment is only needed for development (running tests, using ruff).

```bash
# Install uv via snap
sudo snap install astral-uv --classic

# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install development dependencies
uv pip install -e ".[dev]"
```

### 2. GitHub Copilot CLI

You need the GitHub Copilot CLI installed and authenticated:

```bash
# Install GitHub Copilot CLI
# Follow instructions at: https://github.com/github/copilot-cli
# Usually via npm: npm install -g @github/copilot

# After installation, authenticate (this will open a browser)
copilot

# Verify it works
copilot --help
```

**Important**: This tool uses GitHub Copilot which requires:
- A GitHub account with Copilot access
- The Copilot CLI to be installed and logged in
  - No environment variables needed - authentication is handled by the CLI itself

### Verifying Installation

The tool will check for all dependencies when you run it:

```bash
python3 pdf_score_splitter.py
# If dependencies are missing, it will tell you exactly what to install
```

## Usage

### Basic Usage

```bash
# Analyze and split a PDF (will ask for confirmation)
python3 pdf_score_splitter.py --pdf your_score.pdf

# Skip confirmation prompt and execute immediately
python3 pdf_score_splitter.py --pdf your_score.pdf --yes
```

### What Happens on execution

1. **Dependency Check**: Verifies all required tools are installed
2. **OCR Extraction**: Converts each PDF page to an image and extracts text (shows progress)
3. **AI Analysis**: Sends all OCR text in one batch to GitHub Copilot (1 API call)
4. **Instrument Recognition**: Copilot identifies each instrument using contextual understanding
5. **Confirmation**: Shows you what will be created and asks for confirmation (unless --yes)
6. **Execution**: Runs pdftk commands to split the PDF
7. **Results**: Reports successes and any failures with actionable guidance

### Example Output

```
python3 pdf_score_splitter.py --pdf score.pdf
Checking dependencies...
✓ All dependencies found

Processing 49 pages from 'score.pdf'...

Step 1: Extracting text with OCR...
  ✓ OCR complete for 49 pages.

Step 2: Analyzing all pages with Copilot (1 request)...
  Sending batch request to Copilot...
  Prompt includes 22,882 characters
  Received response: 1,168 characters
  Successfully parsed 49 instrument assignments

Step 3: Processing results...
  Page 1: Flöte
  Page 2: Oboe
  Page 3: Fagott
...
  Page 43: Keyboard
  Page 44: Keyboard (continuation)
  Page 45: Gitarre
  Page 46: Vocal/Gesang
  Page 47: Percussion
  Page 48: Percussion
  Page 49: Percussion (continuation)

================================================================================
Ready to split 'score.pdf' into 46 files:
================================================================================
  Flöte.pdf                                (page 1)
  Oboe.pdf                                 (page 2)
  Fagott.pdf                               (page 3)
...
  Keyboard.pdf                             (pages 43-44)
  Gitarre.pdf                              (page 45)
  VocalGesang.pdf                          (page 46)
  Percussion.pdf                           (pages 47-49)
================================================================================

Proceed with splitting? [Y/n]: Y

Executing 46 split commands...

[1/46] Creating Flöte.pdf... ✓
[2/46] Creating Oboe.pdf... ✓
[3/46] Creating Fagott.pdf... ✓
...
[45/46] Creating VocalGesang.pdf... ✓
[46/46] Creating Percussion.pdf... ✓

================================================================================
✓ Successfully created 46 files
================================================================================

✓ All files created successfully!
```

## Development

### Setup for Development

First, set up your development environment:

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install development dependencies
uv pip install -e ".[dev]"
```

**Important**: Always activate the virtual environment before running development tools (pytest, ruff). The main tool works without it, but dev tools require it.

### Running Tests

The project includes comprehensive tests with mocked Copilot API calls (no tokens consumed):

```bash
# Make sure you're in the virtual environment first!
source .venv/bin/activate

# Run tests with coverage
pytest

# Run tests with detailed output
pytest -v

# Run tests and show coverage report
pytest --cov=pdf_score_splitter --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=pdf_score_splitter --cov-report=html
# Then open htmlcov/index.html in your browser
```

### Code Quality

The project uses `ruff` for linting and formatting:

```bash
# Make sure you're in the virtual environment first!
source .venv/bin/activate

# Check code style
ruff check pdf_score_splitter.py test_pdf_score_splitter.py

# Auto-fix issues
ruff check --fix pdf_score_splitter.py test_pdf_score_splitter.py

# Format code
ruff format pdf_score_splitter.py test_pdf_score_splitter.py
```

### Development Workflow

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Make changes to the code

# 3. Run ruff to ensure code quality
ruff check --fix . && ruff format .

# 4. Run tests to verify functionality
pytest

# 5. Check coverage
pytest --cov=pdf_score_splitter --cov-report=term-missing
```

## Troubleshooting

### "GitHub Copilot CLI not found"
- Install the Copilot CLI: https://github.com/github/gh-copilot
- Make sure it's in your PATH
- Run `copilot` once to authenticate

### "Missing required dependencies"
The error message will tell you exactly what's missing and how to install it:
```
Missing required dependencies:
  - tesseract: Install with: apt-get install tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng
  - pdftk: Install with: apt-get install pdftk
```

### "Failed to read PDF file"
- Ensure the PDF file exists and is not corrupted
- Try opening it in a PDF reader to verify it's valid
- Check file permissions

### "OCR failed for page X"
- The page may have very poor scan quality
- The tool will continue with other pages
- Check the specific page manually if needed

### Poor instrument recognition
- OCR quality depends on scan quality - higher resolution scans work better
- The AI is usually robust to errors, but very poor scans may confuse it
- You can manually check and adjust any misidentified files afterward

## License

MIT License - Use freely for personal and educational purposes with legally obtained scores!

## Contributing

Contributions welcome! Please:
1. Run `ruff check --fix . && ruff format .` before committing
2. Ensure all tests pass: `pytest`
3. Add tests for new functionality
4. Update README if adding features
