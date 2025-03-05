# podcast2md

A powerful command-line tool for transcribing podcasts into beautifully formatted Markdown files.

## Overview

podcast2md leverages OpenAI's Whisper for offline speech-to-text conversion, creating high-quality podcast transcripts formatted in Markdown. It extracts metadata, organizes content into paragraphs, and offers rich formatting options.

## Features

- **Powerful Transcription**: Uses OpenAI Whisper for accurate, local speech-to-text conversion
- **Flexible Input**: Process local audio files or download from URLs
- **Metadata Extraction**: Automatically extracts podcast metadata and cover art
- **Intelligent Formatting**:
  - Smart paragraph organization
  - Section detection
  - Clean, readable output
- **Enhanced Formatting Options**:
  - Bold, italic formatting for technical terms
  - Obsidian-compatible wiki links
  - External reference footnotes
- **Obsidian Integration**:
  - Link to existing files in your vault
  - Uses shortest matching filenames
  - Perfect for building a podcast knowledge base

## Prerequisites

- Python 3.12+
- FFmpeg (required for audio processing)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

No installation required! Simply use `uvx` to run the tool directly from the repository.

Make sure you have FFmpeg installed:

**macOS**:
```bash
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows**:
Download from the [FFmpeg website](https://ffmpeg.org/download.html) or install via Chocolatey:
```bash
choco install ffmpeg
```

## Usage

First create an alias for **podcast2md**:


```bash
alias podcast2md="uvx --with git+https://github.com/reuteras/podcast2md podcast2md"
```

### Basic Usage

Transcribe a local podcast file:
```bash
podcast2md my_podcast.mp3
```

Transcribe from a URL:
```bash
podcast2md https://example.com/podcast.mp3
```

### Output Options

Specify output location:
```bash
podcast2md podcast.mp3 --output transcripts/
podcast2md podcast.mp3 --output transcripts/episode42.md
```

### Model Selection

Choose the Whisper model size based on your needs for accuracy vs. speed:
```bash
podcast2md podcast.mp3 --model tiny    # Fastest, least accurate
podcast2md podcast.mp3 --model base    # Good balance (default)
podcast2md podcast.mp3 --model medium  # Better accuracy
podcast2md podcast.mp3 --model large   # Best accuracy, slowest
```

### Formatting Options

Apply rich text formatting:
```bash
podcast2md podcast.mp3 --format
```

Create Obsidian-style wiki links:
```bash
podcast2md podcast.mp3 --links
```

Add external references as footnotes:
```bash
podcast2md podcast.mp3 --refs
```

Link to existing files in Obsidian vault:
```bash
podcast2md podcast.mp3 --links --vault /path/to/obsidian/vault
```

Combine multiple options:
```bash
podcast2md podcast.mp3 --format --links --refs --vault /path/to/obsidian/vault
```

## Example Output

A typical output file includes:

1. **Header with metadata** (title, artist, date, etc.)
2. **Cover image** (if available)
3. **Content sections** with well-formatted paragraphs
4. **References section** with footnotes (if enabled)

## Customization

If you wish to customize the tool, you can clone the repository and modify:
- Paragraph length and sentence thresholds
- Technical terms for formatting
- External reference links
- Default formatting patterns

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [mutagen](https://github.com/quodlibet/mutagen) for metadata extraction
- [pydub](https://github.com/jiaaro/pydub) for audio processing
- [uv](https://github.com/astral-sh/uv) for fast Python package management
