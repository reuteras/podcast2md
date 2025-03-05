"""podcast2md"""
import os
import whisper
import argparse
import re
import requests
import tempfile
import warnings
from urllib.parse import urlparse
import mutagen
import glob
import shutil

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

def is_url(path):
    """Check if the provided path is a URL"""
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_file(url, output_dir=None):
    """Download file from URL to the specified output directory"""
    if output_dir is None:
        output_dir = os.getcwd()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the filename from the URL
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)

    # If no filename in URL, create one
    if not file_name:
        file_name = "podcast_download.mp3"
    elif not os.path.splitext(file_name)[1]:
        file_name += ".mp3"  # Add extension if missing

    output_file = os.path.join(output_dir, file_name)

    print(f"Downloading audio from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {output_file}")
    return output_file, url

def extract_metadata(audio_file):
    """Extract metadata from audio file"""
    metadata = {}
    output_dir = os.path.dirname(os.path.abspath(audio_file))

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    try:
        audio = mutagen.File(audio_file)
        if audio is not None:
            # Extract common metadata fields
            if hasattr(audio, 'tags') and audio.tags:
                for key in audio.tags.keys():
                    metadata[key] = str(audio.tags[key])

            # For MP3 files (ID3 tags)
            if hasattr(audio, 'ID3'):
                for key in ['TIT2', 'TPE1', 'TALB', 'TDRC', 'TCON', 'COMM']:
                    if key in audio:
                        metadata[key] = str(audio[key])

            # For MP4/M4A files
            if hasattr(audio, 'items'):
                for key, value in audio.items():
                    metadata[key] = str(value)

            # Look for chapters in metadata
            chapters = []
            if hasattr(audio, 'chapters'):
                chapters = audio.chapters
            elif 'CHAP' in metadata:
                chapters.append(metadata['CHAP'])
    except Exception as e:
        print(f"Error extracting metadata: {e}")

    # Create readable metadata dictionary
    readable_metadata = {}
    readable_metadata['title'] = metadata.get('TIT2', metadata.get('©nam', metadata.get('title', '')))
    readable_metadata['artist'] = metadata.get('TPE1', metadata.get('©ART', metadata.get('artist', '')))
    readable_metadata['album'] = metadata.get('TALB', metadata.get('©alb', metadata.get('album', '')))
    readable_metadata['date'] = metadata.get('TDRC', metadata.get('©day', metadata.get('date', '')))
    readable_metadata['genre'] = metadata.get('TCON', metadata.get('©gen', metadata.get('genre', '')))
    readable_metadata['comment'] = metadata.get('COMM', metadata.get('©cmt', metadata.get('comment', '')))

    # Extract chapters if available
    if 'chapters' in locals() and chapters:
        readable_metadata['chapters'] = chapters

    # Extract cover art if available
    try:
        cover_path = None
        if hasattr(audio, 'pictures'):
            for p in audio.pictures:
                cover_data = p.data
                break
        elif hasattr(audio, 'tags') and hasattr(audio.tags, 'getall'):
            apic = audio.tags.getall('APIC')
            if apic:
                cover_data = apic[0].data
        elif 'covr' in metadata:
            cover_data = metadata['covr']

        if 'cover_data' in locals():
            base_filename = os.path.basename(os.path.splitext(audio_file)[0])
            cover_filename = f"{base_filename}_cover.jpg"

            # Save to the images directory instead of the same directory
            cover_path = os.path.join(images_dir, cover_filename)
            with open(cover_path, 'wb') as f:
                f.write(cover_data)

            # Store the relative path to the image for markdown
            readable_metadata['cover_image'] = os.path.join("images", cover_filename)
    except Exception as e:
        print(f"Error extracting cover art: {e}")

    return readable_metadata

def transcribe_audio(audio_file, model_size="base"):
    """
    Transcribe an audio file using OpenAI's Whisper model locally.

    Parameters:
    audio_file (str): Path to the audio file
    model_size (str): Size of the Whisper model to use
                      Options: "tiny", "base", "small", "medium", "large"

    Returns:
    dict: Transcription results including text
    """
    print(f"Loading Whisper {model_size} model...")
    model = whisper.load_model(model_size)

    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)

    return result

def identify_sections(segments, metadata):
    """
    Identify section breaks in the transcript based on metadata

    Parameters:
    segments (list): List of transcript segments with timestamps
    metadata (dict): Metadata containing chapter information

    Returns:
    list: List of (timestamp, title) tuples for each section
    """
    sections = []

    # Extract chapter information from metadata if available
    if 'chapters' in metadata and metadata['chapters']:
        chapters = metadata['chapters']
        for chapter in chapters:
            # Format depends on the specific metadata format, adjust as needed
            if hasattr(chapter, 'start') and hasattr(chapter, 'title'):
                sections.append((chapter.start, chapter.title))
            elif isinstance(chapter, dict) and 'start' in chapter and 'title' in chapter:
                sections.append((chapter['start'], chapter['title']))

    # If no chapters in metadata, attempt to detect section breaks heuristically
    if not sections and len(segments) > 10:
        # Look for longer pauses between segments
        for i in range(1, len(segments)):
            prev_end = segments[i-1]['end']
            curr_start = segments[i]['start']

            # If there's a significant pause (more than 2 seconds)
            if curr_start - prev_end > 2.0:
                # Look for section indicator phrases
                text = segments[i]['text'].strip()
                section_indicators = [
                    "chapter", "section", "part", "episode",
                    "segment", "act", "introduction", "conclusion"
                ]

                if any(indicator in text.lower() for indicator in section_indicators):
                    sections.append((segments[i]['start'], text))

    return sections

def format_paragraphs(segments):
    """Break text into reasonably sized paragraphs without extra line breaks"""
    # First, join all text
    full_text = ""
    for segment in segments:
        text = segment['text'].strip()

        # Ensure there's a space between segments but no extra line breaks
        if full_text and not full_text.endswith(" "):
            full_text += " "

        full_text += text

    # Clean up extra whitespace
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    # Split into sentences (basic rule: split on . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    paragraphs = []
    current_para = []
    current_length = 0
    min_sentences = 2
    max_length = 500

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > max_length and len(current_para) >= min_sentences:
            # Start a new paragraph if current one is getting too long
            paragraphs.append(' '.join(current_para))
            current_para = [sentence]
            current_length = sentence_length
        else:
            # Add to current paragraph
            current_para.append(sentence)
            current_length += sentence_length

    # Add the last paragraph if there's anything left
    if current_para:
        paragraphs.append(' '.join(current_para))

    return paragraphs

def get_output_filename(input_path, output_path=None):
    """Determine the output filename based on input and output paths"""
    if output_path:
        # If output_path is a directory, use the base name of the input file
        if os.path.isdir(output_path):
            base_name = os.path.basename(os.path.splitext(input_path)[0])
            return os.path.join(output_path, f"{base_name}.md")
        # If output_path has an extension, use it as is
        elif os.path.splitext(output_path)[1]:
            return output_path
        # If output_path has no extension, add .md
        else:
            return f"{output_path}.md"
    else:
        # Use the current directory with the base name of the input file
        base_name = os.path.basename(os.path.splitext(input_path)[0])
        return os.path.join(os.getcwd(), f"{base_name}.md")

def get_existing_vault_files(vault_path):
    """Get a list of existing markdown files in the Obsidian vault"""
    if not vault_path or not os.path.isdir(vault_path):
        return []

    # Find all markdown files in the vault
    md_files = []
    for ext in ['.md', '.markdown']:
        md_files.extend(glob.glob(f"{vault_path}/**/*{ext}", recursive=True))

    # Extract just the base filename without extension
    return [os.path.splitext(os.path.basename(f))[0] for f in md_files]

def find_best_match(term, existing_files):
    """Find the shortest matching existing file for a term"""
    if not existing_files:
        return None

    # Convert term to lowercase and remove special characters
    clean_term = re.sub(r'[^a-zA-Z0-9\s]', '', term.lower())

    # Check for exact matches first (with shortest length if multiple matches)
    exact_matches = [file for file in existing_files if file.lower() == clean_term]
    if exact_matches:
        return min(exact_matches, key=len)

    # Check for terms that are contained in file names (shortest match wins)
    contained_matches = [file for file in existing_files if clean_term in file.lower()]
    if contained_matches:
        return min(contained_matches, key=len)

    # Check for files that contain parts of the term (shortest match wins)
    term_parts = clean_term.split()
    if len(term_parts) > 1:
        part_matches = []
        for file in existing_files:
            if any(part in file.lower() for part in term_parts):
                part_matches.append(file)
        if part_matches:
            return min(part_matches, key=len)

    return None

def apply_text_formatting(text, apply_formatting=False):
    """Apply rich text formatting to the transcript text"""
    if not apply_formatting:
        return text

    # Define patterns for various types of content to format
    patterns = [
        # Format Windows executables (.exe files) as bold
        (r'\b([a-zA-Z0-9]+\.exe)\b', r'**\1**'),
        (r'\b(smss|csrss|winlogon|lsass|services|svchost|explorer|userinit|winit|wininet|lsm)(?:\.exe)?\b', r'**\1.exe**'),

        # Format Windows system terms and concepts as italic
        (r'\b(kernel|system idle|idle process|system process|session manager subsystem|client server runtime subsystem|windows initialization|service control manager|local security authority subsystem|credential guard)\b', r'*\1*'),

        # Format technical abbreviations as bold-italic
        (r'\b(SMSS|CSRSS|LSASS|PID|DFIR|DLL|UI|LSM|SAM|LSAISO)\b', r'***\1***'),

        # Format registry references
        (r'\b(registry (key|hive)s?|HKEY_[A-Z_]+)\b', r'*\1*'),
    ]

    # Apply each pattern
    formatted_text = text
    for pattern, replacement in patterns:
        formatted_text = re.sub(pattern, replacement, formatted_text, flags=re.IGNORECASE)

    return formatted_text

def add_obsidian_links(text, create_links=False, vault_files=None):
    """
    Add Obsidian-style links to relevant terms in format [[filename|display text]]

    Parameters:
    text (str): Text to process
    create_links (bool): Whether to create links
    vault_files (list): List of existing markdown files in the vault

    Returns:
    str: Text with Obsidian links
    """
    if not create_links:
        return text

    # Define key terms and their destinations
    link_patterns = [
        # Core Windows processes
        (r'\b(smss\.exe|session manager subsystem)\b', 'Windows Process - SMSS', 'Session Manager Subsystem'),
        (r'\b(csrss\.exe|client server runtime subsystem)\b', 'Windows Process - CSRSS', 'Client Server Runtime Subsystem'),
        (r'\b(winlogon\.exe)\b', 'Windows Process - WinLogon', 'WinLogon'),
        (r'\b(lsass\.exe|local security authority subsystem service)\b', 'Windows Process - LSASS', 'Local Security Authority Subsystem'),
        (r'\b(services\.exe|service control manager)\b', 'Windows Process - Services', 'Service Control Manager'),
        (r'\b(svchost\.exe|service host)\b', 'Windows Process - SvcHost', 'Service Host'),
        (r'\b(explorer\.exe)\b', 'Windows Process - Explorer', 'Windows Explorer'),

        # DFIR concepts
        (r'\b(DFIR|digital forensics|incident response)\b', 'Digital Forensics and Incident Response', 'DFIR'),
        (r'\b(triage)\b', 'DFIR Triage Process', 'Triage'),
        (r'\b(windows internals)\b', 'Windows Internals', 'Windows Internals'),
        (r'\b(persistence)\b', 'Malware Persistence Techniques', 'Persistence'),
        (r'\bmalware\b', 'Malware Analysis', 'Malware'),

        # Registry related
        (r'\b(registry)\b', 'Windows Registry', 'Registry'),
        (r'\b(active directory)\b', 'Active Directory', 'Active Directory'),
        (r'\b(SAM database)\b', 'Security Accounts Manager', 'SAM Database'),
    ]

    # Apply each link pattern
    linked_text = text
    for pattern, default_page, display_text in link_patterns:
        def replace_with_link(match):
            # If we have vault files, try to find a matching file
            if vault_files:
                best_match = find_best_match(default_page, vault_files)
                if best_match:
                    page_name = best_match
                else:
                    page_name = default_page
            else:
                page_name = default_page

            # Use the original matched text as display_text if it's the same term
            if match.group(0).lower() == display_text.lower():
                return f"[[{page_name}|{match.group(0)}]]"
            else:
                return f"[[{page_name}|{match.group(0)}]]"

        linked_text = re.sub(pattern, replace_with_link, linked_text, flags=re.IGNORECASE)

    return linked_text

def add_external_references(text, create_refs=False, footnote_map=None):
    """
    Add external references in the form Term[^1] and collect footnotes
    Ensures footnotes are applied AFTER any Obsidian links

    Parameters:
    text (str): Text to process
    create_refs (bool): Whether to create external references
    footnote_map (dict): Map of terms to footnote numbers

    Returns:
    tuple: (processed text, dict of term->footnote mappings)
    """
    if not create_refs:
        return text, footnote_map or {}

    if footnote_map is None:
        footnote_map = {}

    # Define key terms and their external references
    reference_patterns = [
        ('Windows', 'https://www.microsoft.com/windows/'),
        ('Microsoft', 'https://www.microsoft.com/'),
        ('DFIR', 'https://www.sans.org/digital-forensics/'),
        ('forensics', 'https://www.sans.org/digital-forensics/'),
        ('malware', 'https://www.malwarebytes.com/'),
        ('active directory', 'https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/get-started/virtual-dc/active-directory-domain-services-overview'),
        ('registry', 'https://learn.microsoft.com/en-us/windows/win32/sysinfo/registry'),
        ('Autopsy', 'https://www.autopsy.com/'),
        ('CyberTriage', 'https://www.cybertriage.com/'),
        ('Atola', 'https://atola.com/'),
        ('smss.exe', 'https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/session-manager-system-processes'),
        ('lsass.exe', 'https://learn.microsoft.com/en-us/windows-server/security/credentials-protection-and-management/configuring-additional-lsa-protection'),
        ('svchost.exe', 'https://learn.microsoft.com/en-us/windows-server/security/windows-services/service-isolation-and-security-hardening'),
    ]

    # Process the text to handle Obsidian links properly
    processed_text = text

    # Helper function to avoid adding footnotes inside Obsidian links
    def process_references():
        nonlocal processed_text

        # Split the text by Obsidian links
        parts = re.split(r'(\[\[[^\]]+\]\])', processed_text)
        result = []

        for i, part in enumerate(parts):
            # Skip Obsidian links
            if i % 2 == 1:
                result.append(part)
                continue

            # Process regular text
            part_text = part
            for term, url in reference_patterns:
                # Skip if the term is not in this part (case insensitive)
                if not re.search(r'\b' + re.escape(term) + r'\b', part_text, re.IGNORECASE):
                    continue

                # Add a footnote only for the first occurrence of each term
                if term.lower() not in footnote_map:
                    footnote_map[term.lower()] = len(footnote_map) + 1

                footnote_number = footnote_map[term.lower()]

                # Replace only the first occurrence
                pattern = r'\b' + re.escape(term) + r'\b'
                replacement = r'\g<0>[^' + str(footnote_number) + ']'
                part_text = re.sub(pattern, replacement, part_text, count=1, flags=re.IGNORECASE)

            result.append(part_text)

        processed_text = ''.join(result)

    process_references()

    return processed_text, footnote_map

def post_process_transcript(paragraphs, apply_formatting=False, create_links=False, create_refs=False, vault_files=None):
    """Apply post-processing to transcript paragraphs"""
    processed_paragraphs = []
    footnote_map = {}

    for paragraph in paragraphs:
        # Apply formatting (bold, italic) if requested
        if apply_formatting:
            paragraph = apply_text_formatting(paragraph, apply_formatting)

        # Add Obsidian-style links if requested
        if create_links:
            paragraph = add_obsidian_links(paragraph, create_links, vault_files)

        # Add external references if requested
        if create_refs:
            paragraph, footnote_map = add_external_references(paragraph, create_refs, footnote_map)

        processed_paragraphs.append(paragraph)

    return processed_paragraphs, footnote_map

def save_transcript_markdown(result, audio_file, metadata=None, output_path=None, source_url=None,
                             apply_formatting=False, create_links=False, create_refs=False, vault_path=None):
    """Save transcription to a markdown file with metadata and paragraphs"""
    output_file = get_output_filename(audio_file, output_path)
    output_dir = os.path.dirname(output_file)

    # Create images directory if it doesn't exist
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Get existing vault files if vault path is provided
    vault_files = get_existing_vault_files(vault_path) if vault_path else None
    if vault_files:
        print(f"Found {len(vault_files)} markdown files in the vault")

    # Extract segments from result
    segments = result["segments"]

    # Identify sections based on metadata
    sections = identify_sections(segments, metadata or {})

    # Initialize footnotes container
    all_footnotes = {}

    # Format text into paragraphs
    if not sections:
        # If no sections, format entire transcript
        paragraphs = format_paragraphs(segments)
        # Apply post-processing
        paragraphs, footnotes = post_process_transcript(
            paragraphs, apply_formatting, create_links, create_refs, vault_files
        )
        all_footnotes.update(footnotes)
        section_content = [("Transcript", paragraphs)]
    else:
        # If sections exist, format each section separately
        section_content = []
        current_section_name = "Introduction"
        current_section_segments = []

        # Sort sections by timestamp
        sections.sort(key=lambda x: x[0])

        # Process segments by section
        for i, segment in enumerate(segments):
            # Check if this segment starts a new section
            new_section = False
            for sec_time, sec_title in sections:
                if segment['start'] >= sec_time and (i == 0 or segments[i-1]['start'] < sec_time):
                    # If we already have content, add the current section
                    if current_section_segments:
                        section_paragraphs = format_paragraphs(current_section_segments)
                        # Apply post-processing
                        section_paragraphs, footnotes = post_process_transcript(
                            section_paragraphs, apply_formatting, create_links, create_refs, vault_files
                        )
                        all_footnotes.update(footnotes)
                        section_content.append((current_section_name, section_paragraphs))

                    # Start new section
                    current_section_name = sec_title
                    current_section_segments = [segment]
                    new_section = True
                    break

            if not new_section:
                current_section_segments.append(segment)

        # Add the last section
        if current_section_segments:
            section_paragraphs = format_paragraphs(current_section_segments)
            # Apply post-processing
            section_paragraphs, footnotes = post_process_transcript(
                section_paragraphs, apply_formatting, create_links, create_refs, vault_files
            )
            all_footnotes.update(footnotes)
            section_content.append((current_section_name, section_paragraphs))

    # Prepare footnotes for writing
    footnote_references = []
    for term, footnote_num in all_footnotes.items():
        # Find the URL for this term
        reference_patterns = [
            ('Windows', 'https://www.microsoft.com/windows/'),
            ('Microsoft', 'https://www.microsoft.com/'),
            ('DFIR', 'https://www.sans.org/digital-forensics/'),
            ('forensics', 'https://www.sans.org/digital-forensics/'),
            ('malware', 'https://www.malwarebytes.com/'),
            ('active directory', 'https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/get-started/virtual-dc/active-directory-domain-services-overview'),
            ('registry', 'https://learn.microsoft.com/en-us/windows/win32/sysinfo/registry'),
            ('Autopsy', 'https://www.autopsy.com/'),
            ('CyberTriage', 'https://www.cybertriage.com/'),
            ('Atola', 'https://atola.com/'),
            ('smss.exe', 'https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/session-manager-system-processes'),
            ('lsass.exe', 'https://learn.microsoft.com/en-us/windows-server/security/credentials-protection-and-management/configuring-additional-lsa-protection'),
            ('svchost.exe', 'https://learn.microsoft.com/en-us/windows-server/security/windows-services/service-isolation-and-security-hardening'),
        ]

        for ref_term, url in reference_patterns:
            if term.lower() == ref_term.lower():
                footnote_references.append((footnote_num, url))
                break

    with open(output_file, "w", encoding="utf-8") as f:
        # Add metadata header
        f.write("# Podcast Transcript\n\n")

        if metadata:
            if metadata.get('title'):
                f.write(f"## {metadata['title']}\n\n")

            # Add metadata section
            f.write("### Metadata\n\n")
            for key, value in metadata.items():
                if key not in ['cover_image', 'chapters'] and value:  # Skip cover image path and chapters
                    f.write(f"- **{key.capitalize()}**: {value}\n")

            # Add source URL if available
            if source_url:
                f.write(f"- **Source URL**: {source_url}\n")

            f.write("\n")

            # Add cover image if available with path to images directory
            if 'cover_image' in metadata:
                cover_rel_path = metadata['cover_image']
                f.write(f"![Podcast Cover]({cover_rel_path})\n\n")
        elif source_url:
            # If no metadata but we have a URL
            f.write("### Source\n\n")
            f.write(f"- **URL**: {source_url}\n\n")

        # Add transcript with sections if available
        f.write("## Content\n\n")

        for section_name, paragraphs in section_content:
            if section_name != "Transcript":  # Don't add heading for generic transcript
                f.write(f"### {section_name}\n\n")

            for paragraph in paragraphs:
                f.write(f"{paragraph}\n\n")

        # Add footnotes if any
        if footnote_references:
            f.write("\n## References\n\n")
            # Sort by footnote number
            for number, url in sorted(footnote_references):
                f.write(f"[^{number}]: {url}\n")

    # Move any remaining images to the images directory and update references
    for file in os.listdir(output_dir):
        if file.endswith(('.jpg', '.jpeg', '.png', '.gif')) and file != "images" and os.path.isfile(os.path.join(output_dir, file)):
            src_path = os.path.join(output_dir, file)
            dst_path = os.path.join(images_dir, file)

            # Move the file
            shutil.move(src_path, dst_path)

            # Update image references in the markdown file
            with open(output_file, 'r') as f:
                content = f.read()

            # Replace direct file references with references to images directory
            updated_content = re.sub(
                r'!\[([^\]]*)\]\(([^)]+)\)',
                lambda m: f'![{m.group(1)}](images/{os.path.basename(m.group(2))})' if os.path.basename(m.group(2)) == file else m.group(0),
                content
            )

            with open(output_file, 'w') as f:
                f.write(updated_content)

            print(f"Moved image {file} to images directory and updated references")

    print(f"Markdown transcript saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe podcast audio files using Whisper")
    parser.add_argument("audio", help="Path to the audio file or URL")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use")
    parser.add_argument("--output", default=None,
                        help="Output file path or directory (defaults to current directory)")
    parser.add_argument("--format", action="store_true",
                        help="Apply rich text formatting (bold, italics) to key terms")
    parser.add_argument("--links", action="store_true",
                        help="Create Obsidian-style links for key terms and concepts")
    parser.add_argument("--refs", action="store_true",
                        help="Add external references as footnotes")
    parser.add_argument("--vault", default=None,
                        help="Path to Obsidian vault for linking to existing files")

    args = parser.parse_args()

    # Determine output directory
    if args.output and os.path.isdir(args.output):
        output_dir = args.output
    elif args.output:
        output_dir = os.path.dirname(args.output) or os.getcwd()
    else:
        output_dir = os.getcwd()

    # Check if input is URL or local file
    if is_url(args.audio):
        audio_path, source_url = download_file(args.audio, output_dir)
    else:
        audio_path = args.audio
        source_url = None

    # Extract metadata from the audio file
    metadata = extract_metadata(audio_path)

    # Transcribe the audio
    result = transcribe_audio(audio_path, args.model)

    # Save as markdown with metadata and paragraphs
    save_transcript_markdown(
        result,
        audio_path,
        metadata,
        args.output,
        source_url,
        apply_formatting=args.format,
        create_links=args.links,
        create_refs=args.refs,
        vault_path=args.vault
    )

if __name__ == "__main__":
    main()
