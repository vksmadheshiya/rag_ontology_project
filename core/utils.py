import re
import requests
import hashlib
from bs4 import BeautifulSoup # Can be helpful for more complex cleaning

def get_book_identifier(url_or_path):
    """Creates a simple unique identifier for a book based on URL or path."""
    # Use SHA256 hash for a consistent identifier
    return hashlib.sha256(url_or_path.encode('utf-8')).hexdigest()[:16] # Use first 16 chars

def load_text_from_url(url):
    """Downloads text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        # Simple text extraction, might need improvement for complex HTML
        # Using response.text assumes text format; adjust if needed
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading URL {url}: {e}")
        return None

def load_text_from_upload(uploaded_file):
    """Reads text from a Streamlit UploadedFile object."""
    if uploaded_file is not None:
        try:
            # Assume UTF-8 encoding, might need adjustment
            return uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            print(f"Error reading uploaded file: {e}")
            return None
    return None

def clean_text(text):
    """Basic text cleaning: removes Gutenberg headers/footers and extra whitespace."""
    if not text:
        return ""

    # Remove Gutenberg header
    text = re.sub(r'\*\*\* START OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*', '', text, flags=re.IGNORECASE)
    # Remove Gutenberg footer
    text = re.sub(r'\*\*\* END OF THIS PROJECT GUTENBERG EBOOK .* \*\*\*', '', text, flags=re.IGNORECASE)
    # Remove Gutenberg licence section (more robust patterns might be needed)
    text = re.sub(r'START OF THE PROJECT GUTENBERG EBOOK.*?Produced by', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'End of the Project Gutenberg EBook.*', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove excessive newlines and whitespace
    text = re.sub(r'\n{3,}', '\n\n', text) # Replace 3+ newlines with 2
    text = re.sub(r'[ \t]{2,}', ' ', text) # Replace 2+ spaces/tabs with 1
    text = text.strip()
    return text

# Example of a more advanced cleaning using BeautifulSoup if source was HTML
def clean_html_text(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    # Get text
    text = soup.get_text()
    # Break into lines and remove leading/trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # Apply generic cleaning
    return clean_text(text)