import re


class DataCleaner:
    def __init__(self, text):
        self.text = text

    def to_lower_case(self):
        """Convert text to lower case."""
        self.text = self.text.lower()
        return self

    def remove_punctuation(self):
        """Remove punctuation from the text."""
        import string
        self.text = self.text.translate(str.maketrans('', '',
                                        string.punctuation.replace('.', '').replace(',', '')))
        return self

    def remove_numbers(self):
        """Remove numbers and hashtags followed by numbers from the text."""
        # Remove # followed by numbers
        self.text = re.sub(r'#\d+', '', self.text)
        # Remove standalone numbers
        self.text = ''.join([i for i in self.text if not i.isdigit()])
        return self

    def remove_whitespace(self):
        """Remove carriage returns and convert multiple newlines to a single newline."""
        self.text = re.sub(r'\r', '', self.text)  # Remove carriage returns
        self.text = re.sub(r'\n+', '\n', self.text)  # Convert multiple newlines to a single newline
        return self

    def remove_urls(self):
        """Remove URLs from the text."""
        # Regex to match URLs
        self.text = re.sub(r'http[s]?://\S+', '', self.text)
        return self

    def get_cleaned_text(self):
        """Return the cleaned text."""
        return self.text

    def clean_data(self):
        """Orchestrate the cleaning process."""
        return (self.to_lower_case()
                .remove_urls()
                # .remove_punctuation()
                .remove_numbers()
                # .remove_whitespace()
                .get_cleaned_text())