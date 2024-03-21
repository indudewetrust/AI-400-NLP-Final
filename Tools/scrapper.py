import requests
from bs4 import BeautifulSoup

def scrape_paragraphs(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    paragraphs = []
    headline = None

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = [paragraph.get_text(strip=True) for paragraph in soup.find_all('p')]

        # Find the headline element (adjust the selector based on the webpage structure)
        headline_element = soup.find('h1')
        if headline_element:
            headline = headline_element.get_text(strip=True)

    except requests.RequestException as e:
        print(f"Error during HTTP request: {e}")
        return [], None  # Return empty lists if an error occurs
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], None  # Return empty lists if an error occurs

    return paragraphs, headline



if __name__ == "__main__":
    # Example usage
    url_to_scrape = ''
    result_paragraphs = scrape_paragraphs(url_to_scrape)

    for paragraph in result_paragraphs:
        print(paragraph)
