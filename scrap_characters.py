import json
from bs4 import BeautifulSoup

def extract_characters(html_file: str) -> list[str]:
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    characters = []

    # Character entries are <li> items inside the main content
    content = soup.find("div", {"id": "mw-content-text"})
    if not content:
        return characters

    for li in content.find_all("li"):
        first_link = li.find("a")
        if first_link and first_link.get("href", "").startswith("/index.php/"):
            name = first_link.get("title") or first_link.get_text(strip=True)
            # Skip navigation/meta links (e.g. house names, event names used as context)
            # Only keep the first link per <li> as the character name
            if name:
                characters.append(name)

    return characters


if __name__ == "__main__":
    html_file = "data/characters.html"
    output_file = "data/characters.json"

    names = extract_characters(html_file)
    for name in names:
        print(name)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)

    print(f"\nTotal: {len(names)} characters found.")
    print(f"Saved to {output_file}")
