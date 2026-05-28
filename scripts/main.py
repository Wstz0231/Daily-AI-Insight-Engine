from pathlib import Path
from util import get_openai_client
from clean_data import run_pipeline as clean_data
from schema import run as structure_data
from report import run as generate_report
from visualization import run as generate_visualizations


def main():
    client = get_openai_client()

    raw_path = Path("data/raw_data.json")
    cleaned_path = Path("data/cleaned_data.json")
    structured_path = Path("data/structured_data.json")

    print("Cleaning...")
    clean_data(raw_path, cleaned_path, client)

    print("Extracting...")
    structure_data(cleaned_path, structured_path)

    print("Generating Report...")
    generate_report()

    print("Visualizing...")
    generate_visualizations()

    print("Done.")


if __name__ == "__main__":
    main()