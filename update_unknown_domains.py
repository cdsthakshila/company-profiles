import csv
import json
import logging
from typing import Dict, Set
from openai import OpenAI
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('update_unknown_domains.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_company_info(domain: str, timezone: str, model: str, client: OpenAI) -> Dict:
    """Get company information using ChatGPT API."""
    logger.info(f"Processing domain: {domain}")
    prompt = f"""
    Given the domain {domain} and timezone {timezone}, please provide the following information in EXACTLY this JSON format:
    {{
        "category": "company industry/category or Unknown",
        "country": "country of operation or Unknown"
    }}

    Rules:
    1. Return ONLY the JSON object, no other text
    2. Use "Unknown" for any information you cannot find
    3. Category should be a specific industry or business type
    4. Country should be the full country name
    5. Focus on finding the correct category and country, even if previously marked as Unknown
    """
    
    try:
        logger.debug(f"Sending request to OpenAI for domain: {domain}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides company information. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        logger.debug(f"Received response for {domain}: {content}")
        
        # Try to find JSON in the response if it's not directly JSON
        if not content.startswith('{'):
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                content = content[json_start:json_end]
                logger.debug(f"Extracted JSON from response: {content}")
        
        try:
            result = json.loads(content)
            logger.info(f"Successfully processed domain: {domain}")
            return {
                'category': result.get('category', 'Unknown'),
                'country': result.get('country', 'Unknown')
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {domain}: {str(e)}")
            logger.error(f"Raw response: {content}")
            return {
                'category': 'Unknown',
                'country': 'Unknown'
            }
    except Exception as e:
        logger.error(f"Error getting info for {domain}: {str(e)}")
        return {
            'category': 'Unknown',
            'country': 'Unknown'
        }

def read_csv_with_encoding(file_path: str) -> list:
    """Read CSV file with UTF-8 encoding and handle encoding errors."""
    try:
        logger.info(f"Reading CSV file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            logger.info(f"Successfully read {len(data)} records from CSV")
            return data
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 encoding failed for {file_path}, trying with UTF-8-sig")
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            logger.info(f"Successfully read {len(data)} records from CSV using UTF-8-sig")
            return data
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        raise

def write_csv_with_encoding(file_path: str, data: list, fieldnames: list):
    """Write CSV file with UTF-8 encoding."""
    try:
        logger.info(f"Writing to CSV file: {file_path}")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        logger.info(f"Successfully wrote {len(data)} records to CSV")
    except Exception as e:
        logger.error(f"Error writing CSV file: {str(e)}")
        raise

def main():
    logger.info("Starting update of unknown domains")
    
    # Load configuration
    try:
        logger.info("Loading configuration from config.json")
        with open('config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise
    
    # Initialize OpenAI client
    client = OpenAI(api_key=config.get('api_key'))
    if not client.api_key:
        logger.error("OpenAI API key not found in config file")
        raise ValueError("OpenAI API key not found in config file")
    
    model = config.get('model', 'gpt-3.5-turbo')
    if not model:
        logger.error("GPT model not found in config file")
        raise ValueError("GPT model not found in config file")
    logger.info(f"Using OpenAI model: {model}")

    # Read input CSV
    input_file = 'data/enriched-email-list-main.csv'
    try:
        email_data = read_csv_with_encoding(input_file)
    except Exception as e:
        logger.error(f"Error reading input file: {str(e)}")
        raise

    # Find domains with unknown category or country
    domains_to_update = set()
    for row in email_data:
        if row['category'] == 'Unknown' or row['country'] == 'Unknown':
            domains_to_update.add(row['domain'])
    
    logger.info(f"Found {len(domains_to_update)} domains with unknown information")

    # Get updated information for domains
    domain_updates = {}
    for i, domain in enumerate(domains_to_update, 1):
        logger.info(f"Processing domain {i}/{len(domains_to_update)}: {domain}")
        # Get timezone from the first email with this domain
        timezone = next((row['timezone_name'] for row in email_data if row['domain'] == domain), 'UTC')
        domain_updates[domain] = get_company_info(domain, timezone, model, client)

    # Update the data with new information
    updated_count = 0
    for row in email_data:
        domain = row['domain']
        if domain in domain_updates:
            updates = domain_updates[domain]
            if row['category'] == 'Unknown':
                row['category'] = updates['category']
                updated_count += 1
            if row['country'] == 'Unknown':
                row['country'] = updates['country']
                updated_count += 1

    logger.info(f"Updated {updated_count} fields with new information")

    # Write updated data back to CSV
    try:
        fieldnames = ['email', 'name', 'last_active', 'timezone_name', 'domain', 'website', 'category', 'country']
        write_csv_with_encoding(input_file, email_data, fieldnames)
        logger.info("Successfully completed updating unknown domains")
    except Exception as e:
        logger.error(f"Error writing updated data: {str(e)}")
        raise

if __name__ == "__main__":
    main() 