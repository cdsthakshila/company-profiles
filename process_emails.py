import csv
import json
import os
import re
import logging
from typing import List, Dict, Set
from openai import OpenAI
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('email_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Common email domains to skip
COMMON_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
    'icloud.com', 'protonmail.com', 'mail.com', 'yandex.com', 'zoho.com',
    'gmx.com', 'live.com', 'msn.com', 'me.com', 'mac.com'
}

def extract_domain(email: str) -> str:
    """Extract domain from email address."""
    return email.split('@')[-1].lower()

def is_valid_domain(domain: str) -> bool:
    """Check if domain is not a common email provider."""
    return domain not in COMMON_DOMAINS and not any(
        domain.endswith(f'.{temp}') for temp in ['temp.com', 'tempmail.com', 'mailinator.com']
    )

def get_company_info(domain: str, timezone: str, model: str, client: OpenAI) -> Dict:
    """Get company information using ChatGPT API."""
    logger.info(f"Processing domain: {domain}")
    prompt = f"""
    Given the domain {domain} and timezone {timezone}, please provide the following information in EXACTLY this JSON format:
    {{
        "website": "company website URL or Unknown",
        "category": "company industry/category or Unknown",
        "country": "country of operation or Unknown"
    }}

    Rules:
    1. Return ONLY the JSON object, no other text
    2. Use "Unknown" for any information you cannot find
    3. Website should be a full URL starting with http:// or https://
    4. Category should be a specific industry or business type
    5. Country should be the full country name
    """
    
    try:
        logger.debug(f"Sending request to OpenAI for domain: {domain}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides company information. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent responses
        )
        
        # Extract the content and clean it
        content = response.choices[0].message.content.strip()
        logger.debug(f"Received response for {domain}: {content}")
        
        # Try to find JSON in the response if it's not directly JSON
        if not content.startswith('{'):
            # Look for JSON-like content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                content = content[json_start:json_end]
                logger.debug(f"Extracted JSON from response: {content}")
        
        # Parse the response
        try:
            result = json.loads(content)
            logger.info(f"Successfully processed domain: {domain}")
            return {
                'website': result.get('website', 'Unknown'),
                'category': result.get('category', 'Unknown'),
                'country': result.get('country', 'Unknown')
            }
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error for {domain}: {str(e)}")
            logger.error(f"Raw response: {content}")
            return {
                'website': 'Unknown',
                'category': 'Unknown',
                'country': 'Unknown'
            }
    except Exception as e:
        logger.error(f"Error getting info for {domain}: {str(e)}")
        return {
            'website': 'Unknown',
            'category': 'Unknown',
            'country': 'Unknown'
        }

def read_csv_with_encoding(file_path: str) -> List[Dict]:
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

def main():
    logger.info("Starting email processing script")
    
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
    input_file = 'data/email-list-main.csv'
    output_file = 'data/enriched-email-list-main.csv'
    
    # Process emails and collect unique domains
    unique_domains = set()
    try:
        email_data = read_csv_with_encoding(input_file)
        
        for row in email_data:
            email = row['email']
            domain = extract_domain(email)
            if is_valid_domain(domain):
                unique_domains.add(domain)
        logger.info(f"Found {len(unique_domains)} unique domains to process")
    except Exception as e:
        logger.error(f"Error processing input file: {str(e)}")
        raise

    # Get company information for each unique domain
    domain_info = {}
    for i, domain in enumerate(unique_domains, 1):
        logger.info(f"Processing domain {i}/{len(unique_domains)}: {domain}")
        # Get timezone from the first email with this domain
        timezone = next((row['timezone_name'] for row in email_data if extract_domain(row['email']) == domain), 'UTC')
        domain_info[domain] = get_company_info(domain, timezone, model, client)

    # Write enriched data to output CSV
    try:
        logger.info(f"Writing enriched data to {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['email', 'name', 'last_active', 'timezone_name', 'domain', 'website', 'category', 'country']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in email_data:
                domain = extract_domain(row['email'])
                info = domain_info.get(domain, {
                    'website': 'Unknown',
                    'category': 'Unknown',
                    'country': 'Unknown'
                })
                
                enriched_row = {
                    **row,
                    'domain': domain,
                    'website': info['website'],
                    'category': info['category'],
                    'country': info['country']
                }
                writer.writerow(enriched_row)
        logger.info("Successfully completed processing")
    except Exception as e:
        logger.error(f"Error writing output file: {str(e)}")
        raise

if __name__ == "__main__":
    main() 