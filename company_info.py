import re
import json
import pandas as pd
from openai import OpenAI
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of common and temporary email domains to ignore
IGNORE_DOMAINS = {
    # Common email providers
    'gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com', 'aol.com', 
    'icloud.com', 'protonmail.com',
    # Temporary email services
    'tempmail.com', 'guerrillamail.com', 'mailinator.com', '10minutemail.com', 
    'yopmail.com', 'disposabled.com', 'throwawaymail.com'
}

def load_config(config_file="config.json"):
    """Load OpenAI API key and model from config.json."""
    try:
        with open(config_file, "r") as f:
            config = json.load(f)
            api_key = config.get("api_key")
            model = config.get("model")
            if not api_key or not model:
                raise ValueError("Missing 'api_key' or 'model' in config.json")
            return api_key, model
    except FileNotFoundError:
        logger.error(f"{config_file} not found.")
        raise
    except json.JSONDecodeError:
        logger.error(f"{config_file} is not a valid JSON file.")
        raise
    except Exception as e:
        logger.error(f"Error loading config from {config_file}: {e}")
        raise

def read_csv_with_encoding(file_path):
    """Read a CSV file with encoding detection or fallback."""
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'windows-1252']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully read {file_path} with encoding {encoding}")
            return df
        except UnicodeDecodeError as e:
            logger.warning(f"Failed to read {file_path} with encoding {encoding}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error reading {file_path} with encoding {encoding}: {e}")
            raise
    logger.error(f"Could not read {file_path} with any of the encodings: {encodings}")
    raise ValueError(f"Unable to read {file_path} with any supported encoding.")

def extract_domain(email):
    """Extract domain from an email address."""
    try:
        match = re.search(r'@([\w.-]+)', email)
        domain = match.group(1) if match else None
        if domain and not re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain):
            logger.warning(f"Invalid domain format: {domain} from email {email}")
            return None
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from {email}: {e}")
        return None

def get_company_info(domain, client, model):
    """Query OpenAI API to get company name, country, and category for a domain."""
    if not domain:
        return None, "Invalid domain"

    prompt = f"""
    Given the domain '{domain}', provide information about the associated company in JSON format:
    {{
        "company_name": "Name of the company (e.g., Google LLC for google.com)",
        "country": "Country where the company is headquartered (e.g., United States)",
        "category": "Industry or category (e.g., Technology, Finance, Retail)"
    }}
    If the domain is not associated with a known company or organization, return:
    {{
        "company_name": null,
        "country": null,
        "category": null
    }}
    Ensure the response is valid JSON.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a business intelligence assistant that provides accurate company information based on domain names."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"Raw API response for {domain}: {content}")
        
        # Clean up response if it contains markdown code fences
        if content.startswith("```json"):
            content = content[7:-3].strip()
        
        # Parse JSON response
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or not all(key in parsed for key in ["company_name", "country", "category"]):
            logger.warning(f"Invalid API response format for {domain}: {content}")
            return None, "Invalid response format"
        
        return parsed, "Success"
    except json.JSONDecodeError:
        logger.error(f"Failed to parse API response for {domain}: {content}")
        return None, "JSON parse error"
    except Exception as e:
        logger.error(f"Error querying OpenAI for {domain}: {e}")
        return None, str(e)

def process_emails(email_data, client, model):
    """Process a list of emails to extract unique domains, count emails, fetch company info, and aggregate timezones."""
    # Dictionary to store domain counts, timezones, and company info
    domain_data = {}
    
    # Process each email to count domains and collect timezones
    for _, row in email_data.iterrows():
        email = row['email']
        timezone = row['timezone_name']
        domain = extract_domain(email)
        
        if domain and domain.lower() not in IGNORE_DOMAINS:
            if domain not in domain_data:
                domain_data[domain] = {"email_count": 0, "timezones": []}
            domain_data[domain]["email_count"] += 1
            domain_data[domain]["timezones"].append(timezone)
        else:
            if domain:
                logger.info(f"Ignored domain: {domain}")
    
    # Fetch company info and determine the most frequent timezone for each domain
    for domain in domain_data:
        # Get company info
        company_info, status = get_company_info(domain, client, model)
        
        # Determine the most frequent timezone
        timezones = domain_data[domain]["timezones"]
        if timezones:
            # Use Counter to find the most common timezone
            most_common = Counter(timezones).most_common(1)
            timezone = most_common[0][0]  # Take the first most frequent timezone
        else:
            timezone = None
            logger.warning(f"No timezone data for domain {domain}")
        
        domain_data[domain].update({
            "company_name": company_info.get("company_name") if company_info else None,
            "country": company_info.get("country") if company_info else None,
            "category": company_info.get("category") if company_info else None,
            "status": status,
            "timezone": timezone
        })
    
    # Convert to list of dictionaries for output
    results = [
        {
            "domain": domain,
            "email_count": data["email_count"],
            "company_name": data.get("company_name"),
            "country": data.get("country"),
            "category": data.get("category"),
            "status": data.get("status"),
            "timezone": data.get("timezone")
        }
        for domain, data in domain_data.items()
    ]
    
    return results

def main():
    # Load API key and model from config.json
    try:
        api_key, model = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Read email list from CSV in data folder
    csv_file = "data/email-all.csv"
    try:
        df = read_csv_with_encoding(csv_file)
        # Validate required columns
        required_columns = ['email', 'timezone_name']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            logger.error(f"Missing required columns in {csv_file}: {missing}")
            return
        emails = df[required_columns + [col for col in df.columns if col not in required_columns]]
        logger.info(f"Loaded {len(emails)} emails from {csv_file}")
    except ValueError as e:
        logger.error(f"Failed to read {csv_file}: {e}")
        return
    except Exception as e:
        logger.error(f"Error reading {csv_file}: {e}")
        return

    # Process emails and get results
    results = process_emails(emails, client, model)
    
    # Convert results to DataFrame for CSV output
    output_df = pd.DataFrame(results)
    
    # Print results to console
    if not output_df.empty:
        logger.info("Processing results:")
        print(output_df.to_string(index=False))
    else:
        logger.info("No valid domains found after ignoring common and temporary email domains.")
        print("No valid domains found.")
    
    # Save results to CSV
    output_file = "data/domain_company_info.csv"
    try:
        output_df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Results saved to {output_file}")
        print(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")
        print(f"Error saving results to {output_file}: {e}")

if __name__ == "__main__":
    main()