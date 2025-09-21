"""
Contains various functions used throughout this project.

Attributes:
    cache_dir (TYPE): Description
    log_dir (TYPE): Description
    logger (TYPE): Description
    output_dir (TYPE): Description
    root_dir (TYPE): Description
"""

import json
import os
import pickle
import smtplib
import subprocess
from email.message import EmailMessage
from typing import Dict

import pycountry

from custom_logger import CustomLogger

root_dir = os.path.dirname(__file__)
cache_dir = os.path.join(root_dir, "_cache")
log_dir = os.path.join(root_dir, "_logs")
output_dir = os.path.join(root_dir, "_output")

logger = CustomLogger(__name__)


def get_secrets(entry_name: str, secret_file_name: str = "secret") -> Dict[str, str]:
    """Open the secrets file and return the requested entry."""
    with open(os.path.join(root_dir, secret_file_name)) as f:
        return json.load(f)[entry_name]


def get_configs(
    entry_name: str,
    config_file_name: str = "config",
    config_default_file_name: str = "default.config",
):
    """
    Open the config file and return the requested entry.
    If no config file is found, open default.config.
    """
    if not check_config():
        return None
    try:
        with open(os.path.join(root_dir, config_file_name)) as f:
            content = json.load(f)
    except FileNotFoundError:
        with open(os.path.join(root_dir, config_default_file_name)) as f:
            content = json.load(f)
    return content[entry_name]


def check_config(
    config_file_name: str = "config", config_default_file_name: str = "default.config"
):
    """
    Check if config file has at least as many rows as default.config.
    """
    try:
        with open(os.path.join(root_dir, config_file_name)) as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error("Config file %s not found.", config_file_name)
        return False
    except json.decoder.JSONDecodeError:
        logger.error("Config file badly formatted. Update based on default.config.")
        return False

    try:
        with open(os.path.join(root_dir, config_default_file_name)) as f:
            default = json.load(f)
    except FileNotFoundError:
        logger.error("Default config file %s not found.", config_default_file_name)
        return False
    except json.decoder.JSONDecodeError:
        logger.error("Default config badly formatted.")
        return False

    if len(config) < len(default):
        logger.error(
            "Config file has %d variables, fewer than %d in default.config. Please update.",
            len(config),
            len(default),
        )
        return False
    return True


def search_dict(dictionary, search_for, nested=False):
    """Search dictionary values for a string. Traverse nested if specified."""
    for k in dictionary:
        if nested:
            for v in dictionary[k]:
                if search_for in v or v in search_for:
                    return k
        else:
            if search_for in dictionary[k] or dictionary[k] in search_for:
                return k
    return None


def save_to_p(file, data, description_data="data"):
    """Save data to a pickle file."""
    path = os.path.join(root_dir, "trust", file)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Saved %s to pickle file %s.", description_data, file)


def load_from_p(file, description_data="data"):
    """Load data from a pickle file."""
    path = os.path.join(root_dir, "trust", file)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %s from pickle file %s.", description_data, file)
    return data


def correct_country(country):
    """Correct common country name variations for pycountry compatibility."""
    corrections = {
        "Russia": "Russian Federation",
        "Syria": "Syrian Arab Republic",
        "South Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Republic of",
        "Korea": "Korea, Republic of",
        "Iran": "Iran, Islamic Republic of",
        "Vietnam": "Viet Nam",
        "Venezuela": "Venezuela, Bolivarian Republic of",
        "Bolivia": "Bolivia, Plurinational State of",
        "Moldova": "Moldova, Republic of",
        "Laos": "Lao People's Democratic Republic",
        "Brunei": "Brunei Darussalam",
        "Czech Republic": "Czechia",
        "Ivory Coast": "Côte d'Ivoire",
        "Cape Verde": "Cabo Verde",
        "Swaziland": "Eswatini",
        "Macau": "Macao",
        "Taiwan": "Taiwan, Province of China",
        "Tanzania": "Tanzania, United Republic of",
        "UK": "United Kingdom",
        "Palestine": "Palestine, State of",
        "Micronesia": "Micronesia, Federated States of",
        "Bahamas": "Bahamas, The",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Turkiye": "Turkey",
        "Türkiye": "Turkey",
        "Congo (Democratic Republic)": "Congo, The Democratic Republic of the",
        "Congo (Congo-Brazzaville)": "Congo",
        "Burma": "Myanmar",
        "East Timor": "Timor-Leste",
        "Saint Kitts": "Saint Kitts and Nevis",
        "Saint Vincent": "Saint Vincent and the Grenadines",
        "Saint Lucia": "Saint Lucia",
        "Antigua": "Antigua and Barbuda",
        "Trinidad": "Trinidad and Tobago",
        "Slovak Republic": "Slovakia",
        "Vatican": "Holy See",
    }
    return corrections.get(country, country)


def iso3_to_country_name(iso3):
    """Convert ISO-3 code to country name."""
    try:
        country = pycountry.countries.get(alpha_3=iso3.upper())
        return country.name if country else None
    except KeyError:
        return None


def get_iso2_country_code(country_name):
    """Get ISO-2 code for a country."""
    if country_name == "Kosovo":
        return "XK"
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_2 if country else "Country not found"
    except KeyError:
        return "Country not found"


def get_iso3_country_code(country_name):
    """Get ISO-3 code for a country."""
    if country_name == "Kosovo":
        return "XKX"
    try:
        country = pycountry.countries.get(name=country_name)
        return country.alpha_3 if country else "Country not found"
    except KeyError:
        return "Country not found"


def git_pull():
    """Pull changes from git repository."""
    try:
        logger.info("Attempting to pull latest changes from git repository...")
        result = subprocess.run(
            ["git", "pull"], capture_output=True, text=True, check=True
        )
        logger.info(f"Git pull successful:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Git pull failed with error:\n{e.stderr}")


def send_email(subject, content, sender, recipients):
    """Send email with a given subject and content."""
    msg = EmailMessage()
    msg.set_content(content)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    try:
        with smtplib.SMTP_SSL(get_secrets("email_smtp"), 465) as smtp:
            smtp.login(get_secrets("email_account"), get_secrets("email_password"))
            smtp.send_message(msg)
            logger.info(f"Sent email to: {recipients}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
