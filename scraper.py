import threading
import time
from bs4 import BeautifulSoup # pip install beautifulsoup4
from models import Procurement
import random
from urllib.request import urlopen
from selenium import webdriver # pip install selenium
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import yaml # pip install pyyaml
import time
import re
from datetime import datetime

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

def write(status, text, link, customer, deadline):
    with app.app_context():
        existing = db.session.query(Procurement).filter_by(link=link).first()
        if existing:
            db.session.query(Procurement).filter(Procurement.link == link).update({
                Procurement.status: status, 
                Procurement.text: text,
                Procurement.customer: customer,
                Procurement.deadline: deadline
            })
        else:
            procurement = Procurement(status, text, link, customer, deadline)
            db.session.add(procurement)
        db.session.commit()

def elements_loaded(driver, num_elements=20):
    rows = driver.find_elements(By.CSS_SELECTOR, "#ResultsRepeater tr")
    return len(rows) >= num_elements

def scrapeEIS():
    print("Scraping...")
    service = Service(config["scraper"]["geckodriver_path"])
    browser = webdriver.Firefox(service=service)

    url = "https://www.eis.gov.lv/EKEIS/Supplier"

    mainUrl = "https://www.eis.gov.lv"
    browser.get(url)

    try:
        WebDriverWait(browser, 15).until(
            EC.presence_of_element_located((By.ID, "ResultsRepeater"))
        )
        WebDriverWait(browser, 15).until(
            lambda browser: elements_loaded(browser, 20)
        )
        while True:
            soup = BeautifulSoup(browser.page_source, "html.parser")
            for id, i in enumerate(soup.find('div', id='ResultsRepeater').find_all('tr')):
                if id == 0:
                    continue
                td = i.find_all('td')
                status = td[0].text.strip()
                text = td[2].text.strip()
                link = td[2].find('a')['href']
                customer = td[3].text.strip()
                endDate = td[4].text.strip()
                match = re.search(r"Iesniegšanas termiņš:\s*(\d{2}\.\d{2}\.\d{4})", endDate)
                if match:
                    endDate = match.group(1)
                    endDate = datetime.strptime(endDate, "%d.%m.%Y").date()
                    write(status, text, mainUrl + link, customer, endDate)
                else:
                    write(status, text, mainUrl + link, customer, None)
            button = browser.find_element(By.ID, "Resultsfooter-next-page")
            if button.value_of_css_property("display") == "none":
                break
            button.click()
            WebDriverWait(browser, 20).until(
                lambda driver: driver.find_element(By.ID, "loader").value_of_css_property("display") == "none"
            )
            time.sleep(random.uniform(4, 7))
    except Exception as e:
        print("An error occurred while waiting for the page to load:", e)
    browser.quit()

def task():
    timestamp_ms = int(time.time() * 1000)
    if (timestamp_ms > config["scraper"]["last_scrape"] + 86400):
        scrapeEIS()

        config["scraper"]["last_scrape"] = timestamp_ms
        with open("config.yml", "w") as file:
            yaml.dump(config, file)

class ProcurementScraper():

    def __init__(self, database, application):
        global db, app
        db = database
        app = application
        if (config["scraper"]["use"]):
            scheduler_thread = threading.Thread(target=self.schedule_task)
            scheduler_thread.daemon = True
            scheduler_thread.start()

    def schedule_task(self):
        while True:
            task_thread = threading.Thread(target=task)
            task_thread.start()
            time.sleep(86400)