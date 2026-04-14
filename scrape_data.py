
import os
import time
import pandas as pd

NFL_YEARS = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]

# === USE PANDAS TO SCRAPE DATA ===
def scrape_data(NFL_YEAR):
    # URL to scrape data from
    URL_DATA = f"https://www.pro-football-reference.com/years/{NFL_YEAR}/"                                             

    # Read the data from the URL
    AFC = pd.read_html(URL_DATA, header=0)[0]
    NFC = pd.read_html(URL_DATA, header=0)[1]
    
    # Clean the data
    AFC = AFC.drop([0, 5, 10, 15])
    NFC = NFC.drop([0, 5, 10, 15])
    
    # combine the dataframes
    NFL = pd.concat([AFC, NFC], ignore_index=True)
    NFL = NFL.dropna()
    NFL = NFL.reset_index(drop=True)
    
    # if Team name has '*' or '+' remove it
    NFL['Tm'] = NFL['Tm'].str.replace('*', '', regex=False)
    NFL['Tm'] = NFL['Tm'].str.replace('+', '', regex=False)
    
    # Save the data to a CSV file to the final save directory
    NFL.to_csv(f"{NFL_YEAR}.csv", index=False)
    
# === MAIN FUNCTION ===
if __name__ == "__main__":
    
    # Call the scrape_data function
    for NFL_YEAR in NFL_YEARS:
        time.sleep(1)  # Sleep for 1 second to avoid bot detection
        scrape_data(NFL_YEAR)
        print(f"Data for {NFL_YEAR} scraped successfully.")
        
    print("Data scraping complete.")