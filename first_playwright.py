from playwright.sync_api import sync_playwright

print("Starting Playwright...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.cricbuzz.com/")
    page.wait_for_timeout(8000)  # wait for results
    page_text = page.content()  # get the page content
    #click on the "Live Scores" link
    page.click("text=Live Scores")
    page.wait_for_timeout(5000)  # wait for the live scores to load
#take a screenshot of the live scores page
    page.screenshot(path="live_scores.png")
    print("Page content preview:", page_text[:1000])