from playwright.sync_api import sync_playwright

BASE = "https://swice-app.epfl.ch"
LOGIN_URL = f"{BASE}/admin/login/"

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # visible
        context = browser.new_context()
        page = context.new_page()

        page.goto(LOGIN_URL, wait_until="domcontentloaded")
        print("Log in manually in the opened browser window.")
        print("After you see the Django admin (e.g. /admin/), come back here and press Enter.")
        input()

        # save session cookies + local storage
        context.storage_state(path="admin_state.json")
        print("Saved admin_state.json")

        browser.close()

if __name__ == "__main__":
    main()
