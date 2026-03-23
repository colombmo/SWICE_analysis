import csv
import re
from playwright.sync_api import sync_playwright

BASE = "https://swice-app.epfl.ch"  # no trailing slash
LOGIN_URL = f"{BASE}/admin/login/"
CHANGE_URL_TMPL = f"{BASE}/admin/experiments/lifestyle/{{id}}/change/"

USERNAME = "admin"
PASSWORD = "27kA8Qy@Tkx6"

IDS = [
    1,2,3,4,5,6,7,8,9,10,
    11,12,13,14,15,16,17,18,19,20,
    21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,
    41,42,43,44,45,46,47,48,49,50,
    51,52,53,54,55,56,57,58,59,60,
    61,62,63,64,65,66,67,68,69,70,
    71,72,73,74,75,76,77,78,79,80,
    81,82,83,84
]

# ---- helpers to read Django admin fields ----
def read_input_value(page, field_name: str):
    # Django admin inputs typically: <input name="field" ...>
    loc = page.locator(f'input[name="{field_name}"]')
    if loc.count() > 0:
        return loc.first.input_value()

    # selects: <select name="field">...</select>
    loc = page.locator(f'select[name="{field_name}"]')
    if loc.count() > 0:
        # returns the visible text of selected option
        return loc.first.locator("option:checked").inner_text()

    # textarea
    loc = page.locator(f'textarea[name="{field_name}"]')
    if loc.count() > 0:
        return loc.first.input_value()

    return None

def dump_field_names(page):
    names = page.eval_on_selector_all(
        "input[name], select[name], textarea[name]",
        "els => els.map(e => e.getAttribute('name'))"
    )
    # Remove duplicates, keep order
    seen = set()
    ordered = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state="admin_state.json")
        page = context.new_page()

        rows = []
        for obj_id in IDS:
            page.goto(CHANGE_URL_TMPL.format(id=obj_id), wait_until="domcontentloaded")

            # if you got kicked out, you'll land back on login/SSO
            if "/admin/login" in page.url or "login" in page.url.lower():
                rows.append((obj_id, None, None, None, "NOT_LOGGED_IN"))
                continue

            # IMPORTANT: replace these field names with the real model field names
            participant = read_input_value(page, "participant")  # or "participant_id", etc.
            cluster1 = read_input_value(page, "lifestyle")
            cluster2 = read_input_value(page, "lifestyle_second")

            rows.append((obj_id, participant, cluster1, cluster2, ""))

            print(f"Read obj_id={obj_id}: participant={participant}, cluster1={cluster1}, cluster2={cluster2}")

        # Write CSV
        with open("lifestyle_triplets.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["admin_id", "participantid", "cluster1", "cluster2", "error"])
            w.writerows(rows)

        context.close()
        browser.close()
        print("Wrote lifestyle_triplets.csv")

if __name__ == "__main__":
    main()
