"""Save current Bulacan PDRRMO gauge readings to build a local history."""

import csv
import re
from datetime import datetime, timezone
from html import unescape

import truststore

truststore.inject_into_ssl()
import requests  # noqa: E402

from common import GAUGE_LOG  # noqa: E402

PDRRMO_URL = "https://pdrrmo.bulacan.gov.ph/"
USER_AGENT = "AquaAlert-Bulacan-research"
FIELDS = ["fetched_utc", "station", "level_m", "alert_m", "alarm_m", "critical_m", "reported_date"]
DAM_FIELDS = ["fetched_utc", "dam", "level_m", "normal_m", "spilling_m", "reported_date"]
ROW_PATTERN = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
NAME_PATTERN = re.compile(r'<th\b[^>]*scope="row"[^>]*>(.*?)</th>', re.S | re.I)
CELL_PATTERN = re.compile(r"<td\b[^>]*>(.*?)</td>", re.S | re.I)
METRES_PATTERN = re.compile(r"^(-?\d+(?:\.\d+)?)\s*meters?$", re.I)
DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")


def clean(html: str) -> str:
    text = re.sub(r'<span class="visually-hidden">.*?</span>', " ", html, flags=re.S | re.I)
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text)).split())


def parse_stations(html: str) -> list[dict]:
    """Read only rows with four metre values and a date, which excludes dam rows."""
    stations = []
    for row in ROW_PATTERN.findall(html):
        name = NAME_PATTERN.search(row)
        cells = [clean(cell) for cell in CELL_PATTERN.findall(row)]
        if not name or len(cells) != 5 or not DATE_PATTERN.match(cells[4]):
            continue
        values = [METRES_PATTERN.match(cell) for cell in cells[:4]]
        if not all(values):
            continue
        level, alert, alarm, critical = (float(match.group(1)) for match in values)
        stations.append({
            "station": clean(name.group(1)), "level_m": level, "alert_m": alert,
            "alarm_m": alarm, "critical_m": critical, "reported_date": cells[4],
        })
    return stations


def parse_dams(html: str) -> list[dict]:
    """Read dam rows, whose normal and spilling levels may each be blank."""
    dams = []
    for row in ROW_PATTERN.findall(html):
        name = NAME_PATTERN.search(row)
        cells = [clean(cell) for cell in CELL_PATTERN.findall(row)]
        if not name or len(cells) != 4 or not DATE_PATTERN.match(cells[3]):
            continue
        current = METRES_PATTERN.match(cells[0])
        if not current:
            continue
        normal, spilling = (METRES_PATTERN.match(cell) for cell in cells[1:3])
        dams.append({
            "dam": clean(name.group(1)), "level_m": float(current.group(1)),
            "normal_m": float(normal.group(1)) if normal else "",
            "spilling_m": float(spilling.group(1)) if spilling else "", "reported_date": cells[3],
        })
    return dams


def append_rows(path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    response = requests.get(PDRRMO_URL, timeout=60, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    stations = parse_stations(response.text)
    dams = parse_dams(response.text)
    if not stations:
        raise RuntimeError("No gauge rows found. The PDRRMO page layout may have changed.")

    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    append_rows(GAUGE_LOG, FIELDS, [{"fetched_utc": fetched, **station} for station in stations])
    append_rows(GAUGE_LOG.with_name("bulacan_dams.csv"), DAM_FIELDS,
                [{"fetched_utc": fetched, **dam} for dam in dams])
    print(f"Saved {len(stations)} gauge and {len(dams)} dam readings to {GAUGE_LOG.parent}")
    for dam in dams:
        print(f"  {dam['dam']}: {dam['level_m']} m (reported {dam['reported_date']})")
    for station in stations:
        print(f"  {station['station']}: {station['level_m']} m "
              f"(alert {station['alert_m']} m, reported {station['reported_date']})")


if __name__ == "__main__":
    main()
