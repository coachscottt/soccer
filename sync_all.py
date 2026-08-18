"""Sync the full stats-db backbone: big-5 leagues + MLS, 3 seasons.

Fully resumable - every response is disk-cached and synced fixtures are
skipped, so re-running after any interruption costs almost nothing.
Run: python sync_all.py
"""
from statsdb.apifootball import sync_league_season, LEAGUE_IDS

# (label, league_id, season). European season N = N/N+1.
QUEUE = [
    ("EPL 2023/24",        LEAGUE_IDS["EPL"],        2023),
    ("EPL 2024/25",        LEAGUE_IDS["EPL"],        2024),
    ("La Liga 2023/24",    LEAGUE_IDS["LaLiga"],     2023),
    ("La Liga 2024/25",    LEAGUE_IDS["LaLiga"],     2024),
    ("La Liga 2025/26",    LEAGUE_IDS["LaLiga"],     2025),
    ("Bundesliga 2023/24", LEAGUE_IDS["Bundesliga"], 2023),
    ("Bundesliga 2024/25", LEAGUE_IDS["Bundesliga"], 2024),
    ("Bundesliga 2025/26", LEAGUE_IDS["Bundesliga"], 2025),
    ("Serie A 2023/24",    LEAGUE_IDS["SerieA"],     2023),
    ("Serie A 2024/25",    LEAGUE_IDS["SerieA"],     2024),
    ("Serie A 2025/26",    LEAGUE_IDS["SerieA"],     2025),
    ("Ligue 1 2023/24",    LEAGUE_IDS["Ligue1"],     2023),
    ("Ligue 1 2024/25",    LEAGUE_IDS["Ligue1"],     2024),
    ("Ligue 1 2025/26",    LEAGUE_IDS["Ligue1"],     2025),
    ("MLS 2024",           LEAGUE_IDS["MLS"],        2024),
    ("MLS 2025",           LEAGUE_IDS["MLS"],        2025),
    # expansion: Championship + more top leagues (Jul 2026)
    ("Championship 2023/24", LEAGUE_IDS["Championship"], 2023),
    ("Championship 2024/25", LEAGUE_IDS["Championship"], 2024),
    ("Championship 2025/26", LEAGUE_IDS["Championship"], 2025),
    ("Eredivisie 2023/24",   LEAGUE_IDS["Eredivisie"],   2023),
    ("Eredivisie 2024/25",   LEAGUE_IDS["Eredivisie"],   2024),
    ("Eredivisie 2025/26",   LEAGUE_IDS["Eredivisie"],   2025),
    ("Primeira 2023/24",     LEAGUE_IDS["PrimeiraLiga"], 2023),
    ("Primeira 2024/25",     LEAGUE_IDS["PrimeiraLiga"], 2024),
    ("Primeira 2025/26",     LEAGUE_IDS["PrimeiraLiga"], 2025),
    ("JupilerPro 2023/24",   LEAGUE_IDS["JupilerPro"],   2023),
    ("JupilerPro 2024/25",   LEAGUE_IDS["JupilerPro"],   2024),
    ("JupilerPro 2025/26",   LEAGUE_IDS["JupilerPro"],   2025),
    ("SuperLig 2023/24",     LEAGUE_IDS["SuperLig"],     2023),
    ("SuperLig 2024/25",     LEAGUE_IDS["SuperLig"],     2024),
    ("SuperLig 2025/26",     LEAGUE_IDS["SuperLig"],     2025),
    ("ScotPrem 2023/24",     LEAGUE_IDS["ScotPrem"],     2023),
    ("ScotPrem 2024/25",     LEAGUE_IDS["ScotPrem"],     2024),
    ("ScotPrem 2025/26",     LEAGUE_IDS["ScotPrem"],     2025),
    ("Brazil 2024",          LEAGUE_IDS["BrazilSerieA"], 2024),
    ("Brazil 2025",          LEAGUE_IDS["BrazilSerieA"], 2025),
    ("Brazil 2026",          LEAGUE_IDS["BrazilSerieA"], 2026),
    ("LigaMX 2024",          LEAGUE_IDS["LigaMX"],       2024),
    ("LigaMX 2025",          LEAGUE_IDS["LigaMX"],       2025),
    ("LigaMX 2026",          LEAGUE_IDS["LigaMX"],       2026),
    # UEFA top-16 completion (Jul 2026): ranks 10-16
    ("Ekstraklasa 2023/24",  LEAGUE_IDS["Ekstraklasa"],  2023),
    ("Ekstraklasa 2024/25",  LEAGUE_IDS["Ekstraklasa"],  2024),
    ("Ekstraklasa 2025/26",  LEAGUE_IDS["Ekstraklasa"],  2025),
    ("CzechLiga 2023/24",    LEAGUE_IDS["CzechLiga"],    2023),
    ("CzechLiga 2024/25",    LEAGUE_IDS["CzechLiga"],    2024),
    ("CzechLiga 2025/26",    LEAGUE_IDS["CzechLiga"],    2025),
    ("GreeceSL 2023/24",     LEAGUE_IDS["GreeceSL"],     2023),
    ("GreeceSL 2024/25",     LEAGUE_IDS["GreeceSL"],     2024),
    ("GreeceSL 2025/26",     LEAGUE_IDS["GreeceSL"],     2025),
    ("DKSuperliga 2023/24",  LEAGUE_IDS["DKSuperliga"],  2023),
    ("DKSuperliga 2024/25",  LEAGUE_IDS["DKSuperliga"],  2024),
    ("DKSuperliga 2025/26",  LEAGUE_IDS["DKSuperliga"],  2025),
    ("Eliteserien 2024",     LEAGUE_IDS["Eliteserien"],  2024),
    ("Eliteserien 2025",     LEAGUE_IDS["Eliteserien"],  2025),
    ("Eliteserien 2026",     LEAGUE_IDS["Eliteserien"],  2026),
    ("CyprusD1 2023/24",     LEAGUE_IDS["CyprusD1"],     2023),
    ("CyprusD1 2024/25",     LEAGUE_IDS["CyprusD1"],     2024),
    ("CyprusD1 2025/26",     LEAGUE_IDS["CyprusD1"],     2025),
    ("SwissSL 2023/24",      LEAGUE_IDS["SwissSL"],      2023),
    ("SwissSL 2024/25",      LEAGUE_IDS["SwissSL"],      2024),
    ("SwissSL 2025/26",      LEAGUE_IDS["SwissSL"],      2025),
    # UEFA ranks 17-22 completion (Jul 2026; Scotland already synced)
    ("HungaryNBI 2023/24",   LEAGUE_IDS["HungaryNBI"],   2023),
    ("HungaryNBI 2024/25",   LEAGUE_IDS["HungaryNBI"],   2024),
    ("HungaryNBI 2025/26",   LEAGUE_IDS["HungaryNBI"],   2025),
    ("Allsvenskan 2024",     LEAGUE_IDS["Allsvenskan"],  2024),
    ("Allsvenskan 2025",     LEAGUE_IDS["Allsvenskan"],  2025),
    ("Allsvenskan 2026",     LEAGUE_IDS["Allsvenskan"],  2026),
    ("RomaniaLigaI 2023/24", LEAGUE_IDS["RomaniaLigaI"], 2023),
    ("RomaniaLigaI 2024/25", LEAGUE_IDS["RomaniaLigaI"], 2024),
    ("RomaniaLigaI 2025/26", LEAGUE_IDS["RomaniaLigaI"], 2025),
    ("AustriaBL 2023/24",    LEAGUE_IDS["AustriaBL"],    2023),
    ("AustriaBL 2024/25",    LEAGUE_IDS["AustriaBL"],    2024),
    ("AustriaBL 2025/26",    LEAGUE_IDS["AustriaBL"],    2025),
    ("CroatiaHNL 2023/24",   LEAGUE_IDS["CroatiaHNL"],   2023),
    ("CroatiaHNL 2024/25",   LEAGUE_IDS["CroatiaHNL"],   2024),
    ("CroatiaHNL 2025/26",   LEAGUE_IDS["CroatiaHNL"],   2025),
    # Asia (Jul 2026): K1 mid-season; J1 switching to autumn-spring,
    # 2026 entry = transitional comp, add 2027 when it starts in August
    ("KLeague1 2024",        LEAGUE_IDS["KLeague1"],     2024),
    ("KLeague1 2025",        LEAGUE_IDS["KLeague1"],     2025),
    ("KLeague1 2026",        LEAGUE_IDS["KLeague1"],     2026),
    ("J1League 2024",        LEAGUE_IDS["J1League"],     2024),
    ("J1League 2025",        LEAGUE_IDS["J1League"],     2025),
    ("J1League 2026",        LEAGUE_IDS["J1League"],     2026),
    # expansion (Jul 2026): Serbia + China; new 25/26->26/27 rollover
    # seasons for Czech + Denmark (scanner leagues need current data)
    ("SerbiaSL 2023/24",     LEAGUE_IDS["SerbiaSL"],     2023),
    ("SerbiaSL 2024/25",     LEAGUE_IDS["SerbiaSL"],     2024),
    ("SerbiaSL 2025/26",     LEAGUE_IDS["SerbiaSL"],     2025),
    ("SerbiaSL 2026/27",     LEAGUE_IDS["SerbiaSL"],     2026),
    ("ChinaSL 2024",         LEAGUE_IDS["ChinaSL"],      2024),
    ("ChinaSL 2025",         LEAGUE_IDS["ChinaSL"],      2025),
    ("ChinaSL 2026",         LEAGUE_IDS["ChinaSL"],      2026),
    ("CzechLiga 2026/27",    LEAGUE_IDS["CzechLiga"],    2026),
    ("DKSuperliga 2026/27",  LEAGUE_IDS["DKSuperliga"],  2026),
    # --- current seasons (added 2026-08-18). Without these the queue
    # topped out at 2025/26 and every European league sat at its May
    # final matchday no matter how often sync_all ran. Season numbers
    # verified against /leagues current=true: split-year seasons key on
    # the START year, EXCEPT J1, whose autumn-spring switch makes the
    # season starting 2026-08-07 season=2027 (2026 = transitional comp).
    ("EPL 2026/27",          LEAGUE_IDS["EPL"],          2026),
    ("LaLiga 2026/27",       LEAGUE_IDS["LaLiga"],       2026),
    ("Bundesliga 2026/27",   LEAGUE_IDS["Bundesliga"],   2026),
    ("SerieA 2026/27",       LEAGUE_IDS["SerieA"],       2026),
    ("Ligue1 2026/27",       LEAGUE_IDS["Ligue1"],       2026),
    ("Championship 2026/27", LEAGUE_IDS["Championship"], 2026),
    ("Eredivisie 2026/27",   LEAGUE_IDS["Eredivisie"],   2026),
    ("Primeira 2026/27",     LEAGUE_IDS["PrimeiraLiga"], 2026),
    ("JupilerPro 2026/27",   LEAGUE_IDS["JupilerPro"],   2026),
    ("ScotPrem 2026/27",     LEAGUE_IDS["ScotPrem"],     2026),
    ("AustriaBL 2026/27",    LEAGUE_IDS["AustriaBL"],    2026),
    ("SuperLig 2026/27",     LEAGUE_IDS["SuperLig"],     2026),
    ("Ekstraklasa 2026/27",  LEAGUE_IDS["Ekstraklasa"],  2026),
    ("GreeceSL 2026/27",     LEAGUE_IDS["GreeceSL"],     2026),
    ("CyprusD1 2026/27",     LEAGUE_IDS["CyprusD1"],     2026),
    ("SwissSL 2026/27",      LEAGUE_IDS["SwissSL"],      2026),
    ("HungaryNBI 2026/27",   LEAGUE_IDS["HungaryNBI"],   2026),
    ("RomaniaLigaI 2026/27", LEAGUE_IDS["RomaniaLigaI"], 2026),
    ("CroatiaHNL 2026/27",   LEAGUE_IDS["CroatiaHNL"],   2026),
    ("J1League 2026/27",     LEAGUE_IDS["J1League"],     2027),
    ("MLS 2026",             LEAGUE_IDS["MLS"],          2026),
    # scanner prices Argentina and the indep engine reads league 128,
    # but it had no sync entry at all until now
    ("Argentina 2025",       LEAGUE_IDS["ArgentinaPrimera"], 2025),
    ("Argentina 2026",       LEAGUE_IDS["ArgentinaPrimera"], 2026),
]

if __name__ == "__main__":
    total_spent = 0
    for label, league_id, season in QUEUE:
        print(f"=== {label} ===", flush=True)
        try:
            c = sync_league_season(league_id, season,
                                   max_requests=2000, min_interval=0.15)
        except Exception as e:
            print(f"  [WARN] {label} failed: {e} - continuing", flush=True)
            continue
        total_spent += c["requests_spent"]
        print(f"  fixtures={c['fixtures']} events={c['events']} "
              f"player_stats={c['player_stats']} "
              f"pending={c['fixtures_pending']} "
              f"spent={c['requests_spent']} (run total {total_spent})",
              flush=True)
    print(f"\nDone. Total requests this run: {total_spent}", flush=True)
