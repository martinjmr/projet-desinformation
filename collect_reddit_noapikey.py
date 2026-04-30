"""
Script de collecte Reddit SANS clé API
Utilise l'API JSON publique de Reddit (pas besoin de compte développeur)

Installation : pip install requests pandas
Usage        : python collect_reddit_noapikey.py
"""

import requests
import pandas as pd
import time
import os

# ------------------------------------------------------------------ #
#  CONFIG                                                              #
# ------------------------------------------------------------------ #
# Subreddits français à interroger
SUBREDDITS = ["france", "francophonie", "ActualiteFrance", "politique"]

# Nombre max de posts par requête (max 100)
MAX_POSTS = 100

# Fichier de sortie
OUTPUT_FILE = "posts_raw.csv"

# Headers pour simuler un navigateur (obligatoire sinon Reddit bloque)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
# ------------------------------------------------------------------ #


def collect_posts_for_query(mots_cles: str, label: str, source: str,
                             theme: str, affirmation: str) -> list:
    """
    Collecte les posts Reddit via l'API JSON publique (sans clé).
    """
    results = []

    for sub in SUBREDDITS:
        url = f"https://www.reddit.com/r/{sub}/search.json"
        params = {
            "q":        mots_cles,
            "sort":     "relevance",
            "t":        "all",       # toutes les périodes
            "limit":    MAX_POSTS,
            "restrict_sr": "true",   # chercher uniquement dans ce subreddit
        }

        try:
            response = requests.get(url, headers=HEADERS,
                                    params=params, timeout=10)

            if response.status_code == 429:
                print(f"    r/{sub} → Rate limit, attente 60s...")
                time.sleep(60)
                response = requests.get(url, headers=HEADERS,
                                        params=params, timeout=10)

            if response.status_code != 200:
                print(f"    r/{sub} → Erreur {response.status_code}")
                continue

            data = response.json()
            posts = data.get("data", {}).get("children", [])

            count = 0
            for post in posts:
                p = post["data"]

                # Ignorer les posts supprimés
                if p.get("selftext") in ["[removed]", "[deleted]"]:
                    continue

                # Titre + corps du post
                texte = p.get("title", "")
                corps = p.get("selftext", "")
                if corps:
                    texte = texte + " " + corps

                results.append({
                    "post_id":     p["id"],
                    "texte":       texte.strip(),
                    "label":       label,
                    "source_fc":   source,
                    "theme":       theme,
                    "affirmation": affirmation,
                    "subreddit":   sub,
                    "date":        pd.Timestamp(p["created_utc"], unit="s"),
                    "score":       p.get("score", 0),
                    "nb_comments": p.get("num_comments", 0),
                    "url":         "https://reddit.com" + p.get("permalink", ""),
                    "query":       mots_cles,
                })
                count += 1

            print(f"    r/{sub} → {count} posts")

        except Exception as e:
            print(f"    r/{sub} → Erreur : {e}")

        time.sleep(2)  # pause polie entre requêtes

    return results


def main():
    if not os.path.exists("factchecks_reddit.csv"):
        print("ERREUR : factchecks_reddit.csv introuvable.")
        return

    df_fc = pd.read_csv("factchecks_reddit.csv")
    print(f"Chargement de {len(df_fc)} fact-checks")
    print(f"Distribution : {df_fc['verdict'].value_counts().to_dict()}\n")

    # Reprise si interruption
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        already_done = set(df_existing["affirmation"].unique())
        print(f"Reprise : {len(df_existing)} posts déjà collectés, "
              f"{len(already_done)} requêtes déjà traitées\n")
    else:
        df_existing = pd.DataFrame()
        already_done = set()

    all_posts = []
    total = len(df_fc)

    for i, row in df_fc.iterrows():
        affirmation = row["affirmation"]

        if affirmation in already_done:
            print(f"[{i+1}/{total}] SKIP : {affirmation[:60]}")
            continue

        print(f"\n[{i+1}/{total}] {row['verdict'].upper()} — {row['theme']}")
        print(f"  Affirmation : {affirmation[:70]}")
        print(f"  Mots-clés   : {row['mots_cles_twitter']}")

        posts = collect_posts_for_query(
            mots_cles   = row["mots_cles_twitter"],
            label       = row["verdict"],
            source      = row["source"],
            theme       = row["theme"],
            affirmation = affirmation,
        )

        print(f"  → Total : {len(posts)} posts collectés")
        all_posts.extend(posts)

        # Sauvegarde incrémentale toutes les 5 requêtes
        if (i + 1) % 5 == 0 and all_posts:
            df_save = pd.concat(
                [df_existing, pd.DataFrame(all_posts)], ignore_index=True
            )
            df_save.drop_duplicates(subset=["post_id"], inplace=True)
            df_save.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
            print(f"  Sauvegarde : {len(df_save)} posts au total")

        time.sleep(3)

    # Sauvegarde finale
    if all_posts:
        df_final = pd.concat(
            [df_existing, pd.DataFrame(all_posts)], ignore_index=True
        )
    else:
        df_final = df_existing

    df_final.drop_duplicates(subset=["post_id"], inplace=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n" + "="*60)
    print("COLLECTE TERMINÉE")
    print(f"Total posts : {len(df_final)}")
    print(f"\nDistribution des labels :")
    print(df_final["label"].value_counts().to_string())
    print(f"\nDistribution par subreddit :")
    print(df_final["subreddit"].value_counts().to_string())
    print(f"\nFichier sauvegardé : {OUTPUT_FILE}")
    print("="*60)


if __name__ == "__main__":
    main()
