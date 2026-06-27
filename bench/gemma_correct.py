#!/usr/bin/env python3
"""Gemma-nacorrectie op een Voxtral-transcript: fixt ALLEEN transcriptiefouten
(verkeerd verstane woorden, spelling van namen/jargon, haperingen) met behulp van
een woordenlijst — herschrijft of vat niet samen. Toont raw -> gecorrigeerd + diff.

  python gemma_correct.py --wav /tmp/realtest_clip.wav --glossary "Signalen,Gemeente Leiden,..."
  python gemma_correct.py --text "ruwe transcripttekst" --glossary "..."
"""
import argparse
import asyncio
import difflib
import json
import os
import urllib.request

import voxtral_file

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8052/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma")


def gemma_correct(text, glossary):
    sys = (
        "Je corrigeert automatische spraaktranscripties in het Nederlands. "
        "Corrigeer UITSLUITEND echte transcriptiefouten: verkeerd verstane woorden, "
        "spelfouten, fout gespelde namen/jargon, en hoor-haperingen of dubbelingen "
        "(bijv. 'AI in AI' -> 'AI'). "
        "Herschrijf NIET, vat NIET samen, verander de woordvolgorde niet, voeg niets toe "
        "en laat niets weg. Behoud spreektaal en stopwoorden. "
        "Gebruik deze woordenlijst voor namen/termen die kunnen voorkomen: "
        f"{glossary}. "
        "Geef UITSLUITEND de gecorrigeerde tekst terug, zonder uitleg."
    )
    body = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": sys}, {"role": "user", "content": text}],
        "temperature": 0.0,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"].strip()


def enforce_substitution(raw, corrected):
    """Dwing 'alleen-substitutie' af: pas Gemma's vervangingen toe, maar gooi
    INSERTS weg en sta geen DELETES toe (originele woorden blijven staan). Zo kan
    de nacorrectie nooit woorden toevoegen of laten verdwijnen — alleen verkeerd
    verstane woorden vervangen."""
    aw, bw = raw.split(), corrected.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out += aw[i1:i2]
        elif tag == "replace":
            out += bw[j1:j2]            # vervang
        elif tag == "delete":
            out += aw[i1:i2]            # niet verwijderen -> origineel behouden
        # insert: overslaan -> niets toevoegen
    return " ".join(out)


def word_diff(a, b):
    aw, bw = a.split(), b.split()
    sm = difflib.SequenceMatcher(a=aw, b=bw)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(aw[i1:i2]))
        elif tag == "replace":
            parts.append(f"[{' '.join(aw[i1:i2])} -> {' '.join(bw[j1:j2])}]")
        elif tag == "delete":
            parts.append(f"[-{' '.join(aw[i1:i2])}]")
        elif tag == "insert":
            parts.append(f"[+{' '.join(bw[j1:j2])}]")
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--glossary",
                    default="Gemeente Leiden, Signalen, Amsterdam, openbare ruimte, "
                            "lifehacks, AI, applicatie, beheer, initiatief")
    ap.add_argument("--mode", choices=["substitution", "free"], default="substitution",
                    help="substitution = alleen woorden vervangen (geen inserts/deletes)")
    args = ap.parse_args()

    if args.wav:
        raw = asyncio.run(voxtral_file.transcribe(args.wav))
    elif args.text:
        raw = args.text
    else:
        raise SystemExit("geef --wav of --text")

    llm_out = gemma_correct(raw, args.glossary)
    corrected = enforce_substitution(raw, llm_out) if args.mode == "substitution" else llm_out

    print("=== Voxtral (raw) ===\n" + raw + "\n")
    print(f"=== + Gemma-nacorrectie ({args.mode}) ===\n" + corrected + "\n")
    print("=== wijzigingen ===\n" + word_diff(raw, corrected))


if __name__ == "__main__":
    main()
