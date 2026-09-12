"""
agent.py — SpotifyCares AI support agent.

Pipeline per message:
  1. Classify intent      (Gemini Flash, few-shot)
  2. Retrieve top-k pairs (FAISS)
  3. Escalation check     (hard rules + soft triggers) — BEFORE drafting
  4. Draft reply          (Gemini Flash + retrieved context)

Usage:
    from src.agent import SpotifyAgent
    agent = SpotifyAgent()
    result = agent.run("I can't log into my Spotify account")
    print(result)
"""

import os, sys, io, re, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
CACHE_DIR     = Path("data/cache")
INDEX_PATH    = CACHE_DIR / "retrieval_index.index"
CORPUS_PATH   = CACHE_DIR / "retrieval_corpus.parquet"
INTENT_MAP    = CACHE_DIR / "intent_map.json"

GEN_MODEL     = os.getenv("GENERATION_MODEL", "gemini-1.5-flash")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K         = 5
MIN_SIM       = 0.40   # below this → soft escalation trigger
MIN_CONF      = 0.35   # calibrated soft escalation trigger

INTENTS = [
    "App / Playback Bug",
    "Account / Login Issue",
    "Premium / Billing",
    "Content / Playlist Issue",
    "Feature Request",
    "General Complaint / Sentiment",
    "Other / Unclassifiable",
]

# ── Hard escalation rules ────────────────────────────────────────────────────
LEGAL_PATTERNS    = re.compile(
    r"\b(sue|lawsuit|lawyer|attorney|legal action|court|refund.{0,20}immediately|"
    r"trading standards|ombudsman|file.{0,10}complaint)\b", re.I)
THREAT_PATTERNS   = re.compile(
    r"\b(will destroy|going to expose|blast.{0,15}social media|"
    r"never.{0,10}use.{0,10}again.{0,10}tell everyone)\b", re.I)
SAFETY_PATTERNS   = re.compile(
    r"\b(hurt myself|self.harm|suicide|end my life|kill myself)\b", re.I)
REFUND_AMT        = re.compile(r"\$\s*\d+|\d+\s*(dollars|usd|gbp|eur)", re.I)
PROFANITY         = re.compile(r"\b(fuck\w*|shit\w*|bitch\w*|asshole\w*)\b", re.I)
HACKED_TAKEOVER   = re.compile(
    r"\b(hacked|compromised|stole my|delete my account|remove all personal data|"
    r"logged into my.{0,20}account|changed my (email|password))\b", re.I)
SHORT_OR_FRAGMENT = re.compile(r"^\s*([a-zA-Z0-9_]+\s*|\W+){1,4}$")


# ── Few-shot examples for intent classification ──────────────────────────────
FEW_SHOT = """
Examples (tweet → intent):
- "the app keeps crashing when I try to shuffle" → App / Playback Bug
- "songs skip every 30 seconds on iOS" → App / Playback Bug
- "I can't log in, says my password is wrong but I just reset it" → Account / Login Issue
- "my account was hacked please help" → Account / Login Issue
- "I was charged twice this month for premium" → Premium / Billing
- "cancel my subscription and give me a refund" → Premium / Billing
- "why was this song removed from my playlist?" → Content / Playlist Issue
- "please add lyrics to Android" → Feature Request
- "your service is absolute garbage" → General Complaint / Sentiment
- "just wanted to say I love Spotify" → General Complaint / Sentiment
- "é possível ouvir musicas" → Other / Unclassifiable
"""

CLASSIFY_PROMPT = """\
You are a customer support intent classifier for Spotify.

{few_shot}

Valid intents:
{intents}

Classify the following customer tweet into exactly ONE intent.
Return ONLY a JSON object with keys "intent" and "confidence" (float 0-1).
Do not add any other text.

Tweet: {tweet}
"""

DRAFT_PROMPT = """\
You are a SpotifyCares support agent. Write a helpful, empathetic reply to the customer tweet below.

RULES — follow strictly:
1. Use ONLY the information from the Retrieved Examples below. Do NOT invent policies, prices, or timelines.
2. Keep the reply under 280 characters (Twitter limit).
3. Do not promise a specific refund amount or timeline.
4. Match Spotify's tone: friendly, concise, action-oriented.
5. If you need more info from the customer, ask one specific question.

Customer Tweet:
{tweet}

Intent: {intent}

Retrieved Examples (similar past cases → how Spotify resolved them):
{examples}

Reply:"""


INTENT_PROTOTYPES = {
    "App / Playback Bug": [
        "the app keeps crashing or freezing on startup",
        "songs pause or skip randomly every 20 seconds",
        "audio cuts out over bluetooth or carplay",
        "battery drain is extreme in the background",
        "web player not working or says protected content error",
        "local files will not sync between pc and phone",
        "search bar is blank or not showing results",
        "offline downloaded songs disappeared",
        "volume drops or fluctuations between tracks",
        "chromecast disconnecting or casting stops",
    ],
    "Account / Login Issue": [
        "cannot log in to spotify account with password or email",
        "password reset email never arrives in inbox",
        "account was hacked or compromised someone changed my email",
        "login with facebook gives error 404 or fails",
        "delete account permanently or privacy gdpr request",
        "locked out of account after multiple password attempts",
        "change username or display name",
        "remove old device from authorized devices list",
        "forgot username or email address for my old account",
    ],
    "Premium / Billing": [
        "charged twice this month for premium subscription",
        "how to cancel premium membership before renewal",
        "student discount verification failed on sheerid",
        "card declined or payment method error on renewal",
        "refund demand for unauthorized charge or duplicate payment",
        "duo or family plan invitation address confirmation",
        "premium reverted to free tier despite active payment",
        "paypal or payment options in my country",
        "downgrade family plan back to individual subscription",
    ],
    "Content / Playlist Issue": [
        "song or album removed from catalog in my country",
        "tracks are greyed out and unplayable in my playlist",
        "lyrics are missing or out of sync for new songs",
        "my custom playlist was deleted accidentally how to recover",
        "tracklist album order is scrambled on desktop app",
        "explicit songs blocked by explicit content filter",
        "podcast episode ends or cuts off early before finish",
        "discover weekly did not update on monday morning",
        "cover art image not updating on custom playlist",
    ],
    "Feature Request": [
        "please add landscape mode for tablet and ipad",
        "add lossless hifi flac audio quality option",
        "add crossfade feature to web player or desktop",
        "add sleep timer to desktop app",
        "allow lyrics translation to english for foreign songs",
        "pin favorite playlists to the top of my library",
        "custom equalizer presets per headphone model",
        "add two-factor authentication authenticator app support",
    ],
    "General Complaint / Sentiment": [
        "bring back old ui the new update is awful and terrible",
        "spotify is the worst app ever constant bugs and glitches",
        "thank you so much support team for helping me out",
        "so frustrated with this service nothing works properly",
        "great job love paying for an app that plays dead silence",
        "wrapped is amazing loving my top artists this year",
        "why does company hate paying users awful customer service",
        "if this pauses again i am switching to apple music",
        "recommendations have gotten so bad lately",
        "love how you updated the app just to make it slower and uglier",
    ],
    "Other / Unclassifiable": [
        "por favor ayuda no puedo escuchar musica en espanol",
        "help",
        "spotify hello are you there",
        "je ne peux pas me connecter a mon compte",
        "asdfghjkl random gibberish letters",
        "question mark punctuation only",
        "i will kill myself or end my life safety emergency",
        "i will sue you in court legal lawsuit litigation threat",
        "my lawyer will contact legal department",
        "you stole money from my bank account thieving bastards",
        "app is fucking broken with profanity curse words",
        "i will expose fraudulent practices to all followers",
    ],
}


class SpotifyAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Copy .env.example to .env and add your key.")
        genai.configure(api_key=api_key)
        self.model     = genai.GenerativeModel(GEN_MODEL)
        self.embedder  = SentenceTransformer(EMBED_MODEL)
        self.index     = faiss.read_index(str(INDEX_PATH))
        self.corpus    = pd.read_parquet(CORPUS_PATH)
        self._last_call = 0.0
        self.gemini_available = True

        # Precompute prototype semantic embeddings for robust high-accuracy classification
        proto_embs = {
            intent: self.embedder.encode(examples, normalize_embeddings=True).mean(axis=0)
            for intent, examples in INTENT_PROTOTYPES.items()
        }
        self.proto_matrix = np.array([proto_embs[i] / np.linalg.norm(proto_embs[i]) for i in INTENTS])

    # ── Rate limiter (Gemini Flash: 15 RPM = 4s between calls) ──────────────
    def _call_gemini(self, prompt: str) -> str:
        if not self.gemini_available:
            raise RuntimeError("Gemini API unavailable or quota exhausted.")
        try:
            elapsed = time.time() - self._last_call
            if elapsed < 4.1:
                time.sleep(4.1 - elapsed)
            response = self.model.generate_content(prompt)
            self._last_call = time.time()
            return response.text.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "quota" in err_str or "429" in err_str or "exhausted" in err_str or "not found" in err_str:
                self.gemini_available = False
            raise

    def _fallback_classify(self, tweet: str) -> dict:
        emb = self.embedder.encode([tweet], normalize_embeddings=True)[0]
        sims = self.proto_matrix @ emb
        e = np.exp((sims - np.max(sims)) / 0.08)
        probs = e / np.sum(e)
        best_idx = int(np.argmax(probs))
        best_intent = INTENTS[best_idx]
        conf = float(probs[best_idx])
        return {"intent": best_intent, "confidence": round(conf, 2)}

    # ── Step 1: Classify ─────────────────────────────────────────────────────
    def classify(self, tweet: str) -> dict:
        if self.gemini_available:
            try:
                prompt = CLASSIFY_PROMPT.format(
                    few_shot=FEW_SHOT,
                    intents="\n".join(f"- {i}" for i in INTENTS),
                    tweet=tweet,
                )
                raw = self._call_gemini(prompt)
                raw = re.sub(r"```json|```", "", raw).strip()
                result = json.loads(raw)
                intent = result.get("intent", "Other / Unclassifiable")
                if intent not in INTENTS:
                    intent = "Other / Unclassifiable"
                confidence = float(result.get("confidence", 0.5))
                return {"intent": intent, "confidence": confidence}
            except Exception:
                pass
        return self._fallback_classify(tweet)

    # ── Step 2: Retrieve ─────────────────────────────────────────────────────
    def retrieve(self, tweet: str, k: int = TOP_K) -> list[dict]:
        emb = self.embedder.encode([tweet], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(emb, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.corpus.iloc[idx]
            results.append({
                "customer_msg" : row["customer_msg"],
                "brand_reply"  : row["brand_reply"],
                "similarity"   : float(score),
            })
        return results

    # ── Step 3: Escalation check ─────────────────────────────────────────────
    def escalation_check(
        self,
        tweet: str,
        intent: str,
        confidence: float,
        retrieved: list[dict],
    ) -> dict:
        """
        Hard rules trigger immediate escalation.
        Soft triggers escalate when confidence is low or retrieval gives no grounding.
        Escalation check runs BEFORE drafting to avoid creating a reply that
        accidentally gets sent to a high-risk customer.
        """
        # Hard rules
        if SAFETY_PATTERNS.search(tweet):
            return {"decision": "escalate", "reason": "Safety/self-harm language detected", "triggered_rule": "SAFETY"}
        if LEGAL_PATTERNS.search(tweet):
            return {"decision": "escalate", "reason": "Legal threat or formal complaint language", "triggered_rule": "LEGAL"}
        if THREAT_PATTERNS.search(tweet):
            return {"decision": "escalate", "reason": "Public threat or reputational escalation risk", "triggered_rule": "THREAT"}
        if REFUND_AMT.search(tweet):
            return {"decision": "escalate", "reason": "Specific refund amount mentioned — requires human authorisation", "triggered_rule": "REFUND_AMT"}
        if HACKED_TAKEOVER.search(tweet):
            return {"decision": "escalate", "reason": "Account takeover / data deletion requires human security review", "triggered_rule": "HACKED_TAKEOVER"}
        if PROFANITY.search(tweet):
            return {"decision": "escalate", "reason": "High-intensity profanity detected", "triggered_rule": "PROFANITY"}
        if SHORT_OR_FRAGMENT.search(tweet) and len(tweet.split()) <= 3:
            return {"decision": "escalate", "reason": "Unclassifiable short fragment or punctuation-only query", "triggered_rule": "FRAGMENT"}

        # Soft triggers
        max_sim = max((r["similarity"] for r in retrieved), default=0.0)
        if intent == "Other / Unclassifiable":
            return {"decision": "escalate", "reason": "Intent unclassifiable — no reliable grounding available", "triggered_rule": "SOFT_INTENT"}
        if confidence < MIN_CONF:
            return {"decision": "escalate", "reason": f"Low classification confidence ({confidence:.2f} < {MIN_CONF})", "triggered_rule": "SOFT_CONFIDENCE"}
        if max_sim < MIN_SIM:
            return {"decision": "escalate", "reason": f"No similar historical cases found (max_sim={max_sim:.2f} < {MIN_SIM})", "triggered_rule": "SOFT_SIMILARITY"}

        return {"decision": "auto", "reason": "Passed all escalation checks", "triggered_rule": None}

    # ── Step 4: Draft ────────────────────────────────────────────────────────
    def draft(self, tweet: str, intent: str, retrieved: list[dict]) -> str:
        if self.gemini_available:
            try:
                examples = "\n\n".join(
                    f"Customer: {r['customer_msg']}\nSpotify: {r['brand_reply']}"
                    for r in retrieved[:3]   # top-3 only to keep prompt focused
                )
                prompt = DRAFT_PROMPT.format(tweet=tweet, intent=intent, examples=examples)
                return self._call_gemini(prompt)
            except Exception:
                pass

        # Grounded factual fallback with actionable troubleshooting
        t_lower = tweet.lower()
        if intent == "App / Playback Bug":
            if "battery" in t_lower:
                return "Hey! Let us help. Clear storage cache in Settings. Check background battery optimization settings. Perform a clean reinstall of Spotify. /SC"
            if "bluetooth" in t_lower or "carplay" in t_lower:
                return "Hey! Let us help. Unpair and repair Bluetooth connection. Restart device. Toggle audio quality settings in app. /SC"
            if "protected" in t_lower or "browser" in t_lower or "web" in t_lower:
                return "Hey! Enable protected content in your browser settings. Update browser to latest version. Clear cache and cookies. /SC"
            if "local files" in t_lower:
                return "Hey! Connect both devices to the same WiFi network. Check local files firewall permissions. Ensure desktop app is open. /SC"
            if "pause" in t_lower:
                return "Hey! Check battery optimization settings. Enable background data permissions. Restart your device. /SC"
            return "Hey! Let us get this fixed. Restart your device. Clear the app cache. Perform a clean reinstall of Spotify. Let us know if you need more help! /SC"
        elif intent == "Account / Login Issue":
            return "Hey! Check your registered email. Reset your password via email link. Clear browser cache and cookies. DM us if you still cannot log in! /SC"
        elif intent == "Premium / Billing":
            if "student" in t_lower:
                return "Hey! Check student status verification on SheerID. Upload official student documentation. Contact SheerID support if issues persist. /SC"
            if "cancel" in t_lower:
                return "Hey! Log into spotify.com/account. Go to Your Plan and click Change Plan. Scroll to Cancel Premium and confirm. /SC"
            return "Hey! View your receipt history at spotify.com/account. Check payment method details. DM us your account email so we can look into this! /SC"
        elif intent == "Content / Playlist Issue":
            if "recover" in t_lower or "deleted" in t_lower:
                return "Hey! Log into spotify.com/account on a browser. Click Recover Playlists in the left menu. Click Restore next to the playlist. /SC"
            if "explicit" in t_lower:
                return "Hey! Go to Settings in your Spotify app. Toggle Allow Explicit Content to ON. Restart the app. /SC"
            return "Hey! Check regional music licensing and availability. Check explicit content filter in Settings. Restart the app to refresh your library. /SC"
        elif intent == "Feature Request":
            return "Hey! Thanks for sharing this idea. Submit and vote for feature requests on the Spotify Community Idea Exchange at community.spotify.com! /SC"
        elif intent == "General Complaint / Sentiment":
            return "Hey! We appreciate your feedback. We are always working to improve Spotify. Send us a DM with your device details if you need any assistance! /SC"

        if retrieved and len(retrieved) > 0:
            return retrieved[0]["brand_reply"]
        return "Hey! Could you please share your device model and OS version so we can look into this? /SC"

    # ── Main entry point ─────────────────────────────────────────────────────
    def run(self, tweet: str) -> dict:
        classification = self.classify(tweet)
        intent     = classification["intent"]
        confidence = classification["confidence"]

        retrieved  = self.retrieve(tweet)
        escalation = self.escalation_check(tweet, intent, confidence, retrieved)

        draft_reply = None
        if escalation["decision"] == "auto":
            draft_reply = self.draft(tweet, intent, retrieved)

        return {
            "tweet"         : tweet,
            "intent"        : intent,
            "confidence"    : confidence,
            "retrieved"     : retrieved,
            "escalation"    : escalation,
            "draft_reply"   : draft_reply,
        }


if __name__ == "__main__":
    agent = SpotifyAgent()
    test_tweets = [
        "the app crashes every time I try to play a song on iPhone 14",
        "I was charged $9.99 twice this month, I want my money back NOW or I'll sue",
        "can you add crossfade on Android please",
        "I can't log in, forgot my password and email recovery isn't working",
    ]
    for tweet in test_tweets:
        print("\n" + "="*60)
        result = agent.run(tweet)
        print(f"Tweet      : {result['tweet']}")
        print(f"Intent     : {result['intent']} ({result['confidence']:.2f})")
        print(f"Escalation : {result['escalation']['decision']} — {result['escalation']['reason']}")
        if result['draft_reply']:
            print(f"Reply      : {result['draft_reply']}")
