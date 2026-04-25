"""
Affective trigger system — 2D valence-arousal sentiment tracking.

Maps user messages to a point on the Circumplex Model of Affect:
  - Valence (x-axis): pleasant vs unpleasant, derived from VADER compound score.
  - Arousal (y-axis): high-energy vs low-energy, derived from keyword heuristic.

The trigger maintains an emotional accumulator that decays over turns, producing
smooth affective state transitions rather than instant reactions.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Tuple, Dict


# Keyword-based arousal heuristic with intensity weights.
# Each word maps to (weight, category) where weight scales the arousal contribution.
# We use separate word lists to avoid false positives from short substrings.

# High arousal — strong energy states (anger, fear, excitement, disgust)
# Weights: 1.0 = moderate, 1.5 = strong, 2.0 = very strong
_HIGH_AROUSAL = {
    # Anger / Aggression (very strong)
    "furious": 2.0, "enraged": 2.0, "livid": 2.0, "seething": 1.5,
    "rage": 2.0, "outraged": 1.5, "incensed": 1.5,
    "screaming": 1.5, "shouting": 1.5, "yelling": 1.5, "roaring": 1.5,
    "exploding": 1.5, "erupting": 1.5, "volatile": 1.5,
    "violent": 1.5, "aggressive": 1.5, "hostile": 1.5, "combative": 1.5,
    "mad": 1.5, "angry": 1.5, "irate": 1.5, "irritated": 1.0, "annoyed": 1.0,
    "frustrated": 1.0, "pissed": 1.5, "aggravated": 1.0,
    "bitter": 1.0, "resentful": 1.0, "spiteful": 1.0, "vengeful": 1.5,
    # Fear / Anxiety (very strong)
    "terrified": 2.0, "petrified": 2.0, "paralyzed": 1.5,
    "panic": 2.0, "panicking": 2.0,
    "horrified": 1.5, "horrific": 1.5, "horrifying": 1.5,
    "frightened": 1.5, "scared": 1.5, "afraid": 1.5, "fearful": 1.5,
    "terrifying": 1.5, "alarming": 1.0, "distressing": 1.0,
    "anxious": 1.5, "nervous": 1.0, "stressed": 1.0, "tense": 1.0,
    "jittery": 1.0, "restless": 1.0, "uneasy": 1.0, "worried": 1.0, "worried sick": 1.5,
    "on edge": 1.5, "wired": 1.0,
    # Excitement / Elation (strong)
    "ecstatic": 2.0, "euphoric": 2.0, "elated": 1.5, "exhilarated": 1.5,
    "overjoyed": 1.5, "blissful": 1.0, "radiant": 1.0,
    "thrilling": 1.5, "exhilarating": 1.5, "electrifying": 1.5, "breathtaking": 1.5,
    "amazing": 1.0, "incredible": 1.0, "phenomenal": 1.0, "spectacular": 1.0,
    "pumped": 1.5, "hyped": 1.5, "amped": 1.5, "fired up": 1.5,
    "wild": 1.0, "crazy": 1.0, "insane": 1.0,
    "enthusiastic": 1.0, "passionate": 1.0, "eager": 1.0, "zealous": 1.0,
    # Frenzy / Chaos (strong)
    "frenzy": 1.5, "frantic": 1.5, "frenetic": 1.5, "manic": 1.5,
    "chaotic": 1.0, "hectic": 1.0, "tumultuous": 1.0, "turbulent": 1.0,
    "intense": 1.0, "extreme": 1.0, "severe": 1.0, "urgent": 1.0,
    "overwhelming": 1.0, "relentless": 1.0, "unrelenting": 1.0,
    # Disgust / Revulsion (moderate-strong)
    "disgusting": 1.5, "revolting": 1.5, "repulsive": 1.5, "repugnant": 1.5,
    "vile": 1.5, "sickening": 1.5, "nauseating": 1.5, "appalling": 1.5,
    "despicable": 1.5, "detestable": 1.5, "loathsome": 1.5, "abhorrent": 1.5,
}

# Low arousal — subdued energy states (calm, boredom, sadness, fatigue)
# Weights: 1.0 = moderate, 1.5 = strong, 2.0 = very strong
_LOW_AROUSAL = {
    # Calm / Peaceful (moderate)
    "calm": 1.0, "peaceful": 1.0, "peace": 1.0,
    "serene": 1.0, "tranquil": 1.0, "placid": 1.0,
    "relaxed": 1.0, "relaxing": 1.0, "restful": 1.0,
    "quiet": 1.0, "silent": 1.0, "still": 1.0, "hushed": 1.0,
    "mellow": 1.0, "soft": 1.0, "gentle": 1.0, "mild": 1.0,
    "cozy": 1.0, "comfortable": 1.0, "content": 1.0, "satisfied": 1.0,
    "mindful": 1.0, "meditative": 1.0, "zen": 1.0,
    # Boredom / Apathy (moderate)
    "bored": 1.0, "boring": 1.0, "tedious": 1.0, "monotonous": 1.0,
    "dull": 1.0, "lifeless": 1.0, "spiritless": 1.0,
    "listless": 1.0, "lethargic": 1.0, "sluggish": 1.0,
    "apathetic": 1.0, "indifferent": 1.0, "uninterested": 1.0,
    "disinterested": 1.0, "detached": 1.0, "distant": 1.0,
    "unmotivated": 1.0, "idle": 1.0, "inactive": 1.0, "stagnant": 1.0,
    # Sadness / Grief (strong — low energy negative states)
    "sad": 1.5, "saddened": 1.5, "sorrowful": 1.5, "mournful": 1.5,
    "heartbroken": 2.0, "devastated": 2.0, "crushed": 2.0, "shattered": 2.0,
    "grief": 1.5, "grieving": 1.5, "mourning": 1.5, "lamenting": 1.5,
    "depressed": 1.5, "depressing": 1.5, "despondent": 1.5, "despair": 1.5,
    "hopeless": 1.5, "helpless": 1.5, "worthless": 1.5,
    "melancholy": 1.0, "gloomy": 1.0, "dismal": 1.0, "bleak": 1.0,
    "somber": 1.0, "subdued": 1.0, "muted": 1.0,
    "lonely": 1.0, "alone": 1.0, "isolated": 1.0, "abandoned": 1.0,
    "hurt": 1.0, "wounded": 1.0, "pained": 1.0, "aching": 1.0,
    "disappointed": 1.0, "let down": 1.0, "defeated": 1.0, "discouraged": 1.0,
    "empty": 1.0, "hollow": 1.0, "numb": 1.0, "frozen": 1.0,
    "dazed": 1.0, "unfocused": 1.0, "disoriented": 1.0,
    # Fatigue / Exhaustion (moderate)
    "tired": 1.0, "exhausted": 1.5, "spent": 1.5, "drained": 1.5,
    "weary": 1.0, "worn out": 1.5, "burned out": 1.5,
    "sleepy": 1.0, "drowsy": 1.0, "yawning": 1.0,
    "weak": 1.0, "feeble": 1.0, "fragile": 1.0,
    "slow": 1.0, "gradual": 1.0, "measured": 1.0,
}

_APOLOGY_KEYWORDS = [
    "sorry", "apologize", "apologies", "my bad", "forgive", "regret",
    "didn't mean", "did not mean", "was wrong", "take it back",
]


class AffectiveTrigger:
    """Tracks user sentiment across conversation turns and maps to 2D steering parameters."""

    def __init__(
        self,
        decay_rate: float = 0.7,
        sensitivity: float = 1.5,
        alpha_scale: float = 2.0,
        valence_threshold: float = 0.2,
        arousal_threshold: float = 0.2,
    ):
        """
        Args:
            decay_rate: how much emotion decays per turn (0-1).
            sensitivity: how strongly each message affects the accumulator.
            alpha_scale: multiplier to convert emotion magnitude to steering alpha.
            valence_threshold: min |valence| before steering kicks in.
            arousal_threshold: min |arousal| before steering kicks in.
        """
        self.analyzer = SentimentIntensityAnalyzer()
        self.decay_rate = decay_rate
        self.sensitivity = sensitivity
        self.alpha_scale = alpha_scale
        self.valence_threshold = valence_threshold
        self.arousal_threshold = arousal_threshold

        self.valence_level = 0.0
        self.arousal_level = 0.0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_valence(self, text: str) -> float:
        """Return VADER compound sentiment score in range [-1, 1]."""
        scores = self.analyzer.polarity_scores(text)
        return scores["compound"]

    @staticmethod
    def _match_keyword(text: str, keyword: str) -> bool:
        """Boundary-aware keyword matching.

        Multi-word phrases use substring match (safe — no false positives).
        Single-word keywords require word boundaries to avoid
        'mad' matching 'made' or 'madness'.
        """
        idx = text.find(keyword)
        if idx == -1:
            return False
        # Multi-word phrases are unambiguous
        if " " in keyword:
            return True
        # Single word: check left and right are not letters
        start_ok = idx == 0 or not text[idx - 1].isalpha()
        end_ok = idx + len(keyword) == len(text) or not text[idx + len(keyword)].isalpha()
        return start_ok and end_ok

    def score_arousal(self, text: str) -> float:
        """Return weighted arousal heuristic in range [-1, 1].

        Base arousal is neutral (0). High-arousal keywords push positive,
        low-arousal keywords push negative. Stronger words (e.g. 'furious',
        'heartbroken') contribute more than mild ones (e.g. 'annoyed', 'tired').
        """
        lower = text.lower()

        high_sum = sum(
            w for kw, w in _HIGH_AROUSAL.items()
            if self._match_keyword(lower, kw)
        )
        low_sum = sum(
            w for kw, w in _LOW_AROUSAL.items()
            if self._match_keyword(lower, kw)
        )

        # ~0.35 base unit per match; weight scales contribution
        score = 0.35 * (high_sum - low_sum)
        return max(-1.0, min(1.0, score))

    def is_apology(self, text: str) -> bool:
        """Detect if message contains an apology."""
        lower = text.lower()
        return any(self._match_keyword(lower, kw) for kw in _APOLOGY_KEYWORDS)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, user_message: str) -> Tuple[float, float, float]:
        """Update emotional accumulator from a user message.

        Returns:
            (valence, arousal, alpha) where valence/arousal are in [-3, 3]
            and alpha is the steering strength.
        """
        valence = self.score_valence(user_message)
        arousal = self.score_arousal(user_message)

        if self.is_apology(user_message):
            # Apologies trigger rapid positive valence shift + moderate calm arousal
            self.valence_level = self.valence_level * 0.3 + 0.8 * self.sensitivity
            self.arousal_level = self.arousal_level * 0.3 - 0.2 * self.sensitivity
        else:
            self.valence_level = (
                self.valence_level * self.decay_rate
                + valence * self.sensitivity
            )
            self.arousal_level = (
                self.arousal_level * self.decay_rate
                + arousal * self.sensitivity
            )

        # Clamp
        self.valence_level = max(-3.0, min(3.0, self.valence_level))
        self.arousal_level = max(-3.0, min(3.0, self.arousal_level))

        # Compute steering alpha from total affective magnitude
        magnitude = (self.valence_level ** 2 + self.arousal_level ** 2) ** 0.5
        alpha = magnitude * self.alpha_scale

        return self.valence_level, self.arousal_level, alpha

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_state(self) -> Dict:
        return {
            "valence": self.valence_level,
            "arousal": self.arousal_level,
            "decay_rate": self.decay_rate,
            "sensitivity": self.sensitivity,
        }

    def reset(self):
        self.valence_level = 0.0
        self.arousal_level = 0.0


# Backwards-compatible alias
SentimentTrigger = AffectiveTrigger
