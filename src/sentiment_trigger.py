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


# Keyword-based arousal heuristic
# High-arousal words push the score toward +1 (excited, angry, terrified)
# Low-arousal words push the score toward -1 (calm, bored, exhausted)
_HIGH_AROUSAL_KEYWORDS = [
    "furious", "rage", "screaming", "shouting", "exploding", "violent",
    "terrified", "panic", "horrified", "terrifying", "frightened", "scared",
    "ecstatic", "thrilling", "exhilarating", "electrifying", "amazing",
    "elated", "euphoric", "overjoyed", "pumped", "hyped", "wild",
    "anxious", "nervous", "stressed", "tense", "jittery", "restless",
    "frenzy", "chaotic", "hectic", "intense", "extreme", "urgent",
    "disgusting", "revolting", "repulsive", "vile", "sickening", "nauseating",
]

_LOW_AROUSAL_KEYWORDS = [
    "calm", "peaceful", "serene", "tranquil", "relaxed", "quiet",
    "bored", "dull", "listless", "lethargic", "apathetic", "indifferent",
    "tired", "exhausted", "sleepy", "drained", "weary", "sluggish",
    "mellow", "soft", "gentle", "slow", "still", "motionless",
    "numb", "empty", "hollow", "dazed", "unfocused", "lifeless",
    "depressed", "melancholy", "gloomy", "dismal", "somber", "subdued",
]

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

    def score_arousal(self, text: str) -> float:
        """Return arousal heuristic in range [-1, 1].

        Base arousal is neutral (0). High-arousal keywords push positive,
        low-arousal keywords push negative. If both appear, they partially cancel.
        """
        lower = text.lower()
        high_count = sum(1 for kw in _HIGH_AROUSAL_KEYWORDS if kw in lower)
        low_count = sum(1 for kw in _LOW_AROUSAL_KEYWORDS if kw in lower)

        # Each keyword match contributes ~0.35; cap at ±1
        score = 0.35 * (high_count - low_count)
        return max(-1.0, min(1.0, score))

    def is_apology(self, text: str) -> bool:
        """Detect if message contains an apology."""
        lower = text.lower()
        return any(kw in lower for kw in _APOLOGY_KEYWORDS)

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
