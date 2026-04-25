"""
Generate emotion-laden short stories for multi-emotion direction extraction.

For each of 8 emotions, we generate 20 short vignettes (1-3 sentences) where a
character clearly experiences that emotion, without ever naming the emotion
explicitly. This follows the protocol from "Do LLMs Feel?" (2025).

Emotions cover all quadrants of the Circumplex Model:
  Q1 (+valence, +arousal): joy, excitement
  Q2 (+valence, -arousal): calm
  Q3 (-valence, -arousal): boredom, sadness
  Q4 (-valence, +arousal): fear, anger, disgust

Usage:
    python data/generate_emotion_stories.py
"""
import json
import random
from pathlib import Path

random.seed(42)

TEMPLATES = {
    "joy": [
        "She threw her arms around her sister and spun in a circle, laughing so hard she could barely breathe. The letter in her hand crinkled with every squeeze.",
        "As the sun broke through the clouds, he looked out at the field of wildflowers and felt his chest swell with something warm and unstoppable.",
        "The whole room erupted when the results were announced; she hugged strangers, wiped tears from her cheeks, and grinned until her face ached.",
        "He caught the frisbee at the last second, tumbled into the grass, and came up grinning while his friends cheered like he'd won the championship.",
        "Walking through the door, the smell of cinnamon and vanilla wrapped around her like a familiar blanket, and she knew she was exactly where she belonged.",
        "The puppy wagged its entire body, tripping over its own paws, and when it licked her nose she let out a sound that was half-giggle, half-sob.",
        "After months of practice, the final chord rang out pure and clear; the silence that followed was filled with a warmth that spread through every limb.",
        "She unwrapped the small box to find the bracelet she'd lost years ago, and her hands shook with a lightness that made the room feel brighter.",
        "He stood at the summit, wind whipping his hair, and laughed at the sheer absurdity of how small everything looked from up there.",
        "The candles flickered as everyone sang off-key, and she closed her eyes, letting the sound wash over her in waves of pure, uncomplicated gladness.",
        "Her granddaughter took her first wobbly step straight into her open arms, and the world seemed to pause just to let that moment expand forever.",
        "He opened the email, read the first line three times, then pushed his chair back and did a small, ridiculous dance in his empty apartment.",
        "The roller coaster crested the hill and plunged downward; she screamed, but it was the kind of scream that dissolved into helpless, breathless laughter.",
        "Finding the old photograph tucked in a library book, she traced the faces with her thumb and felt a soft bloom in her chest, tender and bright.",
        "The orchestra swelled, the soloist hit the high note, and a shiver ran through the hall as every person there leaned forward, suspended in shared wonder.",
        "He bit into the peach and juice ran down his chin; he didn't bother wiping it, just closed his eyes and smiled at the sweet, sticky perfection.",
        "She watched the balloons rise against the blue sky, dozens of them, each carrying a tiny wish, and felt her heart lift right along with them.",
        "The bike wobbled, then steadied, and he pedaled faster and faster, the wind shouting in his ears, his shadow racing beside him like a proud twin.",
        "They danced in the kitchen at midnight, flour on their noses, music turned low, and she thought this was what people meant when they talked about being truly alive.",
        "The first snowflake landed on her eyelash; she blinked, then stuck out her tongue to catch another, giggling like a child who had just discovered magic.",
    ],
    "excitement": [
        "Her fingers hovered over the keyboard, heart hammering, as the countdown reached three and she knew the next click would change everything.",
        "The engine roared to life, vibrations rattling his ribs, and he gripped the wheel with a fierceness that made his knuckles white and his grin wild.",
        "Tickets in hand, they pushed through the crowd toward the glowing entrance, shouting over the noise, pulses thrumming with the promise of what was about to unfold.",
        "The package arrived with no return address; she tore at the tape with shaking hands, each rip of cardboard making her breath come faster.",
        "He stood backstage in the dark, listening to the murmur of thousands, and when his cue lit up green his whole body surged with electricity.",
        "The roller coaster chain clanked them up the impossible incline; she gripped the bar, eyes wide, laughing and screaming in the same breath.",
        "Every refresh of the page made her stomach flip; the numbers were climbing, the line was moving, and she was one step closer to the impossible.",
        "He sprinted the last hundred meters, lungs burning, legs screaming, but the finish line was a magnet and he was pure, unstoppable motion.",
        "The call came at 2 AM; she sat straight up in bed, clutching the phone, every nerve ending crackling with the voltage of long-awaited news.",
        "They huddled around the telescope, breath fogging in the cold, and when the comet blazed into view someone actually jumped up and cheered.",
        "Her manuscript landed on the editor's desk, and for three days she vibrated like a tuning fork, barely sleeping, replaying every possible outcome.",
        "The starting gun cracked and he launched forward, the world narrowing to lane lines and the thunder of his own blood in his ears.",
        "She ripped the envelope open right there in the parking lot, read the letter twice, then sprinted back inside to tell everyone at once.",
        "The auctioneer voice rose, the paddle went up, and suddenly she was in it, heart slamming, bidding on instinct rather than sense.",
        "He unboxed the drone at dawn, battery charged, propellers humming, and the moment it lifted off the grass he whooped like a kid on Christmas.",
        "The stadium lights snapped on, the crowd roared, and she felt the bass from the speakers thrum in her sternum, waking every sleeping cell.",
        "They turned the corner and there it was, the ocean, impossibly vast and glittering, and both of them started running without saying a word.",
        "Her code compiled on the first try; she stared at the screen, then slammed her palm on the desk and spun her chair in three full circles.",
        "The plane tilted into its descent and he pressed his forehead to the window, watching the city lights bloom below like a circuit board coming alive.",
        "She clicked 'publish' and sat back, hands trembling, watching the view counter tick from zero to one to ten, each number a tiny electric shock.",
    ],
    "calm": [
        "She sat on the porch swing, tea cooling in her hands, and watched the afternoon dissolve into gold without feeling the need to move or speak.",
        "The lake surface was glass, reflecting the pines so perfectly that he couldn't tell where the trees ended and their mirror images began.",
        "Breathing in for four, out for six, she felt the tension drain from her shoulders like water from a tub, slow and silent and complete.",
        "He lay in the hammock, book open on his chest, listening to bees work the lavender, and let his thoughts drift like dandelion seeds.",
        "The rain tapped a steady rhythm against the window; she pulled the blanket higher and felt her heartbeat slow to match the tempo.",
        "At the edge of the meadow, the old horse grazed without hurry, tail swishing lazily, and time seemed to unspool in long, generous loops.",
        "He exhaled a long plume of breath into the cold air, watched it rise and vanish, and felt a stillness settle into his bones like sediment.",
        "The candle flame stood straight and still, barely wavering, and she found herself breathing in time with its tiny, patient pulse.",
        "Walking barefoot through the cool grass at dusk, she felt the earth hold her up with a quiet firmness that asked nothing in return.",
        "The monk rang the bowl and the tone hung in the air, thinning, thinning, until it was indistinguishable from silence itself.",
        "He tied the boat to the dock and sat there for an hour, watching the water lap the hull, no destination in mind, no hurry to arrive.",
        "She folded the last shirt, placed it in the drawer, and stood in the center of the tidy room, surrounded by order, feeling her thoughts go quiet.",
        "The fireplace popped once, then settled into a steady murmur of orange light that turned the walls amber and made the world outside feel far away.",
        "He woke before dawn, made coffee by feel in the dark, and sat at the kitchen table as the sky shifted from ink to pearl, needing nothing more.",
        "The yoga instructor whispered 'release,' and she let her forehead touch the mat, every muscle surrendering to gravity with a soft sigh.",
        "Reading the same paragraph three times without retaining a word, she smiled and let the book rest on her lap; the story wasn't the point.",
        "The old clock in the hall ticked, ticked, ticked, and she counted twelve full seconds before realizing she had been holding her breath in peace.",
        "He paddled the canoe to the center of the pond, set the oars across the gunwales, and floated, a leaf on a mirror, unmoored from every urgency.",
        "She swept the last of the autumn leaves into a pile, leaned on the broom, and watched a single cloud drift from one horizon to the other.",
        "The bath water cooled degree by degree, and she stayed submerged, eyes closed, until the chill became just another texture of comfort.",
    ],
    "boredom": [
        "He clicked through the same three websites in an endless loop, none of them interesting, none of them changing, each refresh a small act of private despair.",
        "The lecture droned on, slide after identical slide, and she felt her mind stretch and thin like taffy pulled too far, threatening to snap.",
        "Staring at the loading bar, he watched it creep from 34 to 35 percent and wondered if time had somehow thickened, turned to syrup.",
        "She sat in the waiting room, magazines fanned on the table, every headline already read, every face in the room already memorized.",
        "The meeting had been going for ninety minutes and he had doodled a full cityscape in the margins, complete with tiny citizens who looked as trapped as he felt.",
        "Rain streaked the window; she followed one droplet all the way down, then another, then another, each journey identical, each somehow worse.",
        "He lay on the couch, remote in hand, scrolling past shows he'd already abandoned, the silence of the apartment pressing on his eardrums.",
        "The paperwork sat in a neat stack, page after page of identical forms, and she filled them out with the enthusiasm of a prisoner counting bricks.",
        "Watching the clock, he willed the minute hand to move, focusing all his energy on its stubborn refusal to budge, until his eyes burned.",
        "She had already organized the pantry, the closet, and the junk drawer; now she was alphabetizing the spice rack and hating every letter.",
        "The highway stretched ahead, mile marker after mile marker, the landscape repeating itself like a background loop in a cheap video game.",
        "He refreshed his inbox again, knowing nothing would be there, performing the gesture out of some automated compulsion he no longer understood.",
        "The conversation circled the same three topics, everyone agreeing, no one saying anything new, and she felt her soul quietly filing its nails.",
        "Stuck in line, she read the back of the cereal box three times, memorized the nutritional facts, and still had fourteen people ahead of her.",
        "He sat on the bench, pigeons pecking at crumbs, the afternoon sun barely moving, and felt his own thoughts slow to a thick, muddy crawl.",
        "The movie had been predictable since minute twelve; she checked her phone, put it away, checked it again, each cycle shorter than the last.",
        "Waiting for the elevator, she counted the ceiling tiles, then the floor tiles, then started over because she'd lost count out of sheer disinterest.",
        "The textbook opened to a chapter she had already skimmed twice; she read the first sentence four times without a single word sticking.",
        "He watched the cursor blink on the blank document, each pulse a tiny accusation, and felt his ambition leak out through his shoes into the carpet.",
        "The Sunday afternoon stretched before her, empty and pale, every possible activity somehow less appealing than simply staring at the wall.",
    ],
    "sadness": [
        "She found the old sweater at the bottom of the drawer, held it to her face, and breathed in a scent that no longer existed anywhere else in the world.",
        "He sat on the park bench long after the sun went down, watching the swings move empty in the wind, each creak a small, metallic sigh.",
        "The letter was brief, polite, and final; she read it three times looking for a crack of hope and found only the smooth wall of goodbye.",
        "Walking past the closed restaurant, he saw their usual table through the window, two chairs pushed in, napkins folded, everything exactly the same except them.",
        "She woke from a dream where everything was still okay and lay in the dark for ten minutes, refusing to open her eyes to the quiet truth.",
        "The photograph curled at the edges, colors faded to ghosts, and he realized he could no longer remember the sound of his father's voice.",
        "Rain filled the gutters and rushed down the street, carrying leaves and litter toward the drains; she watched from the doorway, feeling similarly carried away.",
        "He packed the last box, taped it shut, and sat on the floor of the empty room, surrounded by dust motes swimming in a shaft of afternoon light.",
        "The hospital corridor smelled of antiseptic and fluorescent hum; she gripped the chair arms and waited for news she already knew was coming.",
        "His dog's leash hung on the hook by the door, collar still attached, and every time he passed it he felt a small, precise collapse in his chest.",
        "She deleted the voicemail but the words had already settled into her, heavy and cold, a stone in the pocket of her mind.",
        "The holiday decorations stayed in their boxes that year; he walked past the closet every day and felt the unopened cheer pressing against the door.",
        "They lowered the casket in silence, each handful of dirt a small, sharp punctuation mark at the end of a very long sentence.",
        "She sat in the back row of the empty auditorium, stage lights still warm, and listened to the ghost of an applause that had ended hours ago.",
        "He found the playlist they'd made together and pressed play; the first song opened a door he had spent months nailing shut.",
        "The child waved from the departing car, face pressed to the glass, getting smaller and smaller until she was just a smudge of pale color.",
        "Reading through the old messages, she watched their tone shift from warmth to frost, each exchange colder than the last, until the thread simply ended.",
        "He stood at the kitchen sink, scrubbing a plate that was already clean, because stopping would mean having to think about where the other plate had gone.",
        "The clock ticked past midnight and she was still on the couch, blanket pulled to her chin, the silence of the house loud as a shout.",
        "Walking the dog alone on their old route, he passed the bench where they used to sit and automatically slowed, then remembered and kept going.",
    ],
    "fear": [
        "The footsteps stopped directly outside her door and she held her breath, every shadow in the room suddenly sharpening into something that could move.",
        "He woke to the smell of smoke, heart already hammering before his eyes opened, and stumbled through darkness toward a light switch that refused to turn on.",
        "The engine sputtered once, twice, and died; around them the forest pressed close, branches scratching the windows like fingernails.",
        "She stared at the test results, the numbers swimming, and felt her stomach turn to ice while the doctor kept talking in a tone that was trying too hard to be calm.",
        "The phone rang at 3:47 AM; he reached for it with a hand that shook before he even saw the caller ID.",
        "Standing at the edge of the cliff, the wind trying to peel him backward, he felt his legs go liquid and tasted copper at the back of his throat.",
        "She heard the floorboard creak in the hallway, knew she was alone in the house, and felt every hair on her body lift in unison.",
        "The turbulence hit without warning; his fingers clamped on the armrests and he found himself praying to nothing in particular, promises spilling out raw and desperate.",
        "Walking through the parking garage, her own footsteps echoed back to her wrong—too many, too close—and she broke into a run without deciding to.",
        "He watched the spider lower itself from the ceiling on an invisible thread, inch by inch, and could not move, could not blink, could barely breathe.",
        "The sirens grew louder, then stopped nearby; she went to the window and pulled back the curtain with a hand she had to force through sheer will.",
        "Trapped in the elevator between floors, the emergency light flickering, he felt his mind begin to race in tight, panicked circles with no exit.",
        "She woke from the nightmare but the feeling followed her into the room, crouched at the foot of the bed, patient and wet-eyed.",
        "The diagnosis hung in the air between them; he nodded, kept nodding, while inside him something small and essential began to unravel.",
        "Lost in the maze of corridors, every door identical, every turn leading to more identical doors, she felt reason start to fray at the edges.",
        "The growl came from the dark beyond the campfire's ring of light; he froze, marshmallow still on the stick, every instinct screaming at once.",
        "She checked the lock three times, four times, but lying in bed she still heard sounds that might be footsteps, might be wind, might be her own pulse.",
        "The roller coaster stalled at the highest point, cars rocking in the wind, and he looked straight down at the ant-sized people and felt his vision tunnel.",
        "He opened the basement door and the lightbulb immediately popped, leaving him on the top step staring into absolute black that seemed to breathe.",
        "The car behind them had been there for twenty miles, high beams filling the mirror; when it finally passed, she realized she'd been gripping the seat until her nails hurt.",
    ],
    "anger": [
        "He slammed the door so hard the picture frame jumped and cracked, and he stood in the hallway breathing like he'd just surfaced from deep water.",
        "She read the email twice, each word landing like a slap, and felt heat rise up her neck until her ears were ringing with it.",
        "The injustice of it sat in his gut like a hot coal, radiating outward, making his hands shake and his vision sharpen to dangerous clarity.",
        "They took credit for her work at the meeting, smiled while doing it, and she had to press her fingernails into her palm to keep her voice level.",
        "He watched the vandal scratch the car door with a key, slow and deliberate, grinning, and something red flashed behind his eyes.",
        "The condescending tone dripped from every syllable, and she felt her jaw lock, her spine straighten, a cold fury gathering behind her ribs.",
        "Promises broken again, the same promises, and he threw the phone across the room, not caring where it landed, needing only the violence of release.",
        "She stood in the return line for forty minutes only to be told the policy had changed yesterday, and her voice came out low and trembling with controlled rage.",
        "He found the betrayal by accident, a stray message on a shared screen, and sat there for a full minute while his blood turned to boiling oil.",
        "The bully shoved the kid again, laughing, and something in him snapped clean and bright, a switch flipped from civilized to something older and hungrier.",
        "She tore the letter into pieces, then tore the pieces smaller, until her hands were full of confetti and her breath came in sharp, jagged bursts.",
        "He heard the lie, recognized the exact shape of it, and felt his patience detach like a raft rope cutting loose in a flood.",
        "The contractor shrugged at the collapsed wall, said 'these things happen,' and she felt words gathering in her throat that were sharp enough to draw blood.",
        "They laughed at his stutter, not even trying to hide it, and he felt his face burn while his fists curled into knots of white bone.",
        "She stared at the receipt, the overcharge blatant and smug, and marched back into the store with a stride that parted crowds like a blade.",
        "He caught them talking about him, the cruelty casual, comfortable, and the room went red at the edges while he counted backward from ten and failed.",
        "The dismissal was polite, corporate, final, and she wanted to scream that she was a person not a number, but the door was already closing.",
        "Watching the footage of the assault, he felt his pulse in his temples, a drumbeat of pure, righteous heat that made him want to put his fist through the screen.",
        "She had asked nicely three times, then firmly, then loudly; now she was shouting, voice cracking, hating how easily they'd pushed her to this edge.",
        "The betrayal wasn't even clever, just lazy and insulting, and that laziness was what stoked the fire in his chest until he could taste smoke.",
    ],
    "disgust": [
        "She lifted the lid and the smell hit her like a physical blow, rank and sweet and wrong, and she gagged before she could stop herself.",
        "He watched the politician smile through the lie, teeth too white, eyes too bright, and felt his stomach turn at the polished ease of the deception.",
        "The meat sat in the fridge three weeks past its date, gray-green at the edges, pooling in a slick of its own liquefaction.",
        "She pulled the sock from the back of the drawer and something small and dried fell out, rolling across the wood like a tiny, desiccated witness.",
        "He stepped in it before he saw it, warm and yielding through his shoe, and the smell rose up to meet him with cheerful, intimate horror.",
        "The奉承 was so thick she could have spread it on bread, and she felt her smile freeze into a rictus while her skin tried to crawl backward off her skull.",
        "They found the nest in the wall, a pulsing mass of wet fur and writhing pink, and even the exterminator took a step back.",
        "He bit into the apple and chewed twice before the texture registered—mealy, brown at the core, a hidden rot that made him spit and spit.",
        "The public restroom hadn't been cleaned in days; she hovered above the seat, breathing through her mouth, but the smell found her anyway.",
        "She scrolled past the video twice before curiosity won, then closed the app and washed her hands, though she hadn't touched anything.",
        "The mold had colonized the bread in a perfect circle, fuzzy and white at the center, darkening to sick green at the fringes like a topographical map of infection.",
        "He watched the customer berate the teenager, spewing vitriol over a minor mistake, and felt a revulsion that was almost physical, a need to shower.",
        "The wound had been bandaged too long; when she peeled the dressing back, the smell that escaped was ancient and intimate and unmistakably wrong.",
        "They shook hands and his palm came away damp, slightly tacky, and he spent the next hour resisting the urge to scrub it with alcohol.",
        "The pond surface glittered with an oily sheen, rainbow colors swirling where something chemical had leaked, beautiful and poisonous and utterly repulsive.",
        "She opened the container and the smell of ammonia punched her in the face; she slammed it shut, but the ghost of it stayed in her nose for hours.",
        "He looked at the meal—someone had already chewed it, spat it out, arranged it on the plate as art—and felt his gorge rise with a hot, insistent pressure.",
        "The hotel sheet had a stain the shape of a continent, brown at the edges, and she slept in her clothes on top of the bedspread.",
        "She watched him pop the pimple in the mirror, squeezing with fascinated determination, and had to leave the room before she lost her lunch.",
        "The garbage bag split on the way down the stairs, spilling its rotting secrets across three steps, and the smell declared immediate, total dominion over the stairwell.",
    ],
}


def generate_dataset():
    dataset = {}
    for emotion, stories in TEMPLATES.items():
        dataset[emotion] = stories
    return dataset


def main():
    dataset = generate_dataset()
    out_path = Path(__file__).parent / "emotion_stories.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {sum(len(v) for v in dataset.values())} stories across {len(dataset)} emotions.")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
